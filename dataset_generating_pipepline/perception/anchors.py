"""Merge Layer-0 detectors into one sorted ``anchors.json`` timeline.

v1: classic onsets + scenes + speech (schema 2.0.0).
v2: onsets_v2 + SED + video_cues_v2 fusion (schema 2.1.0, additive fields).

Caching: artifacts live under ``features/<vid>/<clip>/perception/`` and are
keyed by ``clip_sha256`` + ``separation_used`` (+ v2 knobs) so changing the
clip or toggling Demucs / SED / onsets_v2 invalidates the cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.schemas import PipelineConfig, PerceptionConfig, SnapConfig

LOG = logging.getLogger("visualmellow.perception.anchors")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data: Any) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_anchors(
    clip_path: Path | str,
    cfg: PipelineConfig,
    *,
    out_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full Layer-0 stack and return (optionally cache) the anchor doc."""
    clip_path = Path(clip_path)
    perc: PerceptionConfig = cfg.perception
    use_separation = bool(cfg.flags.separation)
    anchor_version = int(getattr(cfg, "anchor_version", None) or perc.anchor_version or 1)
    sed_backend = perc.sed.backend if perc.sed.enabled else "off"

    clip_sha = _hash_file(clip_path)
    if out_dir is not None and not force:
        cached = _try_load_cache(
            out_dir, clip_sha, use_separation,
            anchor_version=anchor_version,
            onsets_v2=bool(perc.onsets_v2.enabled and anchor_version >= 2),
            sed_backend=sed_backend if anchor_version >= 2 else "off",
        )
        if cached is not None:
            LOG.info("Reusing cached anchors (%d entries) for %s", len(cached.get("anchors") or []), clip_path.name)
            return cached

    if anchor_version >= 2 and perc.onsets_v2.enabled:
        return _build_anchors_v2(clip_path, cfg, out_dir=out_dir, clip_sha=clip_sha, force=force)
    return _build_anchors_v1(clip_path, cfg, out_dir=out_dir, clip_sha=clip_sha)


def load_or_build_anchors(
    clip_path: Path | str,
    cfg: PipelineConfig,
    out_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper that always caches under ``out_dir``."""
    return build_anchors(clip_path, cfg, out_dir=out_dir, force=force)


def _build_anchors_v1(
    clip_path: Path,
    cfg: PipelineConfig,
    *,
    out_dir: Path | None,
    clip_sha: str,
) -> dict[str, Any]:
    from perception.audio_io import decode_mono
    from perception.onsets import detect_onsets
    from perception.beats import detect_beats, flag_on_beat
    from perception.tagger import tag_audio, tags_near
    from perception.scenes import extract_visual_anchors, visual_anchors_as_dicts
    from perception.speech import transcribe_words, speech_anchors
    from perception.separation import separate_stems, save_stems

    perc = cfg.perception
    snap = cfg.snap
    use_separation = bool(cfg.flags.separation)

    t0 = time.perf_counter()
    audio = decode_mono(clip_path, sample_rate=perc.sample_rate)
    duration_ms = int(round(len(audio) / perc.sample_rate * 1000))

    stems = None
    separation_used = False
    if use_separation:
        stems = separate_stems(audio, sample_rate=perc.sample_rate)
        separation_used = stems is not None
        if not separation_used:
            LOG.warning("Separation requested but unavailable; using mixture + beat-grid filter")

    onset_kwargs = dict(
        hop_ms=perc.onset_hop_ms,
        delta=perc.onset_delta,
        min_gap_ms=perc.onset_min_gap_ms,
    )
    if separation_used and stems is not None:
        mixture_onsets = detect_onsets(stems["mixture"], perc.sample_rate, stem="mixture", **onset_kwargs)
        effects_onsets = detect_onsets(stems["effects"], perc.sample_rate, stem="effects", **onset_kwargs)
        dialogue_onsets = detect_onsets(stems["dialogue"], perc.sample_rate, stem="dialogue", **onset_kwargs)
        music_onsets = detect_onsets(stems["music"], perc.sample_rate, stem="music", **onset_kwargs)
        all_onsets = effects_onsets + dialogue_onsets + music_onsets
        stem_times = {o["t_ms"] for o in all_onsets}
        for o in mixture_onsets:
            if all(abs(o["t_ms"] - t) > snap.dedupe_ms for t in stem_times):
                all_onsets.append(o)
    else:
        all_onsets = detect_onsets(audio, perc.sample_rate, stem="mixture", **onset_kwargs)

    beat_info = detect_beats(audio, perc.sample_rate)
    all_onsets = flag_on_beat(all_onsets, beat_info["beats_ms"], tol_ms=50)
    tag_result = tag_audio(audio, perc.sample_rate, topk=perc.tagger_topk)
    visual = extract_visual_anchors(clip_path, fps=perc.anchor_fps)
    visual_dicts = visual_anchors_as_dicts(visual)
    cuts = [v for v in visual_dicts if v["kind"] == "cut"]
    motions = [v for v in visual_dicts if v["kind"] == "motion"]
    asr = transcribe_words(audio, perc.sample_rate, model_name=perc.whisper_model)
    speech = speech_anchors(asr)
    music_boundaries = _music_boundaries(
        stems["music"] if separation_used and stems is not None else audio,
        perc.sample_rate,
    )

    raw_events: list[dict[str, Any]] = []
    for o in all_onsets:
        raw_events.append({
            "t_ms": int(o["t_ms"]),
            "kind": "audio_onset",
            "stem": o.get("stem", "mixture"),
            "strength": float(o.get("strength") or 0.0),
            "on_music_beat": bool(o.get("on_music_beat")),
            "tagger_topk": tags_near(tag_result, int(o["t_ms"])),
        })
    for c in cuts:
        raw_events.append({
            "t_ms": int(c["t_ms"]), "kind": "cut", "stem": "visual",
            "strength": float(c.get("strength") or 0.0),
        })
    for m in motions:
        raw_events.append({
            "t_ms": int(m["t_ms"]), "kind": "motion", "stem": "visual",
            "strength": float(m.get("strength") or 0.0),
        })
    for s in speech:
        raw_events.append(s)
    for b in music_boundaries:
        raw_events.append(b)

    anchors = _dedupe_and_enrich(raw_events, cuts, motions, snap)
    for i, a in enumerate(anchors):
        a["id"] = f"a{i:03d}"

    elapsed = time.perf_counter() - t0
    doc = {
        "schema_version": "2.0.0",
        "anchor_version": 1,
        "clip_path": str(clip_path),
        "clip_sha256": clip_sha,
        "clip_duration_ms": duration_ms,
        "separation_used": separation_used,
        "sample_rate": perc.sample_rate,
        "tempo_bpm": beat_info.get("tempo_bpm"),
        "beats_ms": beat_info.get("beats_ms") or [],
        "tagger": {"available": tag_result.get("available", False), "clip_topk": tag_result.get("clip_topk") or []},
        "asr": {
            "available": asr.get("available", False),
            "language": asr.get("language"),
            "segments": asr.get("segments") or [],
            "word_count": len(asr.get("words") or []),
        },
        "counts": _counts(anchors),
        "anchors": anchors,
        "elapsed_s": round(elapsed, 3),
        "created_at": _utc_now(),
    }
    return _finalize(doc, out_dir, stems, perc.sample_rate, separation_used, elapsed)


def _build_anchors_v2(
    clip_path: Path,
    cfg: PipelineConfig,
    *,
    out_dir: Path | None,
    clip_sha: str,
    force: bool = False,
) -> dict[str, Any]:
    from perception.audio_io import decode_mono, decode_dsp
    from perception.onsets_v2 import detect_onsets_v2
    from perception.beats import detect_beats, flag_on_beat
    from perception.tagger import tag_audio, tags_near
    from perception.speech import transcribe_words, speech_anchors
    from perception.separation import separate_stems, save_stems
    from perception.timebase import build_timebase_doc
    from perception.avsync import estimate_av_offset

    perc = cfg.perception
    snap = cfg.snap
    use_separation = bool(cfg.flags.separation)
    ov2 = perc.onsets_v2

    t0 = time.perf_counter()
    timebase = build_timebase_doc(clip_path, perception_dir=out_dir, extract_pts=True)
    audio16 = decode_mono(clip_path, sample_rate=perc.sample_rate)
    audio32 = decode_dsp(clip_path, sample_rate=ov2.sr_dsp)
    duration_ms = int(round(len(audio16) / perc.sample_rate * 1000))

    stems16 = None
    stems32 = None
    separation_used = False
    if use_separation:
        stems16 = separate_stems(audio16, sample_rate=perc.sample_rate)
        separation_used = stems16 is not None
        if separation_used and stems16 is not None:
            # Resample stems to DSP rate for onset v2.
            import librosa
            stems32 = {
                k: librosa.resample(v.astype("float32"), orig_sr=perc.sample_rate, target_sr=ov2.sr_dsp)
                if k != "mixture" else librosa.resample(
                    (stems16.get("mixture") if stems16.get("mixture") is not None else audio16).astype("float32"),
                    orig_sr=perc.sample_rate, target_sr=ov2.sr_dsp,
                )
                for k, v in stems16.items()
            }
            stems32["mixture"] = librosa.resample(audio16.astype("float32"), orig_sr=perc.sample_rate, target_sr=ov2.sr_dsp)
        else:
            LOG.warning("Separation requested but unavailable; mixture-only onsets v2")

    ov2_cfg = ov2.as_dict()
    if separation_used and stems32 is not None:
        effects = detect_onsets_v2(stems32["effects"], ov2.sr_dsp, stem="effects", cfg=ov2_cfg)
        dialogue = detect_onsets_v2(stems32["dialogue"], ov2.sr_dsp, stem="dialogue", cfg=ov2_cfg)
        music = detect_onsets_v2(stems32["music"], ov2.sr_dsp, stem="music", cfg=ov2_cfg)
        mixture = detect_onsets_v2(stems32["mixture"], ov2.sr_dsp, stem="mixture", cfg=ov2_cfg)
        all_onsets = effects + dialogue + music
        stem_times = [float(o["t_ms_f"]) for o in all_onsets]
        for o in mixture:
            if all(abs(float(o["t_ms_f"]) - t) > 20 for t in stem_times):
                all_onsets.append(o)
    else:
        all_onsets = detect_onsets_v2(audio32, ov2.sr_dsp, stem="mixture", cfg=ov2_cfg)

    beat_info = detect_beats(audio16, perc.sample_rate)
    # flag_on_beat expects t_ms
    all_onsets = flag_on_beat(all_onsets, beat_info["beats_ms"], tol_ms=50)

    tag_result = tag_audio(audio16, perc.sample_rate, topk=perc.tagger_topk)
    for o in all_onsets:
        o["tagger_topk"] = tags_near(tag_result, int(o["t_ms"]))

    # SED
    sed_doc: dict[str, Any] = {"available": False, "events": [], "backend": "off"}
    if perc.sed.enabled:
        try:
            from perception.sed.infer import run_sed

            sed_audio = audio32
            if separation_used and stems32 is not None and "effects" in stems32:
                sed_audio = stems32["effects"]
            sed_doc = run_sed(
                sed_audio,
                ov2.sr_dsp,
                backend_name=perc.sed.backend,
                out_dir=out_dir,
                clip_sha=clip_sha,
                separation_used=separation_used,
                onsets=all_onsets,
                force=force,
                cfg=perc.sed.as_dict(),
            )
        except Exception as exc:
            LOG.warning("SED failed: %s", exc)

    # Speech + music boundaries
    asr = transcribe_words(audio16, perc.sample_rate, model_name=perc.whisper_model)
    speech = speech_anchors(asr)
    music_boundaries = _music_boundaries(
        stems16["music"] if separation_used and stems16 is not None else audio16,
        perc.sample_rate,
    )

    # Video cues v2
    video_doc: dict[str, Any] = {"anchors": [], "motion_times_ms": [], "motion_energy": [], "available": False}
    if perc.video_cues_v2.enabled and cfg.flags.video_cues_v2:
        try:
            from perception.video_cues import extract_video_cues_v2
            import numpy as np

            pts = None
            if out_dir is not None:
                pts_path = Path(out_dir) / "frame_pts_s.npy"
                if pts_path.is_file():
                    pts = np.load(pts_path)
            video_doc = extract_video_cues_v2(
                clip_path,
                frame_pts_s=pts,
                use_farneback=perc.video_cues_v2.use_farneback,
                use_transnet=perc.video_cues_v2.use_transnet,
                min_intensity_change=float(
                    getattr(perc.video_cues_v2, "min_intensity_change", 5.0) or 5.0
                ),
            )
        except Exception as exc:
            LOG.warning("video_cues_v2 failed (%s); falling back to scenes.py", exc)
            from perception.scenes import extract_visual_anchors, visual_anchors_as_dicts

            visual = extract_visual_anchors(clip_path, fps=perc.anchor_fps)
            video_doc = {"anchors": visual_anchors_as_dicts(visual), "available": True,
                         "motion_times_ms": [], "motion_energy": []}

    avsync = estimate_av_offset(
        [float(o.get("t_ms_f", o["t_ms"])) for o in all_onsets],
        [float(o.get("strength_norm") or o.get("strength") or 0) for o in all_onsets],
        video_doc.get("motion_times_ms") or [],
        video_doc.get("motion_energy") or [],
    )
    if cfg.apply_av_offset and avsync.get("available") and abs(float(avsync.get("av_offset_ms") or 0)) > 1:
        delta = float(avsync["av_offset_ms"])
        LOG.warning("Applying A/V offset %+0.1f ms to audio anchors", delta)
        for o in all_onsets:
            o["t_ms_f"] = float(o.get("t_ms_f", o["t_ms"])) + delta
            o["t_ms"] = int(round(o["t_ms_f"]))
            if o.get("t_end_ms_f") is not None:
                o["t_end_ms_f"] = float(o["t_end_ms_f"]) + delta

    anchors = _fuse_v2(
        onsets=all_onsets,
        sed_events=sed_doc.get("events") or [],
        speech=speech,
        music_boundaries=music_boundaries,
        video_anchors=video_doc.get("anchors") or [],
        fusion=perc.fusion,
        snap=snap,
    )
    for i, a in enumerate(anchors):
        a["id"] = f"a{i:03d}"
        if not a.get("tagger_topk"):
            a["tagger_topk"] = tags_near(tag_result, int(a["t_ms"]))

    elapsed = time.perf_counter() - t0
    snrs = [float(a["snr_db"]) for a in anchors if a.get("snr_db") is not None]
    doc = {
        "schema_version": "2.1.0",
        "anchor_version": 2,
        "clip_path": str(clip_path),
        "clip_sha256": clip_sha,
        "clip_duration_ms": duration_ms,
        "separation_used": separation_used,
        "onsets_v2": True,
        "sed_backend": sed_doc.get("backend"),
        "sample_rate": perc.sample_rate,
        "sample_rate_dsp": ov2.sr_dsp,
        "tempo_bpm": beat_info.get("tempo_bpm"),
        "beats_ms": beat_info.get("beats_ms") or [],
        "timebase": timebase,
        "av_offset_ms": avsync.get("av_offset_ms"),
        "av_offset_confidence": avsync.get("av_offset_confidence"),
        "tagger": {"available": tag_result.get("available", False), "clip_topk": tag_result.get("clip_topk") or []},
        "asr": {
            "available": asr.get("available", False),
            "language": asr.get("language"),
            "segments": asr.get("segments") or [],
            "word_count": len(asr.get("words") or []),
        },
        "sed": {
            "available": sed_doc.get("available", False),
            "backend": sed_doc.get("backend"),
            "n_events": sed_doc.get("n_events", 0),
            "native_resolution_s": sed_doc.get("native_resolution_s"),
        },
        "mean_snr_db": (sum(snrs) / len(snrs)) if snrs else None,
        "counts": _counts(anchors),
        "anchors": anchors,
        "elapsed_s": round(elapsed, 3),
        "created_at": _utc_now(),
    }
    return _finalize(doc, out_dir, stems16, perc.sample_rate, separation_used, elapsed)


def _fuse_v2(
    *,
    onsets: list[dict[str, Any]],
    sed_events: list[dict[str, Any]],
    speech: list[dict[str, Any]],
    music_boundaries: list[dict[str, Any]],
    video_anchors: list[dict[str, Any]],
    fusion,
    snap: SnapConfig,
) -> list[dict[str, Any]]:
    """Cluster audio candidates and attach visual support."""
    audio_cands: list[dict[str, Any]] = []
    for o in onsets:
        audio_cands.append(dict(o))
    for e in sed_events:
        audio_cands.append(dict(e))
    for s in speech:
        item = dict(s)
        item.setdefault("t_ms_f", float(item.get("t_ms") or 0))
        item.setdefault("timing_source", "asr")
        audio_cands.append(item)
    for b in music_boundaries:
        item = dict(b)
        item.setdefault("t_ms_f", float(item.get("t_ms") or 0))
        audio_cands.append(item)

    cluster_ms = float(getattr(fusion, "audio_cluster_ms", 40))
    audio_cands.sort(key=lambda e: (float(e.get("t_ms_f", e.get("t_ms") or 0)), -_rank(e)))

    merged: list[dict[str, Any]] = []
    for e in audio_cands:
        t = float(e.get("t_ms_f", e.get("t_ms") or 0))
        if merged and abs(t - float(merged[-1].get("t_ms_f", merged[-1].get("t_ms") or 0))) <= cluster_ms:
            merged[-1] = _prefer_merge(merged[-1], e)
            continue
        merged.append(dict(e))

    # Attach visual support within ±visual_attach_ms; keep video-only separately.
    vis_tol = float(getattr(fusion, "visual_attach_ms", 100))
    visual_only: list[dict[str, Any]] = []
    used_visual = set()
    for vi, v in enumerate(video_anchors):
        vt = float(v.get("t_ms_f", v.get("t_ms") or 0))
        best_i = None
        best_d = None
        for i, a in enumerate(merged):
            at = float(a.get("t_ms_f", a.get("t_ms") or 0))
            d = abs(at - vt)
            if d <= vis_tol and (best_d is None or d < best_d):
                best_d, best_i = d, i
        kind = v.get("kind")
        if best_i is not None:
            a = merged[best_i]
            vs = dict(a.get("visual_support") or {})
            if kind in {"motion_stop", "motion_spike", "motion"}:
                vs["motion_stop_dt_ms" if kind == "motion_stop" else "motion_spike_dt_ms"] = int(round(
                    float(a.get("t_ms_f", a["t_ms"])) - vt
                ))
            if kind == "cut":
                vs["cut_dt_ms"] = int(round(float(a.get("t_ms_f", a["t_ms"])) - vt))
            if kind == "intensity_change":
                vs["intensity_change_dt_ms"] = int(round(float(a.get("t_ms_f", a["t_ms"])) - vt))
                vs["intensity_delta"] = v.get("delta")
                vs["intensity_direction"] = v.get("direction")
            a["visual_support"] = vs
            src = list(a.get("sources") or [])
            src.append(str(kind))
            a["sources"] = sorted(set(src))
            used_visual.add(vi)
        else:
            # Skip duplicate motion aliases for visual-only list (keep motion_spike not motion).
            if kind == "motion":
                continue
            item = dict(v)
            item["timing_source"] = "visual"
            # Intensity changes are a strong visual cue (notebook-validated); keep them
            # as first-class visual anchors for the LLM, not low-confidence fillers.
            if kind != "intensity_change":
                item["low_confidence"] = True
            visual_only.append(item)

    # Score audio merged anchors.
    for a in merged:
        a["score"] = _score_anchor(a, fusion)
        a.setdefault("t_ms", int(round(float(a.get("t_ms_f", a.get("t_ms") or 0)))))
        a.setdefault("t_ms_f", float(a["t_ms"]))
        a.setdefault("on_music_beat", False)
        a.setdefault("tagger_topk", [])
        a.setdefault("sources", _default_sources(a))

    # Near cut/motion enrich from video list.
    cut_times = [int(v.get("t_ms") or 0) for v in video_anchors if v.get("kind") == "cut"]
    motion_times = [int(v.get("t_ms") or 0) for v in video_anchors if v.get("kind") in {"motion", "motion_spike", "motion_stop"}]
    for a in merged + visual_only:
        t = int(a.get("t_ms") or 0)
        a["near_cut_ms"] = _nearest_time(cut_times, t, snap.near_cut_tol_ms)
        a["near_motion_peak_ms"] = _nearest_time(motion_times, t, snap.near_cut_tol_ms)

    all_anchors = merged + visual_only
    all_anchors.sort(key=lambda a: float(a.get("t_ms_f", a.get("t_ms") or 0)))
    return all_anchors


def _rank(e: dict[str, Any]) -> float:
    kind = e.get("kind")
    # speech > sed_event (prob>0.5) > audio_onset > music_boundary > cut > motion*
    if kind == "speech_start":
        return 50
    if kind == "sed_event" and float(e.get("prob") or 0) > 0.5:
        return 40 + float(e.get("prob") or 0)
    if kind == "audio_onset":
        return 30 + float(e.get("strength_norm") or 0)
    if kind == "sed_event":
        return 25
    if kind == "music_boundary":
        return 20
    if kind == "cut":
        return 10
    return 0


def _prefer_merge(prev: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    if _rank(new) > _rank(prev):
        combined = dict(new)
        for key in ("transcript", "end_ms", "tagger_topk", "on_music_beat", "bands", "sed",
                    "t_end_ms_f", "end_reliable", "snr_db", "rise_time_ms", "sources", "visual_support"):
            if key not in combined and key in prev:
                combined[key] = prev[key]
        # Keep SED hints even when onset wins.
        if prev.get("kind") == "sed_event" and not combined.get("sed"):
            combined["sed"] = prev.get("sed") or [{
                "class": prev.get("class"), "prob": prev.get("prob"), "group": prev.get("group"),
            }]
            src = list(combined.get("sources") or [])
            src.append("sed")
            combined["sources"] = sorted(set(src))
        return combined
    for key in ("transcript", "end_ms", "tagger_topk", "on_music_beat", "bands", "sed",
                "t_end_ms_f", "end_reliable", "snr_db", "sources", "visual_support"):
        if key not in prev and key in new:
            prev[key] = new[key]
    if new.get("kind") == "sed_event":
        prev["sed"] = new.get("sed") or [{
            "class": new.get("class"), "prob": new.get("prob"), "group": new.get("group"),
        }]
        src = list(prev.get("sources") or [])
        src.append("sed")
        prev["sources"] = sorted(set(src))
    return prev


def _score_anchor(a: dict[str, Any], fusion) -> float:
    onset_n = float(a.get("strength_norm") or 0)
    if not onset_n and a.get("strength"):
        onset_n = min(1.0, float(a["strength"]) / 10.0)
    sed_p = 0.0
    if a.get("sed"):
        sed_p = max(float(x.get("prob") or 0) for x in a["sed"])
    elif a.get("kind") == "sed_event":
        sed_p = float(a.get("prob") or 0)
    snr = float(a.get("snr_db") or 0)
    snr_n = max(0.0, min(1.0, snr / 20.0))
    vis = 1.0 if a.get("visual_support") else 0.0
    beat_pen = 0.0 if not a.get("on_music_beat") else 1.0  # small penalty term in plan
    return float(
        getattr(fusion, "w_onset", 0.35) * onset_n
        + getattr(fusion, "w_sed", 0.30) * sed_p
        + getattr(fusion, "w_snr", 0.15) * snr_n
        + getattr(fusion, "w_visual", 0.15) * vis
        + getattr(fusion, "w_beat", 0.05) * (1.0 - beat_pen)
    )


def _default_sources(a: dict[str, Any]) -> list[str]:
    src = []
    if a.get("kind") == "audio_onset":
        bands = a.get("bands") or []
        src.extend([f"superflux_{b}" for b in bands] or ["superflux"])
    if a.get("kind") == "sed_event" or a.get("sed"):
        src.append("sed")
    if a.get("kind") == "speech_start":
        src.append("asr")
    return sorted(set(src))


def _nearest_time(times: list[int], t: int, tol: int) -> int | None:
    if not times:
        return None
    best = min(times, key=lambda x: abs(x - t))
    return best if abs(best - t) <= tol else None


def _counts(anchors: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "audio_onset", "sed_event", "cut", "motion", "motion_spike", "motion_stop",
        "flash", "intensity_change", "speech_start", "music_boundary",
    )
    out = {k: sum(1 for a in anchors if a.get("kind") == k) for k in keys}
    out["total"] = len(anchors)
    return out


def _finalize(doc, out_dir, stems, sample_rate, separation_used, elapsed):
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(out_dir / "anchors.json", doc)
        if separation_used and stems is not None:
            try:
                from perception.separation import save_stems
                save_stems(stems, out_dir / "stems", sample_rate=sample_rate)
            except Exception as exc:
                LOG.warning("Could not save stem WAVs: %s", exc)
        LOG.info("Anchors written: %d total (%.1fs) → %s", len(doc["anchors"]), elapsed, out_dir / "anchors.json")
    else:
        LOG.info("Anchors built: %d total (%.1fs)", len(doc["anchors"]), elapsed)
    return doc


def _try_load_cache(
    out_dir: Path,
    clip_sha: str,
    separation_used: bool,
    *,
    anchor_version: int = 1,
    onsets_v2: bool = False,
    sed_backend: str = "off",
) -> dict[str, Any] | None:
    path = Path(out_dir) / "anchors.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("clip_sha256") != clip_sha:
            return None
        if bool(doc.get("separation_used")) != bool(separation_used):
            return None
        if int(doc.get("anchor_version") or (2 if str(doc.get("schema_version", "")).startswith("2.1") else 1)) != int(anchor_version):
            return None
        if anchor_version >= 2:
            if bool(doc.get("onsets_v2")) != bool(onsets_v2):
                return None
            if (doc.get("sed_backend") or "off") != (sed_backend or "off"):
                return None
        if not isinstance(doc.get("anchors"), list):
            return None
        return doc
    except Exception:
        return None


def _music_boundaries(audio, sample_rate: int, hop_s: float = 0.5) -> list[dict[str, Any]]:
    """MFCC self-similarity novelty → coarse music-section boundaries."""
    import numpy as np
    import librosa

    try:
        hop = max(1, int(round(sample_rate * hop_s)))
        mf = librosa.feature.mfcc(y=audio.astype(np.float32), sr=sample_rate, n_mfcc=20, hop_length=hop)
        mf = (mf - mf.mean(1, keepdims=True)) / (mf.std(1, keepdims=True) + 1e-6)
        if mf.shape[1] < 3:
            return []
        d = np.linalg.norm(np.diff(mf, axis=1), axis=0)
        thr = float(d.mean() + 1.0 * d.std())
        out = []
        last_t = -9999.0
        for i, score in enumerate(d):
            t_s = (i + 1) * hop_s
            if score < thr:
                continue
            left = d[i - 1] if i > 0 else 0.0
            right = d[i + 1] if i + 1 < len(d) else 0.0
            if score < left or score < right:
                continue
            if t_s - last_t < 2.0:
                continue
            out.append({
                "t_ms": int(round(t_s * 1000)),
                "t_ms_f": round(t_s * 1000, 1),
                "kind": "music_boundary",
                "stem": "music",
                "strength": float(score),
            })
            last_t = t_s
        return out
    except Exception as exc:
        LOG.warning("Music-boundary detection failed: %s", exc)
        return []


def _dedupe_and_enrich(
    events: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
    motions: list[dict[str, Any]],
    snap: SnapConfig,
) -> list[dict[str, Any]]:
    """v1 dedupe within ``dedupe_ms``, attach nearest cut/motion, sort by time."""
    if not events:
        return []
    events = sorted(events, key=lambda e: (int(e["t_ms"]), -float(e.get("strength") or 0.0)))
    merged: list[dict[str, Any]] = []
    for e in events:
        if merged and abs(int(e["t_ms"]) - int(merged[-1]["t_ms"])) <= snap.dedupe_ms:
            prev = merged[-1]
            prefer_new = False
            rank = {"speech_start": 3, "audio_onset": 2, "music_boundary": 2, "cut": 1, "motion": 0}
            if rank.get(e["kind"], 0) > rank.get(prev["kind"], 0):
                prefer_new = True
            elif rank.get(e["kind"], 0) == rank.get(prev["kind"], 0) and float(e.get("strength") or 0) > float(prev.get("strength") or 0):
                prefer_new = True
            if prefer_new:
                combined = dict(e)
                for key in ("transcript", "end_ms", "tagger_topk", "on_music_beat", "word_count"):
                    if key not in combined and key in prev:
                        combined[key] = prev[key]
                merged[-1] = combined
            else:
                for key in ("transcript", "end_ms", "tagger_topk", "on_music_beat"):
                    if key not in prev and key in e:
                        prev[key] = e[key]
            continue
        merged.append(dict(e))

    cut_times = [int(c["t_ms"]) for c in cuts]
    motion_times = [int(m["t_ms"]) for m in motions]
    for a in merged:
        t = int(a["t_ms"])
        a["near_cut_ms"] = _nearest_time(cut_times, t, snap.near_cut_tol_ms)
        a["near_motion_peak_ms"] = _nearest_time(motion_times, t, snap.near_cut_tol_ms)
        a.setdefault("on_music_beat", False)
        a.setdefault("tagger_topk", [])
    return sorted(merged, key=lambda a: int(a["t_ms"]))


def format_anchors_for_prompt(
    anchors: list[dict[str, Any]],
    start_ms: int,
    end_ms: int,
    *,
    limit: int = 40,
) -> str:
    """Human-readable anchor list for Gemini slot-filling prompts (v2 denser)."""
    sel = [a for a in anchors if start_ms <= int(a.get("t_ms") or 0) <= end_ms]
    if len(sel) > limit:
        priority = {
            "speech_start": 0, "sed_event": 1, "audio_onset": 2, "music_boundary": 3,
            "intensity_change": 4, "cut": 5, "motion_stop": 6, "motion_spike": 7,
            "motion": 8, "flash": 9,
        }
        sel = sorted(
            sel,
            key=lambda a: (
                priority.get(a.get("kind"), 9),
                -float(a.get("score") or a.get("strength") or 0),
            ),
        )[:limit]
        sel = sorted(sel, key=lambda a: int(a.get("t_ms") or 0))

    lines = []
    for a in sel:
        t_s = float(a.get("t_ms_f", a.get("t_ms") or 0)) / 1000.0
        bits = [f"{a.get('id', '?')} t={t_s:.3f}s {a.get('kind')}"]
        if a.get("t_end_ms_f") is not None:
            bits.append(f"end≈{float(a['t_end_ms_f'])/1000.0:.2f}s")
        if a.get("snr_db") is not None:
            bits.append(f"snr={float(a['snr_db']):.0f}dB")
        if a.get("sed"):
            sed = ",".join(
                f"{x.get('class') or x.get('label')} {float(x.get('prob') or 0):.2f}"
                for x in a["sed"][:3]
            )
            bits.append(f"sed=[{sed}]")
        elif a.get("kind") == "sed_event" and a.get("class"):
            bits.append(f"sed=[{a.get('class')} {float(a.get('prob') or 0):.2f}]")
        vs = a.get("visual_support") or {}
        if vs.get("motion_stop_dt_ms") is not None:
            bits.append(f"vis=motion_stop({int(vs['motion_stop_dt_ms']):+d}ms)")
        elif vs.get("cut_dt_ms") is not None:
            bits.append(f"vis=cut({int(vs['cut_dt_ms']):+d}ms)")
        elif vs.get("intensity_change_dt_ms") is not None:
            bits.append(
                f"vis=intensity({vs.get('intensity_direction', '?')}"
                f",Δ={vs.get('intensity_delta', '?')})"
            )
        if a.get("kind") == "intensity_change":
            bits.append(
                f"Δ={a.get('delta')} ({a.get('direction')}) "
                f"{a.get('intensity_before')}→{a.get('intensity_after')}"
            )
        if a.get("stem"):
            bits.append(f"stem={a['stem']}")
        if a.get("bands"):
            bits.append("bands=" + "+".join(a["bands"]))
        if a.get("on_music_beat"):
            bits.append("on_beat")
        if a.get("transcript"):
            bits.append(f'text="{str(a["transcript"])[:80]}"')
        if a.get("score") is not None:
            bits.append(f"score={float(a['score']):.2f}")
        lines.append("  " + " ".join(bits))
    return "\n".join(lines) if lines else "  (none detected)"
