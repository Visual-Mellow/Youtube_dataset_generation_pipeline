#!/usr/bin/env python3
"""Per-component Layer-0 anchor diagnostic (noise isolation).

Pass a video path; each detector runs in isolation and writes its own
JSON + Audacity label track so you can see which component floods anchors.

Usage
-----
  cd dataset_generating_pipepline

  # All components (recommended first pass)
  python -m tools.diagnose_anchors --video ../Dataset/raw_videos/nq1M_Wc4FIc/clip_000.mp4

  # One component only
  python -m tools.diagnose_anchors --video /path/to/clip.mp4 --only onsets_v2
  python -m tools.diagnose_anchors --video /path/to/clip.mp4 --only intensity_change
  python -m tools.diagnose_anchors --video /path/to/clip.mp4 --only video_cues

  # Skip heavy deps
  python -m tools.diagnose_anchors --video clip.mp4 --skip sed,speech,separation

Source files (what decides each anchor kind)
--------------------------------------------
  onsets_v1         perception/onsets.py
  onsets_v2         perception/onsets_v2.py          ← usually the noisiest
  onsets_v2_effects perception/onsets_v2.py + separation.py (effects stem only)
  intensity_change  perception/video_cues.py::detect_intensity_changes
  video_cues        perception/video_cues.py         (cut / motion_* / flash / intensity)
  scenes            perception/scenes.py             (ffmpeg scene_score cut/motion)
  speech            perception/speech.py             (Whisper speech_start)
  sed               perception/sed/*                 (frame SED / windowed tagger)
  beats             perception/beats.py              (tempo grid only; not cue anchors)
  fusion            perception/anchors.py            (merge of the above → anchors.json)

Outputs land under ``--out-dir`` (default: ``<video_stem>_anchor_diag/``):
  summary.json
  summary.txt
  <component>.json
  <component>.txt          # Audacity: start\\tend\\tlabel
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

LOG = logging.getLogger("visualmellow.diagnose_anchors")

COMPONENT_SOURCES: dict[str, str] = {
    "onsets_v1": "perception/onsets.py",
    "onsets_v2": "perception/onsets_v2.py",
    "onsets_v2_effects": "perception/onsets_v2.py + perception/separation.py",
    "intensity_change": "perception/video_cues.py::detect_intensity_changes",
    "video_cues": "perception/video_cues.py",
    "scenes": "perception/scenes.py",
    "speech": "perception/speech.py",
    "sed": "perception/sed/{backend,infer,events,ontology}.py",
    "beats": "perception/beats.py",
    "fusion": "perception/anchors.py",
}

ALL_COMPONENTS = list(COMPONENT_SOURCES.keys())


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_audacity(path: Path, events: list[dict[str, Any]], *, label_key: str = "label") -> None:
    """Audacity label track: start_s \\t end_s \\t label."""
    lines: list[str] = []
    for e in sorted(events, key=lambda x: float(x.get("t_ms_f", x.get("t_ms", 0)) or 0)):
        t0 = float(e.get("t_ms_f", e.get("t_ms", 0)) or 0) / 1000.0
        t1_raw = e.get("t_end_ms_f", e.get("end_ms"))
        t1 = float(t1_raw) / 1000.0 if t1_raw is not None else t0 + 0.05
        if t1 <= t0:
            t1 = t0 + 0.05
        label = str(e.get(label_key) or e.get("kind") or "event")
        label = label.replace("\t", " ")[:100]
        lines.append(f"{t0:.6f}\t{t1:.6f}\t{label}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _event(
    *,
    t_ms: float,
    kind: str,
    end_ms: float | None = None,
    strength: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "t_ms": int(round(t_ms)),
        "t_ms_f": round(float(t_ms), 1),
        "kind": kind,
        "label": kind,
    }
    if end_ms is not None:
        out["t_end_ms_f"] = round(float(end_ms), 1)
    if strength is not None:
        out["strength"] = float(strength)
    out.update(extra)
    if extra:
        # Richer Audacity label
        bits = [kind]
        if extra.get("stem"):
            bits.append(str(extra["stem"]))
        if extra.get("direction"):
            bits.append(str(extra["direction"]))
        if extra.get("delta") is not None:
            bits.append(f"d{extra['delta']}")
        if extra.get("bands"):
            bits.append("+".join(extra["bands"]))
        out["label"] = "|".join(bits)
    return out


def density_stats(events: list[dict[str, Any]], duration_ms: int) -> dict[str, Any]:
    if not events:
        return {
            "n": 0,
            "per_sec": 0.0,
            "median_gap_ms": None,
            "p10_gap_ms": None,
            "noise_score": 0.0,
        }
    times = sorted(float(e.get("t_ms_f", e.get("t_ms", 0)) or 0) for e in events)
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    dur_s = max(duration_ms / 1000.0, 1e-3)
    per_sec = len(events) / dur_s
    med_gap = float(np.median(gaps)) if gaps else None
    p10 = float(np.percentile(gaps, 10)) if gaps else None
    # Heuristic: >8 events/sec or median gap <80ms ⇒ noisy for LLM slot-fill.
    noise = 0.0
    if per_sec > 8:
        noise += min(1.0, (per_sec - 8) / 20.0)
    if med_gap is not None and med_gap < 80:
        noise += min(1.0, (80 - med_gap) / 80.0)
    return {
        "n": len(events),
        "per_sec": round(per_sec, 2),
        "median_gap_ms": None if med_gap is None else round(med_gap, 1),
        "p10_gap_ms": None if p10 is None else round(p10, 1),
        "noise_score": round(min(1.0, noise), 3),
    }


# ---------------------------------------------------------------------------
# Component runners
# ---------------------------------------------------------------------------

def run_onsets_v1(audio16: np.ndarray, sr16: int, **_) -> list[dict[str, Any]]:
    from perception.onsets import detect_onsets

    peaks = detect_onsets(audio16, sr16, stem="mixture")
    return [
        _event(t_ms=p["t_ms"], kind="audio_onset", strength=p.get("strength"), stem=p.get("stem", "mixture"), source="onsets_v1")
        for p in peaks
    ]


def run_onsets_v2(audio32: np.ndarray, sr32: int, cfg_dict: dict | None = None, **_) -> list[dict[str, Any]]:
    from perception.onsets_v2 import detect_onsets_v2

    peaks = detect_onsets_v2(audio32, sr32, stem="mixture", cfg=cfg_dict or {})
    out = []
    for p in peaks:
        out.append(_event(
            t_ms=float(p.get("t_ms_f", p["t_ms"])),
            kind="audio_onset",
            end_ms=p.get("t_end_ms_f"),
            strength=p.get("strength_norm") or p.get("strength"),
            stem=p.get("stem", "mixture"),
            bands=p.get("bands"),
            snr_db=p.get("snr_db"),
            source="onsets_v2",
        ))
    return out


def run_onsets_v2_effects(
    audio16: np.ndarray,
    sr16: int,
    audio32: np.ndarray,
    sr32: int,
    cfg_dict: dict | None = None,
    **_,
) -> list[dict[str, Any]]:
    from perception.separation import separate_stems
    from perception.onsets_v2 import detect_onsets_v2
    import librosa

    stems = separate_stems(audio16, sample_rate=sr16)
    if stems is None:
        LOG.warning("Demucs unavailable — onsets_v2_effects skipped")
        return []
    effects32 = librosa.resample(stems["effects"].astype(np.float32), orig_sr=sr16, target_sr=sr32)
    peaks = detect_onsets_v2(effects32, sr32, stem="effects", cfg=cfg_dict or {})
    return [
        _event(
            t_ms=float(p.get("t_ms_f", p["t_ms"])),
            kind="audio_onset",
            end_ms=p.get("t_end_ms_f"),
            strength=p.get("strength_norm") or p.get("strength"),
            stem="effects",
            bands=p.get("bands"),
            snr_db=p.get("snr_db"),
            source="onsets_v2_effects",
        )
        for p in peaks
    ]


def run_intensity_change(video: Path, min_change: float = 5.0, **_) -> list[dict[str, Any]]:
    import cv2
    from perception.timebase import extract_frame_pts
    from perception.video_cues import detect_intensity_changes

    cap = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, max(1, int(frame.shape[0] * 160 / max(frame.shape[1], 1)))))
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 2:
        return []
    pts = extract_frame_pts(video)
    if len(pts) < len(frames):
        # pad/truncate
        fps = len(frames) / max(float(pts[-1]) if len(pts) else 1.0, 1e-3)
        pts = np.arange(len(frames)) / max(fps, 1e-3)
    else:
        pts = pts[: len(frames)]
    raw = detect_intensity_changes(frames, pts, min_change=min_change)
    return [
        _event(
            t_ms=float(a["t_ms_f"]),
            kind="intensity_change",
            end_ms=a.get("t_end_ms_f"),
            strength=a.get("abs_delta") or a.get("strength"),
            direction=a.get("direction"),
            delta=a.get("delta"),
            intensity_before=a.get("intensity_before"),
            intensity_after=a.get("intensity_after"),
            source="intensity_change",
        )
        for a in raw
    ]


def run_video_cues(video: Path, min_change: float = 5.0, use_farneback: bool = True, **_) -> list[dict[str, Any]]:
    from perception.timebase import extract_frame_pts
    from perception.video_cues import extract_video_cues_v2

    pts = extract_frame_pts(video)
    doc = extract_video_cues_v2(
        video,
        frame_pts_s=pts,
        use_farneback=use_farneback,
        min_intensity_change=min_change,
    )
    out = []
    for a in doc.get("anchors") or []:
        # Drop motion alias duplicates for cleaner comparison
        if a.get("kind") == "motion":
            continue
        out.append(_event(
            t_ms=float(a.get("t_ms_f", a.get("t_ms", 0))),
            kind=str(a.get("kind")),
            end_ms=a.get("t_end_ms_f"),
            strength=a.get("strength"),
            direction=a.get("direction"),
            delta=a.get("delta"),
            source="video_cues",
        ))
    return out


def run_scenes(video: Path, **_) -> list[dict[str, Any]]:
    from perception.scenes import extract_visual_anchors, visual_anchors_as_dicts

    raw = visual_anchors_as_dicts(extract_visual_anchors(video))
    return [
        _event(t_ms=float(a["t_ms"]), kind=str(a["kind"]), strength=a.get("strength"), source="scenes")
        for a in raw
    ]


def run_speech(audio16: np.ndarray, sr16: int, **_) -> list[dict[str, Any]]:
    from perception.speech import transcribe_words, speech_anchors

    asr = transcribe_words(audio16, sr16)
    anchors = speech_anchors(asr)
    return [
        _event(
            t_ms=float(a.get("t_ms", 0)),
            kind="speech_start",
            end_ms=a.get("end_ms"),
            transcript=(a.get("transcript") or "")[:80],
            source="speech",
        )
        for a in anchors
    ]


def run_sed(audio16: np.ndarray, sr16: int, audio32: np.ndarray, sr32: int, **_) -> list[dict[str, Any]]:
    from perception.sed.infer import run_sed as _run_sed

    doc = _run_sed(audio32, sr32, backend_name="panns_cnn14_sed", onsets=[], force=True)
    return [
        _event(
            t_ms=float(e.get("t_ms_f", e.get("t_ms", 0))),
            kind="sed_event",
            end_ms=e.get("t_end_ms_f"),
            strength=e.get("prob"),
            class_name=e.get("class"),
            group=e.get("group"),
            source="sed",
        )
        for e in (doc.get("events") or [])
    ]


def run_beats(audio16: np.ndarray, sr16: int, **_) -> list[dict[str, Any]]:
    from perception.beats import detect_beats

    info = detect_beats(audio16, sr16)
    return [
        _event(t_ms=float(t), kind="beat", strength=1.0, tempo_bpm=info.get("tempo_bpm"), source="beats")
        for t in (info.get("beats_ms") or [])
    ]


def run_fusion(video: Path, level: str = "low", **_) -> list[dict[str, Any]]:
    from core.config import load_pipeline_config
    from perception.anchors import build_anchors

    cfg = load_pipeline_config(level)
    doc = build_anchors(video, cfg, out_dir=None, force=True)
    return [
        _event(
            t_ms=float(a.get("t_ms_f", a.get("t_ms", 0))),
            kind=str(a.get("kind")),
            end_ms=a.get("t_end_ms_f") or a.get("end_ms"),
            strength=a.get("score") or a.get("strength"),
            stem=a.get("stem"),
            source="fusion",
            id=a.get("id"),
        )
        for a in (doc.get("anchors") or [])
    ]


RUNNERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "onsets_v1": run_onsets_v1,
    "onsets_v2": run_onsets_v2,
    "onsets_v2_effects": run_onsets_v2_effects,
    "intensity_change": run_intensity_change,
    "video_cues": run_video_cues,
    "scenes": run_scenes,
    "speech": run_speech,
    "sed": run_sed,
    "beats": run_beats,
    "fusion": run_fusion,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def diagnose(
    video: Path,
    *,
    out_dir: Path,
    components: list[str],
    level: str = "low",
    min_intensity_change: float = 5.0,
    use_farneback: bool = True,
    mad_k: float | None = None,
) -> dict[str, Any]:
    from perception.audio_io import decode_mono, decode_dsp
    from perception.timebase import probe
    from core.config import load_pipeline_config

    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = probe(video)
    duration_ms = int(meta.get("duration_ms") or round(float(meta.get("duration") or 0) * 1000))
    LOG.info("Video %s duration=%dms fps=%s", video.name, duration_ms, meta.get("fps"))

    need_audio = any(c in components for c in (
        "onsets_v1", "onsets_v2", "onsets_v2_effects", "speech", "sed", "beats", "fusion",
    ))
    audio16 = audio32 = None
    sr16, sr32 = 16000, 32000
    if need_audio:
        audio16 = decode_mono(video, sample_rate=sr16)
        audio32 = decode_dsp(video, sample_rate=sr32)
        duration_ms = max(duration_ms, int(round(len(audio16) / sr16 * 1000)))

    cfg = load_pipeline_config(level)
    ov2_cfg = cfg.perception.onsets_v2.as_dict()
    if mad_k is not None:
        ov2_cfg["mad_k"] = float(mad_k)

    results: dict[str, Any] = {
        "video": str(video),
        "duration_ms": duration_ms,
        "created_at": _utc(),
        "component_sources": {c: COMPONENT_SOURCES[c] for c in components},
        "components": {},
    }

    ctx = dict(
        video=video,
        audio16=audio16,
        sr16=sr16,
        audio32=audio32,
        sr32=sr32,
        cfg_dict=ov2_cfg,
        min_change=min_intensity_change,
        use_farneback=use_farneback,
        level=level,
    )

    print("\n" + "=" * 72)
    print(f"ANCHOR DIAGNOSTIC  {video.name}")
    print(f"duration={duration_ms}ms  out={out_dir}")
    print("=" * 72)
    print(f"{'component':22s} {'n':>6} {'/sec':>7} {'med_gap':>8} {'noise':>6}  source file")
    print("-" * 72)

    ranking: list[tuple[float, str, int]] = []

    for name in components:
        runner = RUNNERS[name]
        t0 = time.perf_counter()
        err = None
        events: list[dict[str, Any]] = []
        try:
            events = runner(**ctx)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            LOG.error("%s failed: %s\n%s", name, exc, traceback.format_exc())
        elapsed = time.perf_counter() - t0
        stats = density_stats(events, duration_ms)
        by_kind: dict[str, int] = {}
        for e in events:
            by_kind[str(e.get("kind"))] = by_kind.get(str(e.get("kind")), 0) + 1

        payload = {
            "component": name,
            "source_file": COMPONENT_SOURCES[name],
            "elapsed_s": round(elapsed, 3),
            "error": err,
            "stats": stats,
            "by_kind": by_kind,
            "events": events,
        }
        results["components"][name] = payload
        _atomic_json(out_dir / f"{name}.json", payload)
        write_audacity(out_dir / f"{name}.txt", events)

        noise = stats["noise_score"]
        med = stats["median_gap_ms"]
        med_s = f"{med:.0f}" if med is not None else "-"
        status = "FAIL" if err else f"n={stats['n']}"
        print(
            f"{name:22s} {stats['n']:6d} {stats['per_sec']:7.2f} {med_s:>8} {noise:6.3f}  "
            f"{COMPONENT_SOURCES[name]}"
            + (f"  ERR={err}" if err else "")
        )
        if not err:
            ranking.append((noise, name, stats["n"]))

    ranking.sort(reverse=True)
    results["noise_ranking"] = [
        {"rank": i + 1, "component": n, "noise_score": sc, "n": nn}
        for i, (sc, n, nn) in enumerate(ranking)
    ]

    # Human summary
    lines = [
        f"Anchor diagnostic for {video}",
        f"duration_ms={duration_ms}",
        f"created_at={results['created_at']}",
        "",
        "NOISE RANKING (highest first) — start fixing from the top:",
        "",
    ]
    for row in results["noise_ranking"]:
        lines.append(
            f"  #{row['rank']}  {row['component']:22s}  noise={row['noise_score']:.3f}  n={row['n']}"
            f"  ← {COMPONENT_SOURCES[row['component']]}"
        )
    lines += [
        "",
        "How to inspect:",
        "  1. Open each <component>.txt as an Audacity label track on the video audio",
        "  2. Compare density visually — the densest track is usually the noise source",
        "  3. Tune that component's knobs in pipeline_config.json / its source file",
        "",
        "Quick knobs:",
        "  onsets_v2     perception.onsets_v2.mad_k (try 5–7), min_gap_ms (try 80–120)",
        "  intensity     perception.video_cues_v2.min_intensity_change (try 8–12)",
        "  fusion        perception/anchors.py visual_attach / audio_cluster",
        "",
    ]
    summary_txt = out_dir / "summary.txt"
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(out_dir / "summary.json", results)

    print("-" * 72)
    print("\n".join(lines[:12]))
    print(f"\nWrote summary → {summary_txt}")
    print(f"Per-component JSON + Audacity labels → {out_dir}/")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", type=Path, default=None, help="Path to an MP4 clip")
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output folder (default: <video_stem>_anchor_diag next to the video)",
    )
    ap.add_argument(
        "--only",
        default=None,
        help=f"Comma-separated subset of: {','.join(ALL_COMPONENTS)}",
    )
    ap.add_argument(
        "--skip",
        default="",
        help="Comma-separated components to skip (e.g. sed,speech,fusion)",
    )
    ap.add_argument("--level", choices=("low", "high"), default="low", help="Used by fusion runner")
    ap.add_argument("--min-intensity-change", type=float, default=5.0)
    ap.add_argument("--no-farneback", action="store_true", help="Faster video_cues (MAD only)")
    ap.add_argument("--mad-k", type=float, default=None, help="Override onsets_v2 mad_k for A/B")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--list", action="store_true", help="Print components + source files and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.list:
        print("Component → source file")
        for k, v in COMPONENT_SOURCES.items():
            print(f"  {k:22s}  {v}")
        return 0

    if args.video is None:
        ap.error("--video is required (unless using --list)")

    if args.only:
        components = [c.strip() for c in args.only.split(",") if c.strip()]
    else:
        components = list(ALL_COMPONENTS)
    skip = {c.strip() for c in args.skip.split(",") if c.strip()}
    components = [c for c in components if c not in skip]
    bad = [c for c in components if c not in RUNNERS]
    if bad:
        ap.error(f"Unknown components: {bad}. Choose from {ALL_COMPONENTS}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.video.resolve().with_name(args.video.stem + "_anchor_diag")

    diagnose(
        args.video,
        out_dir=out_dir,
        components=components,
        level=args.level,
        min_intensity_change=args.min_intensity_change,
        use_farneback=not args.no_farneback,
        mad_k=args.mad_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
