"""Layer 1 — Gemini cue proposer (anchored slot-filling).

LOW: one A/V call over the whole clip with the full anchor list injected.
HIGH: per ~6 s window (snapped to cuts), A/V call + optional audio-only call;
proposals are merged afterward.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.schemas import PipelineConfig
from perception.anchors import format_anchors_for_prompt
from annotate.prompts import PROPOSER_PROMPT, AUDIO_ONLY_PROPOSER_PROMPT

LOG = logging.getLogger("visualmellow.annotate.proposer")

_AUDIO_TYPES = {"SFX", "AMBIENCE", "MUSIC", "MOVIE_BGM", "NARRATOR", "DIALOGUE"}
_EVENT_TYPES = {"INSTANT", "EVENT", "CONTINUOUS", "SEGMENT", "GLOBAL"}
_ALIGN = {"SYNCHRONOUS", "ANTICIPATORY", "REACTIVE", "CONTEXTUAL", "TRANSITIONAL"}
_STOP = {"a", "an", "the", "of", "on", "in", "with", "and", "continuous", "close", "mid", "distant", "not", "none"}


def build_windows(
    duration_ms: int,
    cfg: PipelineConfig,
    anchors: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return ``[(start_ms, end_ms), ...]`` sub-windows for HIGH proposers.

    Window edges snap to the nearest cut within ``cut_snap_tol_ms`` when that
    keeps the window near the configured size.
    """
    w = cfg.windows.window_ms
    overlap = cfg.windows.window_overlap_ms
    step = max(1000, w - overlap)
    tol = cfg.windows.cut_snap_tol_ms
    cuts = sorted(int(a["t_ms"]) for a in anchors if a.get("kind") == "cut")

    windows: list[tuple[int, int]] = []
    start = 0
    while start < duration_ms:
        end = min(duration_ms, start + w)
        # Snap end to a nearby cut if it doesn't shrink the window too much.
        if cuts and end < duration_ms:
            near = min(cuts, key=lambda c: abs(c - end))
            if abs(near - end) <= tol and near > start + w // 2:
                end = near
        if end <= start:
            break
        windows.append((start, end))
        if end >= duration_ms:
            break
        next_start = start + step
        # Snap next start to a cut near the planned step.
        if cuts:
            near = min(cuts, key=lambda c: abs(c - next_start))
            if abs(near - next_start) <= tol and near > start:
                next_start = near
        start = next_start
    if not windows:
        windows = [(0, duration_ms)]
    LOG.info("Proposer windows: %s", windows)
    return windows


def propose_cues(
    *,
    clip_path,
    clip_bytes: bytes,
    duration_ms: int,
    explanation: dict[str, Any],
    anchors_doc: dict[str, Any],
    cfg: PipelineConfig,
    llm_call,
) -> list[dict[str, Any]]:
    """Run LOW or HIGH proposer and return raw cue dicts (pre-verifier).

    ``llm_call(stage, model, prompt, *, video_bytes=None, audio_bytes=None,
    mime_type=..., video_fps=...) -> Any`` is injected by the orchestrator so
    every call is logged centrally.
    """
    anchors = list(anchors_doc.get("anchors") or [])
    story = json.dumps(explanation, ensure_ascii=False)

    if cfg.flags.per_window_proposer:
        windows = build_windows(duration_ms, cfg, anchors)
        all_cues: list[dict[str, Any]] = []
        for w_start, w_end in windows:
            av = _propose_window(
                clip_path=clip_path,
                duration_ms=duration_ms,
                window_start_ms=w_start,
                window_end_ms=w_end,
                story=story,
                anchors=anchors,
                cfg=cfg,
                llm_call=llm_call,
                audio_only=False,
                full_clip_bytes=clip_bytes if (w_start == 0 and w_end >= duration_ms) else None,
            )
            all_cues.extend(av)
            if cfg.flags.audio_only_pass:
                ao = _propose_window(
                    clip_path=clip_path,
                    duration_ms=duration_ms,
                    window_start_ms=w_start,
                    window_end_ms=w_end,
                    story=story,
                    anchors=anchors,
                    cfg=cfg,
                    llm_call=llm_call,
                    audio_only=True,
                    full_clip_bytes=None,
                )
                all_cues = merge_proposals(all_cues, ao, prefer_audio_only_labels=True)
        # Also seed dialogue cues directly from ASR so we never miss speech.
        all_cues = _seed_dialogue_from_asr(all_cues, anchors_doc)
        return all_cues

    # LOW: single whole-clip call.
    cues = _propose_window(
        clip_path=clip_path,
        duration_ms=duration_ms,
        window_start_ms=0,
        window_end_ms=duration_ms,
        story=story,
        anchors=anchors,
        cfg=cfg,
        llm_call=llm_call,
        audio_only=False,
        full_clip_bytes=clip_bytes,
    )
    return _seed_dialogue_from_asr(cues, anchors_doc)


def _propose_window(
    *,
    clip_path,
    duration_ms: int,
    window_start_ms: int,
    window_end_ms: int,
    story: str,
    anchors: list[dict[str, Any]],
    cfg: PipelineConfig,
    llm_call,
    audio_only: bool,
    full_clip_bytes: bytes | None,
) -> list[dict[str, Any]]:
    from perception.audio_io import extract_audio_bytes, extract_video_bytes

    clip_duration_ms = window_end_ms - window_start_ms
    clip_origin_ms = window_start_ms
    anchor_text = format_anchors_for_prompt(
        anchors, window_start_ms, window_end_ms, limit=cfg.windows.max_anchors_per_window,
    )
    template = AUDIO_ONLY_PROPOSER_PROMPT if audio_only else PROPOSER_PROMPT
    prompt = template.format(
        video_duration_ms=duration_ms,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        clip_origin_ms=clip_origin_ms,
        clip_duration_ms=clip_duration_ms,
        visual_story=story,
        anchors=anchor_text,
    )

    stage = "proposer_audio_only" if audio_only else "proposer_av"
    if audio_only:
        audio_bytes = extract_audio_bytes(
            clip_path, start_ms=window_start_ms, duration_ms=clip_duration_ms,
            sample_rate=cfg.perception.sample_rate,
        )
        result = llm_call(
            stage, cfg.models.proposer, prompt,
            audio_bytes=audio_bytes, mime_type="audio/wav",
        )
    else:
        if full_clip_bytes is not None:
            video_bytes = full_clip_bytes
        else:
            video_bytes = extract_video_bytes(
                clip_path, start_ms=window_start_ms, duration_ms=clip_duration_ms,
            )
        result = llm_call(
            stage, cfg.models.proposer, prompt,
            video_bytes=video_bytes, mime_type="video/mp4",
            video_fps=cfg.perception.gemini_video_fps,
        )

    raw = _parse_cues(result)
    # Clock-shift sanity: model timed against the attached sub-clip instead of ORIGINAL.
    if raw and clip_origin_ms > 0:
        starts = [int(c.get("start_time_ms") or 0) for c in raw]
        if all(0 <= s < clip_duration_ms for s in starts) and not any(
            window_start_ms <= s <= window_end_ms for s in starts
        ):
            LOG.info("%s: shifting clip-relative times by +%dms", stage, clip_origin_ms)
            for c in raw:
                c["start_time_ms"] = int(c.get("start_time_ms") or 0) + clip_origin_ms
                if c.get("end_time_ms") is not None:
                    c["end_time_ms"] = int(c["end_time_ms"]) + clip_origin_ms
                if c.get("impact_times_ms"):
                    c["impact_times_ms"] = [int(t) + clip_origin_ms for t in c["impact_times_ms"]]

    out = []
    for c in raw:
        start = int(c.get("start_time_ms") or 0)
        if not (window_start_ms - 200 <= start <= window_end_ms + 200):
            continue
        c["source"] = "audio_only_agent" if audio_only else "foley_agent"
        out.append(c)
    LOG.info("%s [%d,%d]: %d cues", stage, window_start_ms, window_end_ms, len(out))
    return out


def _parse_cues(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, dict):
        raw = result.get("cues") or result.get("audio_cues") or []
    elif isinstance(result, list):
        raw = result
    else:
        return []
    out = []
    for i, c in enumerate(raw, start=1):
        if not isinstance(c, dict):
            continue
        audio_class = str(c.get("audio_class") or "").strip()
        if not audio_class or audio_class.lower() == "none":
            continue
        typ = str(c.get("audio_type") or "SFX").upper()
        if typ not in _AUDIO_TYPES:
            typ = "SFX"
        try:
            start = int(round(float(c.get("start_time_ms") or 0)))
        except (TypeError, ValueError):
            continue
        try:
            dur = max(50, int(round(float(c.get("duration_ms") or 500))))
        except (TypeError, ValueError):
            dur = 500
        end_raw = c.get("end_time_ms")
        try:
            end = int(round(float(end_raw))) if end_raw is not None else start + dur
        except (TypeError, ValueError):
            end = start + dur
        if end <= start:
            end = start + dur
        event_type = str(c.get("event_type") or "EVENT").upper()
        if event_type not in _EVENT_TYPES:
            event_type = "EVENT"
        align = str(c.get("alignment_relation") or "SYNCHRONOUS").upper()
        if align not in _ALIGN:
            align = "SYNCHRONOUS"
        impacts = c.get("impact_times_ms") or []
        if not isinstance(impacts, list):
            impacts = []
        try:
            impacts = [int(round(float(t))) for t in impacts]
        except (TypeError, ValueError):
            impacts = []
        out.append({
            "cue_id": str(c.get("cue_id") or f"cue_{i:03d}"),
            "anchor_id": c.get("anchor_id"),
            "audio_type": typ,
            "audio_class": audio_class,
            "audio_prompt": audio_class,
            "cue_trigger": str(c.get("cue_trigger") or audio_class).strip(),
            "alignment_relation": align,
            "event_type": event_type,
            "start_time_ms": start,
            "duration_ms": end - start,
            "end_time_ms": end,
            "visual_evidence": str(c.get("visual_evidence") or "").strip(),
            "source_audio_context": str(c.get("source_audio_context") or "").strip(),
            "impact_times_ms": impacts,
            "transcript": c.get("transcript"),
        })
    return out


def _stem(t: str) -> str:
    for suf in ("ing", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _tokens(text: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP}


def _text_sim(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def merge_proposals(
    primary: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    *,
    prefer_audio_only_labels: bool = False,
    time_tol_ms: int = 400,
) -> list[dict[str, Any]]:
    """Merge two cue lists; drop near-duplicates (same type/time/meaning).

    When ``prefer_audio_only_labels`` is set and an audio-only cue matches an
    A/V cue, keep the A/V timing but prefer the audio-only ``audio_class``
    (reduces visual hallucination in descriptions).
    """
    out = [dict(p) for p in primary]
    added = 0
    for e in extra:
        e_start = int(e.get("start_time_ms") or 0)
        e_type = str(e.get("audio_type") or "").upper()
        dup = False
        for p in out:
            if str(p.get("audio_type") or "").upper() != e_type:
                continue
            try:
                p_start = int(p.get("start_time_ms") or 0)
            except (TypeError, ValueError):
                continue
            if abs(p_start - e_start) <= time_tol_ms and _text_sim(
                str(p.get("audio_class") or ""), str(e.get("audio_class") or "")
            ) >= 0.5:
                dup = True
                if prefer_audio_only_labels and e.get("source") == "audio_only_agent":
                    # Prefer audio-only naming; mark disagreement for the judge.
                    if _text_sim(str(p.get("audio_class") or ""), str(e.get("audio_class") or "")) < 0.85:
                        p["audio_only_class"] = e.get("audio_class")
                        p["av_audio_only_disagree"] = True
                    else:
                        p["audio_class"] = e.get("audio_class")
                        p["audio_prompt"] = e.get("audio_class")
                break
        if not dup:
            out.append(dict(e))
            added += 1
    LOG.info("Merge proposals: %d primary + %d new (%d extras considered)", len(primary), added, len(extra))
    return out


def _seed_dialogue_from_asr(
    cues: list[dict[str, Any]],
    anchors_doc: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ensure every ASR segment has a DIALOGUE cue (timing from Whisper)."""
    segments = (anchors_doc.get("asr") or {}).get("segments") or []
    if not segments:
        return cues
    existing = [
        (int(c["start_time_ms"]), int(c["end_time_ms"]))
        for c in cues
        if str(c.get("audio_type") or "").upper() in {"DIALOGUE", "NARRATOR"}
    ]
    out = list(cues)
    for i, seg in enumerate(segments):
        start = int(seg["start_ms"])
        end = int(seg["end_ms"])
        text = (seg.get("text") or "").strip()
        if not text or end <= start:
            continue
        # Skip if an existing dialogue cue already covers this span.
        if any(abs(s - start) < 400 or (s <= start < e) for s, e in existing):
            continue
        out.append({
            "cue_id": f"dlg_{i:03d}",
            "anchor_id": None,
            "audio_type": "DIALOGUE",
            "audio_class": f"spoken dialogue: {text[:120]}",
            "audio_prompt": f"spoken dialogue: {text[:120]}",
            "cue_trigger": "ASR speech segment",
            "alignment_relation": "SYNCHRONOUS",
            "event_type": "EVENT",
            "start_time_ms": start,
            "duration_ms": end - start,
            "end_time_ms": end,
            "visual_evidence": "",
            "source_audio_context": "whisper word timestamps",
            "impact_times_ms": [],
            "transcript": text,
            "source": "asr",
        })
    return out
