"""Deterministic timing snap to Layer-0 anchors.

v1: nearest onset within tolerances (legacy).
v2: class-aware Hungarian assignment within ±assign_window_ms.

Never invents times; only moves a cue onto an existing anchor.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

import numpy as np

from core.schemas import PipelineConfig

LOG = logging.getLogger("visualmellow.annotate.snap")

_BED_TYPES = {"AMBIENCE", "MUSIC", "MOVIE_BGM"}
_SFX_STEMS = {"effects", "mixture"}
_STOP = {"a", "an", "the", "of", "on", "in", "with", "and", "continuous", "close", "mid", "distant"}


def _nearest(
    anchors: Sequence[dict[str, Any]],
    ms: int,
    *,
    kinds: set[str] | None = None,
    stems: set[str] | None = None,
    tol_ms: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    best = None
    best_d = None
    for a in anchors:
        if kinds and a.get("kind") not in kinds:
            continue
        if stems and a.get("stem") not in stems and a.get("kind") != "speech_start":
            if "speech_start" not in (kinds or set()):
                continue
        d = abs(int(a["t_ms"]) - ms)
        if best is None or d < best_d:
            best, best_d = a, d
    if best is not None and tol_ms is not None and best_d > tol_ms:
        return None
    return best


def _anchor_t_ms(a: dict[str, Any]) -> float:
    if a.get("t_ms_f") is not None:
        return float(a["t_ms_f"])
    return float(a.get("t_ms") or 0)


def _class_match(cue: dict[str, Any], anchor: dict[str, Any]) -> float:
    """SED probability overlap between cue audio_class and anchor SED hints."""
    sed = anchor.get("sed") or []
    if anchor.get("kind") == "sed_event" and anchor.get("prob") is not None:
        sed = list(sed) + [{"class": anchor.get("class"), "prob": anchor.get("prob"), "group": anchor.get("group")}]
    if not sed:
        return 0.5  # graceful degrade when SED unavailable
    text = str(cue.get("audio_class") or cue.get("audio_prompt") or "").lower()
    tokens = {t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOP}
    best = 0.0
    for s in sed:
        label = str(s.get("class") or s.get("label") or "").lower()
        prob = float(s.get("prob") or 0)
        ltoks = {t for t in re.findall(r"[a-z0-9]+", label) if t not in _STOP}
        overlap = len(tokens & ltoks) / max(1, min(len(tokens), len(ltoks)) if tokens and ltoks else 1)
        if not tokens or not ltoks:
            # Group-level soft match via ontology keywords
            group = str(s.get("group") or "")
            if group == "impact" and any(k in text for k in ("hit", "gun", "glass", "bang", "shot", "impact", "punch")):
                overlap = 0.6
            elif group == "speech" and str(cue.get("audio_type") or "").upper() in {"DIALOGUE", "NARRATOR"}:
                overlap = 0.8
        best = max(best, prob * max(overlap, 0.15))
    return float(np.clip(best, 0.0, 1.0))


def _strength_norm(anchor: dict[str, Any]) -> float:
    if anchor.get("strength_norm") is not None:
        return float(anchor["strength_norm"])
    if anchor.get("score") is not None:
        return float(np.clip(anchor["score"], 0, 1))
    if anchor.get("prob") is not None:
        return float(anchor["prob"])
    s = float(anchor.get("strength") or 0)
    return float(np.clip(s / 10.0, 0, 1))


def snap_cue(
    cue: dict[str, Any],
    anchors: Sequence[dict[str, Any]],
    cfg: PipelineConfig,
    duration_ms: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Snap one cue (v1 nearest); return (updated_cue, action_or_None)."""
    a_type = str(cue.get("audio_type") or "").upper()
    start = int(cue.get("start_time_ms") or 0)
    dur = int(cue.get("duration_ms") or 100)
    end = start + dur
    new_start, new_end = start, end
    anchor = None
    large = False
    end_source = "llm"

    if a_type == "SFX":
        anchor = _nearest(anchors, start, kinds={"audio_onset", "sed_event"}, stems={"effects"}, tol_ms=cfg.snap.sfx_snap_ms)
        if anchor is None:
            anchor = _nearest(anchors, start, kinds={"audio_onset", "sed_event"}, stems={"mixture"}, tol_ms=cfg.snap.sfx_snap_ms)
        if anchor is not None:
            new_start = int(round(_anchor_t_ms(anchor)))
            new_end, end_source = _end_from_anchor(anchor, new_start, dur, cue)
    elif a_type in {"DIALOGUE", "NARRATOR"}:
        anchor = _nearest(anchors, start, kinds={"speech_start"}, tol_ms=cfg.snap.sfx_snap_ms * 2)
        if anchor is not None:
            new_start = int(round(_anchor_t_ms(anchor)))
            if anchor.get("end_ms"):
                new_end = int(anchor["end_ms"])
                end_source = "asr"
            elif anchor.get("t_end_ms_f") is not None:
                new_end = int(round(float(anchor["t_end_ms_f"])))
                end_source = "dsp"
            else:
                new_end = new_start + dur
    elif a_type in _BED_TYPES:
        anchor = _nearest(anchors, start, kinds={"cut", "music_boundary"}, tol_ms=cfg.snap.bed_snap_ms)
        if anchor is not None:
            new_start = int(round(_anchor_t_ms(anchor)))
        if a_type != "MOVIE_BGM":
            end_anchor = _nearest(anchors, end, kinds={"cut", "music_boundary"}, tol_ms=cfg.snap.bed_snap_ms)
            if end_anchor is not None and int(round(_anchor_t_ms(end_anchor))) > new_start + 100:
                new_end = int(round(_anchor_t_ms(end_anchor)))
                end_source = "dsp"
            else:
                new_end = new_start + (end - start)
        else:
            new_end = new_start + dur
    else:
        return cue, None

    new_start = max(0, min(new_start, max(duration_ms - 50, 0)))
    new_end = max(new_start + 50, min(new_end, duration_ms))
    delta = new_start - start
    if abs(delta) > cfg.snap.bed_snap_ms:
        large = True
    if new_start == start and new_end == end:
        if cue.get("anchor_id") is None and anchor is not None:
            cue = dict(cue)
            cue["anchor_id"] = anchor.get("id")
            cue["timing_source"] = anchor.get("timing_source")
            cue["uncertainty_ms"] = anchor.get("uncertainty_ms")
            cue["end_source"] = end_source
        return cue, None

    updated = dict(cue)
    updated["start_time_ms"] = new_start
    updated["end_time_ms"] = new_end
    updated["duration_ms"] = new_end - new_start
    updated["snap_delta_ms"] = delta
    updated["end_source"] = end_source
    if anchor is not None:
        updated["anchor_id"] = anchor.get("id")
        updated["timing_source"] = anchor.get("timing_source")
        updated["uncertainty_ms"] = anchor.get("uncertainty_ms")
    action = {
        "action": "retime",
        "actor": "rule:snap",
        "before": {"start_time_ms": start, "duration_ms": dur, "end_time_ms": end},
        "after": {"start_time_ms": new_start, "duration_ms": new_end - new_start, "end_time_ms": new_end},
        "evidence": {
            "anchor_id": (anchor or {}).get("id"),
            "anchor_t_ms": (anchor or {}).get("t_ms"),
            "delta_ms": delta,
            "large_correction": large,
            "end_source": end_source,
        },
        "reason": f"snapped {a_type} by {delta:+d} ms to nearest anchor",
    }
    return updated, action


def _end_from_anchor(anchor: dict[str, Any], new_start: int, dur: int, cue: dict[str, Any]) -> tuple[int, str]:
    event_type = str(cue.get("event_type") or "").upper()
    if event_type in {"CONTINUOUS", "SEGMENT", "GLOBAL"}:
        return new_start + dur, "llm"
    if anchor.get("end_reliable") and anchor.get("t_end_ms_f") is not None:
        return max(new_start + 50, int(round(float(anchor["t_end_ms_f"])))), "dsp"
    if anchor.get("t_end_ms_f") is not None and str(anchor.get("timing_source") or "").startswith("sed"):
        return max(new_start + 50, int(round(float(anchor["t_end_ms_f"])))), "sed"
    return new_start + max(dur, 50), "llm"


def snap_all_v2(
    cues: list[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
    cfg: PipelineConfig,
    duration_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Class-aware Hungarian assignment for SFX; beds/dialogue use v1 rules."""
    from scipy.optimize import linear_sum_assignment

    actions: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = [None] * len(cues)  # type: ignore
    window = float(getattr(cfg.snap, "assign_window_ms", 400))
    max_cost = float(getattr(cfg.snap, "max_cost", 0.85))
    w_t = float(getattr(cfg.snap, "w_t", 0.45))
    w_c = float(getattr(cfg.snap, "w_c", 0.30))
    w_s = float(getattr(cfg.snap, "w_s", 0.15))
    w_stem = float(getattr(cfg.snap, "w_stem", 0.10))

    sfx_idx = []
    for i, cue in enumerate(cues):
        a_type = str(cue.get("audio_type") or "").upper()
        if a_type == "SFX":
            sfx_idx.append(i)
        else:
            updated, action = snap_cue(cue, anchors, cfg, duration_ms)
            out[i] = updated
            if action is not None:
                action["cue_id"] = updated.get("cue_id")
                actions.append(action)

    # Candidates: audio_onset + sed_event
    cand_anchors = [
        a for a in anchors
        if a.get("kind") in {"audio_onset", "sed_event"}
    ]
    # Allow impact_times_ms bursts to share anchors — handled after unique assign.
    if not sfx_idx or not cand_anchors:
        for i in sfx_idx:
            updated, action = snap_cue(cues[i], anchors, cfg, duration_ms)
            out[i] = updated
            if action is not None:
                action["cue_id"] = updated.get("cue_id")
                actions.append(action)
        return out, actions  # type: ignore

    n_c, n_a = len(sfx_idx), len(cand_anchors)
    cost = np.full((n_c, n_a), 1e6, dtype=np.float64)
    for ci, cue_i in enumerate(sfx_idx):
        cue = cues[cue_i]
        start = float(cue.get("start_time_ms") or 0)
        for aj, a in enumerate(cand_anchors):
            at = _anchor_t_ms(a)
            dt = abs(at - start)
            if dt > window:
                continue
            class_m = _class_match(cue, a)
            strength = _strength_norm(a)
            stem_pen = 0.0 if a.get("stem") in _SFX_STEMS or a.get("kind") == "sed_event" else 1.0
            cost[ci, aj] = (
                w_t * (dt / window)
                + w_c * (1.0 - class_m)
                + w_s * (1.0 - strength)
                + w_stem * stem_pen
            )

    ri, ci_a = linear_sum_assignment(cost)
    assigned_anchor = {int(r): int(c) for r, c in zip(ri, ci_a) if cost[r, c] < 1e5}

    used_anchors: set[int] = set()
    for ci, cue_i in enumerate(sfx_idx):
        cue = cues[cue_i]
        start = int(cue.get("start_time_ms") or 0)
        dur = int(cue.get("duration_ms") or 100)
        aj = assigned_anchor.get(ci)
        if aj is None or cost[ci, aj] > max_cost or aj in used_anchors:
            # Reject → mark not_audible (verifier will soft-reject).
            updated = dict(cue)
            updated["audibility"] = "not_audible"
            updated["snap_rejected"] = True
            updated["snap_cost"] = float(cost[ci, aj]) if aj is not None else None
            out[cue_i] = updated
            actions.append({
                "cue_id": cue.get("cue_id"),
                "action": "snap_reject",
                "actor": "rule:snap_v2",
                "before": {"start_time_ms": start},
                "after": {"audibility": "not_audible"},
                "evidence": {"cost": updated.get("snap_cost"), "max_cost": max_cost},
                "reason": "no class-aware anchor within cost budget",
            })
            continue

        used_anchors.add(aj)
        anchor = cand_anchors[aj]
        new_start = int(round(_anchor_t_ms(anchor)))
        new_end, end_source = _end_from_anchor(anchor, new_start, dur, cue)
        new_start = max(0, min(new_start, max(duration_ms - 50, 0)))
        new_end = max(new_start + 50, min(new_end, duration_ms))
        delta = new_start - start
        updated = dict(cue)
        updated["start_time_ms"] = new_start
        updated["end_time_ms"] = new_end
        updated["duration_ms"] = new_end - new_start
        updated["snap_delta_ms"] = delta
        updated["anchor_id"] = anchor.get("id")
        updated["timing_source"] = anchor.get("timing_source")
        updated["uncertainty_ms"] = anchor.get("uncertainty_ms")
        updated["end_source"] = end_source
        updated["snap_cost"] = float(cost[ci, aj])
        updated["audibility"] = updated.get("audibility") or "audible"
        out[cue_i] = updated
        actions.append({
            "cue_id": cue.get("cue_id"),
            "action": "retime",
            "actor": "rule:snap_v2",
            "before": {"start_time_ms": start, "duration_ms": dur},
            "after": {"start_time_ms": new_start, "duration_ms": new_end - new_start, "end_time_ms": new_end},
            "evidence": {
                "anchor_id": anchor.get("id"),
                "anchor_t_ms": anchor.get("t_ms"),
                "delta_ms": delta,
                "cost": float(cost[ci, aj]),
                "end_source": end_source,
            },
            "reason": f"snap_v2 assigned to {anchor.get('id')} cost={cost[ci, aj]:.3f}",
        })

        # Burst impact_times_ms: snap each listed time without exclusive lock.
        impacts = cue.get("impact_times_ms") or []
        if impacts:
            snapped_impacts = []
            for it in impacts:
                near = _nearest(cand_anchors, int(it), kinds={"audio_onset", "sed_event"}, tol_ms=int(window))
                snapped_impacts.append(int(round(_anchor_t_ms(near))) if near else int(it))
            updated["impact_times_ms"] = snapped_impacts

    LOG.info("Snap v2: %d SFX assigned, %d actions", len(used_anchors), len(actions))
    return out, actions  # type: ignore


def snap_all(
    cues: list[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
    cfg: PipelineConfig,
    duration_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Snap every cue; dispatch v1/v2 from config."""
    version = int(getattr(cfg.snap, "version", 1) or 1)
    if version >= 2 and getattr(cfg.flags, "snap_v2", True):
        return snap_all_v2(cues, anchors, cfg, duration_ms)
    out = []
    actions = []
    for cue in cues:
        updated, action = snap_cue(cue, anchors, cfg, duration_ms)
        out.append(updated)
        if action is not None:
            action["cue_id"] = updated.get("cue_id")
            actions.append(action)
    LOG.info("Snap: %d/%d cues retimed", len(actions), len(cues))
    return out, actions
