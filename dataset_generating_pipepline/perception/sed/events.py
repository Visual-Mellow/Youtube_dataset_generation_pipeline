"""Hysteresis event extraction + SED-gated onset refinement."""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from perception.sed.ontology import map_label

LOG = logging.getLogger("visualmellow.perception.sed.events")


def extract_events(
    probs: np.ndarray,
    classes: Sequence[str],
    *,
    frame_hop_s: float = 0.01,
    on_thr: float = 0.5,
    off_thr: float = 0.3,
    min_dur_ms: float = 40,
    merge_gap_ms: float = 80,
    class_thresholds: dict[str, float] | None = None,
    median_frames: int = 3,
) -> list[dict[str, Any]]:
    """Per-class hysteresis events from [T, C] probabilities."""
    if probs.size == 0:
        return []
    T, C = probs.shape
    events: list[dict[str, Any]] = []
    thr_map = class_thresholds or {}

    for ci in range(min(C, len(classes) if classes else C)):
        name = classes[ci] if ci < len(classes) else f"class_{ci}"
        on = float(thr_map.get(name, on_thr))
        off = min(off_thr, on - 0.05)
        series = probs[:, ci].astype(np.float64)
        if median_frames and median_frames > 1:
            try:
                from scipy.ndimage import median_filter

                series = median_filter(series, size=min(median_frames, 5))
            except Exception:
                pass

        active = False
        start_i = 0
        peak = 0.0
        for i, v in enumerate(series):
            if not active and v >= on:
                active = True
                start_i = i
                peak = float(v)
            elif active:
                peak = max(peak, float(v))
                if v <= off:
                    end_i = i
                    _emit(events, name, start_i, end_i, peak, frame_hop_s, min_dur_ms)
                    active = False
        if active:
            _emit(events, name, start_i, T - 1, peak, frame_hop_s, min_dur_ms)

    events.sort(key=lambda e: e["t_ms_f"])
    return _merge_gaps(events, merge_gap_ms)


def _emit(events, name, start_i, end_i, peak, hop_s, min_dur_ms):
    t0 = start_i * hop_s * 1000.0
    t1 = max(t0 + min_dur_ms, (end_i + 1) * hop_s * 1000.0)
    if t1 - t0 < min_dur_ms:
        return
    meta = map_label(name)
    events.append({
        "kind": "sed_event",
        "label": name,
        "class": name,
        "prob": float(peak),
        "audio_type": meta["audio_type"],
        "group": meta["group"],
        "bands": meta["bands"],
        "t_ms": int(round(t0)),
        "t_ms_f": round(t0, 1),
        "t_end_ms_f": round(t1, 1),
        "timing_source": "sed_coarse",
        "uncertainty_ms": 120.0,
        "end_reliable": False,
        "stem": "effects",
        "strength": float(peak),
        "strength_norm": float(peak),
    })


def _merge_gaps(events: list[dict[str, Any]], merge_gap_ms: float) -> list[dict[str, Any]]:
    if not events:
        return []
    out = [dict(events[0])]
    for e in events[1:]:
        prev = out[-1]
        same = prev.get("class") == e.get("class")
        gap = float(e["t_ms_f"]) - float(prev.get("t_end_ms_f") or prev["t_ms_f"])
        if same and gap <= merge_gap_ms:
            prev["t_end_ms_f"] = max(float(prev.get("t_end_ms_f") or 0), float(e["t_end_ms_f"]))
            prev["prob"] = max(float(prev.get("prob") or 0), float(e.get("prob") or 0))
            continue
        out.append(dict(e))
    return out


def gate_with_onsets(
    sed_events: Sequence[dict[str, Any]],
    onsets: Sequence[dict[str, Any]],
    audio: np.ndarray | None = None,
    sample_rate: int = 32000,
    *,
    search_ms: float = 250,
) -> list[dict[str, Any]]:
    """Attach refined DSP times to SED events when a matching onset exists."""
    from perception.onsets_v2 import refine_onset, estimate_offset

    out: list[dict[str, Any]] = []
    for ev in sed_events:
        t0 = float(ev["t_ms_f"])
        t1 = float(ev.get("t_end_ms_f") or (t0 + 100))
        bands = set(ev.get("bands") or [])
        # Prefer onsets whose bands intersect the class group bands.
        candidates = []
        for o in onsets:
            ot = float(o.get("t_ms_f") if o.get("t_ms_f") is not None else o.get("t_ms") or 0)
            if abs(ot - t0) > search_ms:
                continue
            ob = set(o.get("bands") or [])
            band_bonus = 0.2 if (not bands or not ob or bands & ob) else 0.0
            strength = float(o.get("strength_norm") or o.get("strength") or 0)
            candidates.append((abs(ot - t0) - band_bonus * 50, -strength, o))
        gated = dict(ev)
        if candidates:
            candidates.sort()
            best = candidates[0][2]
            gated["t_ms_f"] = float(best.get("t_ms_f") if best.get("t_ms_f") is not None else best["t_ms"])
            gated["t_ms"] = int(round(gated["t_ms_f"]))
            gated["t_ms_coarse"] = float(ev["t_ms_f"])
            if best.get("t_end_ms_f") is not None and best.get("end_reliable"):
                gated["t_end_ms_f"] = float(best["t_end_ms_f"])
                gated["end_reliable"] = True
            else:
                gated["t_end_ms_f"] = t1
            gated["timing_source"] = "dsp_refined"
            gated["uncertainty_ms"] = float(best.get("uncertainty_ms") or 10)
            gated["anchor_onset_ref"] = best.get("t_ms_f")
            gated["snr_db"] = best.get("snr_db")
            gated["sources"] = list(set((gated.get("sources") or []) + ["sed", "superflux"]))
        elif audio is not None and len(audio) > 0:
            # Direct envelope refine inside SED window.
            try:
                r = refine_onset(
                    audio, sample_rate, t0 / 1000.0, list(ev.get("bands") or ["mid"]),
                )
                gated["t_ms_f"] = float(r["t_ms_f"])
                gated["t_ms"] = int(round(gated["t_ms_f"]))
                gated["t_ms_coarse"] = t0
                off = estimate_offset(audio, sample_rate, r, next_onset_ms=t1)
                gated["t_end_ms_f"] = float(off["t_end_ms_f"])
                gated["end_reliable"] = bool(off["end_reliable"])
                gated["timing_source"] = "dsp_refined"
                gated["uncertainty_ms"] = 25.0
                gated["snr_db"] = r.get("snr_db")
            except Exception:
                gated["timing_source"] = "sed_coarse"
                gated["uncertainty_ms"] = 150.0
        else:
            gated["timing_source"] = "sed_coarse"
            gated["uncertainty_ms"] = max(float(gated.get("uncertainty_ms") or 120), 120.0)

        gated["sed"] = [{"class": ev.get("class"), "prob": ev.get("prob"), "group": ev.get("group")}]
        out.append(gated)
    return out
