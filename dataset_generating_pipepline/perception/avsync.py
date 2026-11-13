"""Estimate A/V offset by correlating audio onset strength with motion energy.

Warn-only by default; apply only when ``--apply-av-offset`` is set.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

LOG = logging.getLogger("visualmellow.perception.avsync")


def estimate_av_offset(
    onset_times_ms: Sequence[float],
    onset_strengths: Sequence[float],
    motion_times_ms: Sequence[float],
    motion_energy: Sequence[float],
    *,
    max_lag_ms: float = 300,
    hop_ms: float = 10,
) -> dict[str, Any]:
    """Cross-correlate onset-strength curve with motion-energy derivative."""
    if len(onset_times_ms) < 2 or len(motion_times_ms) < 3:
        return {"av_offset_ms": 0.0, "av_offset_confidence": 0.0, "available": False}

    duration = max(max(onset_times_ms), max(motion_times_ms)) + 100
    n = int(np.ceil(duration / hop_ms)) + 1
    onset_curve = np.zeros(n, dtype=np.float64)
    motion_curve = np.zeros(n, dtype=np.float64)
    for t, s in zip(onset_times_ms, onset_strengths):
        i = int(round(t / hop_ms))
        if 0 <= i < n:
            onset_curve[i] = max(onset_curve[i], float(s))
    for t, e in zip(motion_times_ms, motion_energy):
        i = int(round(t / hop_ms))
        if 0 <= i < n:
            motion_curve[i] = float(e)

    # Smooth lightly.
    k = 3
    kernel = np.ones(k) / k
    onset_curve = np.convolve(onset_curve, kernel, mode="same")
    motion_curve = np.convolve(motion_curve, kernel, mode="same")
    motion_deriv = np.diff(motion_curve, prepend=motion_curve[0])

    max_lag = int(round(max_lag_ms / hop_ms))
    lags = range(-max_lag, max_lag + 1)
    best_lag = 0
    best_score = -1e18
    scores = []
    for lag in lags:
        if lag < 0:
            a = onset_curve[-lag:]
            b = motion_deriv[: len(a)]
        elif lag > 0:
            b = motion_deriv[lag:]
            a = onset_curve[: len(b)]
        else:
            a = onset_curve
            b = motion_deriv
        m = min(len(a), len(b))
        if m < 8:
            scores.append(0.0)
            continue
        a = a[:m] - a[:m].mean()
        b = b[:m] - b[:m].mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        score = float(np.dot(a, b) / denom)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_lag = lag

    offset_ms = best_lag * hop_ms
    # Confidence: peak prominence vs mean of other lags.
    scores_arr = np.asarray(scores, dtype=np.float64)
    conf = 0.0
    if len(scores_arr):
        peak = float(scores_arr.max())
        mean_rest = float(np.mean(np.delete(scores_arr, int(np.argmax(scores_arr))))) if len(scores_arr) > 1 else 0.0
        conf = float(np.clip((peak - mean_rest) / (abs(mean_rest) + 0.05), 0, 1))

    if abs(offset_ms) > 20:
        LOG.warning("Estimated A/V offset %.1f ms (confidence=%.2f) — warn only unless apply flag set", offset_ms, conf)

    return {
        "av_offset_ms": float(offset_ms),
        "av_offset_confidence": conf,
        "available": True,
        "max_corr": float(best_score),
    }
