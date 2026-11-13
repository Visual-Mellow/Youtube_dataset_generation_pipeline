"""Mel spectral-flux onset detection.

Algorithm (validated on clip_000 → 86 onsets at 10 ms frames):
  1. STFT → mel spectrogram (64 bands) → log magnitude.
  2. Per-frame positive first difference summed across bands = onset strength.
  3. Peak-pick: local max in ±30 ms, above local mean (±200 ms) + delta,
     and ≥ min_gap after the previous onset.

Returns a list of ``{t_ms, strength, stem}`` dicts. The optional ``stem``
label (``mixture`` / ``effects`` / ``dialogue`` / ``music``) is set by the
caller so the anchor merger knows which source produced each onset.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger("visualmellow.perception.onsets")


def onset_strength_envelope(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hop_ms: int = 10,
    n_mels: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_seconds, onset_strength) at ``hop_ms`` resolution."""
    import librosa

    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    env = librosa.onset.onset_strength(y=audio.astype(np.float32), sr=sample_rate, hop_length=hop, n_mels=n_mels)
    times = np.arange(len(env)) * hop / sample_rate
    return times, env.astype(np.float64)


def pick_onsets(
    times: np.ndarray,
    env: np.ndarray,
    *,
    delta: float = 0.6,
    min_gap_ms: int = 100,
    local_max_frames: int = 3,
    local_mean_frames: int = 20,
) -> list[dict[str, Any]]:
    """Numpy peak picker over an onset-strength envelope.

    Parameters mirror the chat-validated settings: ±3 frames (~30 ms) for
    local-max, ±20 frames (~200 ms) for local-mean, ``delta=0.6``, 100 ms gap.
    """
    if len(env) < local_max_frames * 2 + 1:
        return []
    min_gap_s = min_gap_ms / 1000.0
    peaks: list[dict[str, Any]] = []
    last_t = -1.0
    for i in range(local_max_frames, len(env) - local_max_frames):
        window = env[i - local_max_frames : i + local_max_frames + 1]
        if env[i] < window.max():
            continue
        mean_lo = max(0, i - local_mean_frames)
        mean_hi = min(len(env), i + local_mean_frames + 1)
        if env[i] <= env[mean_lo:mean_hi].mean() + delta:
            continue
        t = float(times[i])
        if t - last_t < min_gap_s:
            continue
        peaks.append({"t_ms": int(round(t * 1000)), "strength": float(env[i])})
        last_t = t
    return peaks


def detect_onsets(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    hop_ms: int = 10,
    delta: float = 0.6,
    min_gap_ms: int = 100,
    stem: str = "mixture",
) -> list[dict[str, Any]]:
    """Full onset pipeline: envelope → peaks → list of onset dicts with ``stem``."""
    times, env = onset_strength_envelope(audio, sample_rate, hop_ms=hop_ms)
    peaks = pick_onsets(times, env, delta=delta, min_gap_ms=min_gap_ms)
    for p in peaks:
        p["stem"] = stem
        p["kind"] = "audio_onset"
    LOG.info("Onsets (%s): %d peaks from %d frames", stem, len(peaks), len(env))
    return peaks
