"""Beat grid via librosa tempo tracking.

Any mixture onset that lands on a beat AND is absent from the effects stem
is likely a kick/snare rather than an SFX hit. The anchor merger uses
``on_music_beat`` to flag those candidates.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger("visualmellow.perception.beats")


def detect_beats(audio: np.ndarray, sample_rate: int = 16000) -> dict[str, Any]:
    """Return tempo (BPM) and beat times in milliseconds.

    Falls back to an empty beat list if librosa cannot estimate a tempo
    (very short / near-silent clips).
    """
    import librosa

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=audio.astype(np.float32), sr=sample_rate)
        # librosa ≥0.10 may return tempo as a 0-d array.
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.item() if tempo.size else 0.0)
        else:
            tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        beats_ms = [int(round(t * 1000)) for t in beat_times]
    except Exception as exc:
        LOG.warning("Beat tracking failed: %s", exc)
        return {"tempo_bpm": 0.0, "beats_ms": []}
    LOG.info("Beats: tempo=%.1f BPM, %d beats", tempo, len(beats_ms))
    return {"tempo_bpm": tempo, "beats_ms": beats_ms}


def flag_on_beat(
    onsets: list[dict[str, Any]],
    beats_ms: list[int],
    *,
    tol_ms: int = 50,
) -> list[dict[str, Any]]:
    """Set ``on_music_beat=True`` on onsets within ``tol_ms`` of a beat."""
    if not beats_ms:
        for o in onsets:
            o.setdefault("on_music_beat", False)
        return onsets
    beats = np.asarray(beats_ms, dtype=np.int64)
    for o in onsets:
        t = int(o.get("t_ms", 0))
        idx = int(np.argmin(np.abs(beats - t)))
        o["on_music_beat"] = bool(abs(int(beats[idx]) - t) <= tol_ms)
    return onsets
