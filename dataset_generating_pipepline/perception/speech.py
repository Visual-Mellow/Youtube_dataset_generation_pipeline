"""Word-level ASR via faster-whisper.

Produces dialogue segments with word timestamps (~50 ms precision). Each
segment becomes a ``dialogue`` / ``speech_start`` anchor and can be promoted
to a ``DIALOGUE`` cue without an extra Gemini call for timing.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger("visualmellow.perception.speech")

_WHISPER = None
_WHISPER_FAILED = False
_WHISPER_NAME: str | None = None


def _get_model(model_name: str = "small"):
    global _WHISPER, _WHISPER_FAILED, _WHISPER_NAME
    if _WHISPER is not None and _WHISPER_NAME == model_name:
        return _WHISPER
    if _WHISPER_FAILED and _WHISPER_NAME == model_name:
        return None
    try:
        from faster_whisper import WhisperModel

        device = "cuda" if _has_cuda() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _WHISPER = WhisperModel(model_name, device=device, compute_type=compute)
        _WHISPER_NAME = model_name
        _WHISPER_FAILED = False
        LOG.info("faster-whisper '%s' loaded on %s", model_name, device)
    except Exception as exc:
        _WHISPER_FAILED = True
        _WHISPER_NAME = model_name
        _WHISPER = None
        LOG.warning("faster-whisper unavailable (%s); skipping ASR", exc)
    return _WHISPER


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def transcribe_words(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    model_name: str = "small",
) -> dict[str, Any]:
    """Return word timestamps and grouped dialogue segments.

    Output::

        {
          "available": true|false,
          "language": "en",
          "words": [{"word": "hello", "start_ms": 1200, "end_ms": 1480}, ...],
          "segments": [{"start_ms": ..., "end_ms": ..., "text": "...", "words": [...]}, ...],
        }
    """
    model = _get_model(model_name)
    if model is None:
        return {"available": False, "language": None, "words": [], "segments": []}

    # faster-whisper accepts a numpy array at the model's expected rate (16 kHz).
    try:
        segments_iter, info = model.transcribe(
            audio.astype(np.float32),
            language=None,
            word_timestamps=True,
            vad_filter=True,
        )
    except Exception as exc:
        LOG.warning("Whisper transcription failed: %s", exc)
        return {"available": False, "language": None, "words": [], "segments": []}

    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for seg in segments_iter:
        seg_words = []
        for w in seg.words or []:
            entry = {
                "word": (w.word or "").strip(),
                "start_ms": int(round(float(w.start) * 1000)),
                "end_ms": int(round(float(w.end) * 1000)),
            }
            if entry["word"]:
                words.append(entry)
                seg_words.append(entry)
        if seg_words or (seg.text or "").strip():
            segments.append({
                "start_ms": int(round(float(seg.start) * 1000)),
                "end_ms": int(round(float(seg.end) * 1000)),
                "text": (seg.text or "").strip(),
                "words": seg_words,
            })

    lang = getattr(info, "language", None)
    LOG.info("ASR: language=%s, %d words, %d segments", lang, len(words), len(segments))
    return {"available": True, "language": lang, "words": words, "segments": segments}


def speech_anchors(asr: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn ASR segments into ``speech_start`` anchors for the timeline."""
    out = []
    for i, seg in enumerate(asr.get("segments") or []):
        out.append({
            "t_ms": int(seg["start_ms"]),
            "kind": "speech_start",
            "stem": "dialogue",
            "strength": 1.0,
            "end_ms": int(seg["end_ms"]),
            "transcript": seg.get("text") or "",
            "word_count": len(seg.get("words") or []),
            "segment_index": i,
        })
    return out
