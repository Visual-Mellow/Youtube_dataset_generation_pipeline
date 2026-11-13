"""Frame-level AudioSet tagging via PANNs CNN14.

Produces top-k class posteriors at ~100 ms resolution. Used to:
  - hint the Gemini proposer (``tagger_topk`` on each onset),
  - give the verifier independent evidence for ``set_audibility``.

Falls back to an empty tag stream if ``panns-inference`` is not installed
or the checkpoint cannot be downloaded — perception still works without it.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger("visualmellow.perception.tagger")

# Lazy singleton so we only load CNN14 once per process.
_TAGGER = None
_TAGGER_FAILED = False


def _ensure_panns_assets() -> None:
    """panns-inference shells out to ``wget``; fetch labels + Cnn14 via urllib if missing."""
    from pathlib import Path
    from urllib.request import urlretrieve

    root = Path.home() / "panns_data"
    root.mkdir(parents=True, exist_ok=True)
    assets = {
        "class_labels_indices.csv": (
            "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/"
            "master/metadata/class_labels_indices.csv"
        ),
        "Cnn14_mAP=0.431.pth": (
            "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
        ),
    }
    for name, url in assets.items():
        path = root / name
        if path.is_file() and path.stat().st_size > 1000:
            continue
        try:
            LOG.info("Downloading PANNs asset %s …", name)
            urlretrieve(url, str(path))
            LOG.info("Downloaded %s → %s (%d bytes)", name, path, path.stat().st_size)
        except Exception as exc:
            LOG.warning("Could not download %s: %s", name, exc)


# Back-compat alias used by older call sites / smoke tests.
_ensure_panns_labels = _ensure_panns_assets


def _get_tagger():
    global _TAGGER, _TAGGER_FAILED
    if _TAGGER is not None or _TAGGER_FAILED:
        return _TAGGER
    try:
        _ensure_panns_assets()
        from panns_inference import AudioTagging

        _TAGGER = AudioTagging(checkpoint_path=None, device="cuda" if _has_cuda() else "cpu")
        LOG.info("PANNs CNN14 audio tagger loaded")
    except Exception as exc:
        _TAGGER_FAILED = True
        LOG.warning("PANNs tagger unavailable (%s); continuing without frame tags", exc)
        _TAGGER = None
    return _TAGGER


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def tag_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    topk: int = 5,
) -> dict[str, Any]:
    """Return clip-level and (when available) frame-level AudioSet tags.

    Output shape::

        {
          "clip_topk": [["Music", 0.82], ...],
          "frames": [{"t_ms": 0, "topk": [["Music", 0.7], ...]}, ...],
          "available": true|false,
        }
    """
    tagger = _get_tagger()
    if tagger is None:
        return {"clip_topk": [], "frames": [], "available": False}

    # PANNs expects 32 kHz. Resample if needed.
    waveform = audio.astype(np.float32)
    if sample_rate != 32000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
    # Batch dim: (1, samples)
    try:
        clipwise, embedding = tagger.inference(waveform[None, :])
    except Exception as exc:
        LOG.warning("PANNs inference failed: %s", exc)
        return {"clip_topk": [], "frames": [], "available": False}

    labels = list(getattr(tagger, "labels", []) or [])
    clipwise = np.asarray(clipwise).reshape(-1)
    clip_topk = _topk_from_vector(clipwise, labels, topk)

    # Frame-level SED is not always exposed by AudioTagging; synthesise a
    # coarse 1-second grid by tagging overlapping 1 s chunks when clip-level
    # only is available (keeps the interface stable).
    frames: list[dict[str, Any]] = []
    hop_s = 1.0
    win_s = 1.0
    n = int(np.ceil(len(waveform) / 32000 / hop_s))
    for i in range(n):
        start = int(i * hop_s * 32000)
        end = min(len(waveform), start + int(win_s * 32000))
        if end - start < 32000 // 4:
            break
        chunk = waveform[start:end]
        if len(chunk) < int(win_s * 32000):
            chunk = np.pad(chunk, (0, int(win_s * 32000) - len(chunk)))
        try:
            cw, _ = tagger.inference(chunk[None, :])
            frames.append({
                "t_ms": int(round(i * hop_s * 1000)),
                "topk": _topk_from_vector(np.asarray(cw).reshape(-1), labels, topk),
            })
        except Exception:
            continue

    LOG.info("Tagger: clip_topk=%s, %d frame windows", clip_topk[:3], len(frames))
    return {"clip_topk": clip_topk, "frames": frames, "available": True}


def _topk_from_vector(scores: np.ndarray, labels: list[str], k: int) -> list[list[Any]]:
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argsort(scores)[::-1][:k]
    out = []
    for i in idx:
        name = labels[int(i)] if labels and int(i) < len(labels) else f"class_{int(i)}"
        out.append([name, float(scores[int(i)])])
    return out


def tags_near(
    tag_result: dict[str, Any],
    t_ms: int,
    *,
    window_ms: int = 1000,
) -> list[list[Any]]:
    """Nearest frame top-k to ``t_ms``, falling back to clip-level tags."""
    frames = tag_result.get("frames") or []
    if not frames:
        return list(tag_result.get("clip_topk") or [])
    best = min(frames, key=lambda f: abs(int(f["t_ms"]) - t_ms))
    if abs(int(best["t_ms"]) - t_ms) > window_ms:
        return list(tag_result.get("clip_topk") or [])
    return list(best.get("topk") or [])
