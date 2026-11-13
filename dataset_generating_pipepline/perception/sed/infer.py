"""SED inference with overlapped windows + npz cache."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from perception.sed.backend import get_backend
from perception.sed.events import extract_events, gate_with_onsets
from perception.sed.ontology import class_thresholds

LOG = logging.getLogger("visualmellow.perception.sed.infer")


def run_sed(
    audio: np.ndarray,
    sample_rate: int,
    *,
    backend_name: str = "panns_cnn14_sed",
    out_dir: Path | None = None,
    clip_sha: str | None = None,
    separation_used: bool = False,
    onsets: list[dict[str, Any]] | None = None,
    window_s: float = 10.0,
    overlap: float = 0.5,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``{backend, probs, events, available}``; cache probs under out_dir."""
    cfg = dict(cfg or {})
    backend = get_backend(backend_name if cfg.get("enabled", True) else "none")
    if backend is None:
        return {"available": False, "backend": "off", "events": [], "probs": None, "classes": []}

    cache_path = None
    if out_dir is not None and clip_sha:
        cache_path = Path(out_dir) / f"sed_probs_{backend.name}_{clip_sha[:12]}_{int(separation_used)}.npz"

    probs = None
    classes = list(backend.classes)
    if cache_path is not None and cache_path.is_file() and not force:
        try:
            data = np.load(cache_path, allow_pickle=True)
            probs = data["probs"]
            classes = list(data["classes"].tolist()) if "classes" in data else classes
            LOG.info("Reusing SED cache %s", cache_path.name)
        except Exception:
            probs = None

    if probs is None:
        probs = _infer_overlapped(backend, audio, sample_rate, window_s=window_s, overlap=overlap)
        classes = list(backend.classes)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, probs=probs, classes=np.array(classes, dtype=object), hop_s=backend.frame_hop_s)
            LOG.info("SED probs cached → %s shape=%s", cache_path.name, probs.shape)

    thr = class_thresholds(cfg.get("class_thresholds"))
    raw_events = extract_events(
        probs,
        classes,
        frame_hop_s=float(getattr(backend, "frame_hop_s", 0.01)),
        on_thr=float(cfg.get("on_thr", 0.5)),
        off_thr=float(cfg.get("off_thr", 0.3)),
        min_dur_ms=float(cfg.get("min_dur_ms", 40)),
        merge_gap_ms=float(cfg.get("merge_gap_ms", 80)),
        class_thresholds=thr,
        median_frames=int(cfg.get("median_frames", 3)),
    )
    gated = gate_with_onsets(
        raw_events,
        onsets or [],
        audio=audio,
        sample_rate=sample_rate,
        search_ms=float(cfg.get("gate_search_ms", 250)),
    )
    return {
        "available": True,
        "backend": backend.name,
        "native_resolution_s": getattr(backend, "native_resolution_s", 0.16),
        "frame_hop_s": getattr(backend, "frame_hop_s", 0.01),
        "classes": classes,
        "probs": probs,
        "events": gated,
        "n_raw_events": len(raw_events),
        "n_events": len(gated),
    }


def _infer_overlapped(backend, audio, sr, *, window_s=10.0, overlap=0.5) -> np.ndarray:
    n = len(audio)
    win = int(window_s * sr)
    step = max(1, int(win * (1.0 - overlap)))
    if n <= win:
        return np.asarray(backend.predict(audio, sr), dtype=np.float32)

    chunks = []
    starts = []
    for start in range(0, n, step):
        end = min(n, start + win)
        chunk = audio[start:end]
        if len(chunk) < sr // 4:
            break
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        pred = np.asarray(backend.predict(chunk, sr), dtype=np.float32)
        chunks.append(pred)
        starts.append(start)
        if end >= n:
            break

    if not chunks:
        return np.asarray(backend.predict(audio, sr), dtype=np.float32)

    # Average overlapping frame probabilities on a global time grid.
    hop_s = float(getattr(backend, "frame_hop_s", 0.01) or 0.01)
    total_s = n / sr
    T = max(1, int(np.ceil(total_s / hop_s)))
    C = chunks[0].shape[1]
    acc = np.zeros((T, C), dtype=np.float64)
    wts = np.zeros((T,), dtype=np.float64)
    for start, pred in zip(starts, chunks):
        for fi in range(pred.shape[0]):
            t = start / sr + fi * hop_s
            idx = int(round(t / hop_s))
            if 0 <= idx < T:
                acc[idx] += pred[fi]
                wts[idx] += 1.0
    wts = np.maximum(wts, 1e-6)
    return (acc / wts[:, None]).astype(np.float32)
