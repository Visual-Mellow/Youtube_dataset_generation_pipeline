"""Always-available SED fallback: sliding-window clip tagger."""
from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger("visualmellow.perception.sed.windowed")


class WindowedTaggerSED:
    name = "windowed_tagger"
    frame_hop_s = 0.1
    native_resolution_s = 1.0

    def __init__(self, win_s: float = 0.75, hop_s: float = 0.1) -> None:
        from perception.tagger import _get_tagger

        self.win_s = win_s
        self.frame_hop_s = hop_s
        self._tagger = _get_tagger()
        self.classes: list[str] = list(getattr(self._tagger, "labels", []) or []) if self._tagger else []
        if self._tagger is None:
            LOG.warning("WindowedTaggerSED: PANNs unavailable; predictions will be zeros")

    def predict(self, wav: np.ndarray, sr: int) -> np.ndarray:
        C = len(self.classes) or 527
        if self._tagger is None or wav.size == 0:
            return np.zeros((1, C), dtype=np.float32)

        waveform = wav.astype(np.float32)
        if sr != 32000:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=32000)

        hop_s = self.frame_hop_s
        win_s = self.win_s
        n = int(np.ceil(len(waveform) / 32000 / hop_s))
        out = np.zeros((max(n, 1), C), dtype=np.float32)
        for i in range(n):
            start = int(i * hop_s * 32000)
            end = min(len(waveform), start + int(win_s * 32000))
            if end - start < 8000:
                break
            chunk = waveform[start:end]
            need = int(win_s * 32000)
            if len(chunk) < need:
                chunk = np.pad(chunk, (0, need - len(chunk)))
            try:
                cw, _ = self._tagger.inference(chunk[None, :])
                vec = np.asarray(cw).reshape(-1)
                out[i, : len(vec)] = vec.astype(np.float32)
            except Exception:
                continue
        return out
