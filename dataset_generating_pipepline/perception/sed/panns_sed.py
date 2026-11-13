"""PANNs CNN14 sound-event / decision-level backend."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger("visualmellow.perception.sed.panns")


class PANNsSED:
    """Prefer SoundEventDetection; fall back to clip tagger windowing."""

    name = "panns_cnn14_sed"
    frame_hop_s = 0.01
    native_resolution_s = 0.16  # ~16x mel hop downsample honesty

    def __init__(self) -> None:
        from perception.tagger import _ensure_panns_assets, _has_cuda

        _ensure_panns_assets()
        self._sed = None
        self._tagger = None
        self.classes: list[str] = []
        try:
            from panns_inference import SoundEventDetection

            device = "cuda" if _has_cuda() else "cpu"
            self._sed = SoundEventDetection(checkpoint_path=None, device=device)
            self.classes = list(getattr(self._sed, "labels", []) or [])
            LOG.info("PANNs SoundEventDetection loaded (%d classes)", len(self.classes))
        except Exception as exc:
            LOG.warning("SoundEventDetection unavailable (%s); using AudioTagging windows", exc)
            from panns_inference import AudioTagging

            device = "cuda" if _has_cuda() else "cpu"
            self._tagger = AudioTagging(checkpoint_path=None, device=device)
            self.classes = list(getattr(self._tagger, "labels", []) or [])
            self.frame_hop_s = 0.1
            self.native_resolution_s = 1.0
            self.name = "panns_cnn14_windowed"

    def predict(self, wav: np.ndarray, sr: int) -> np.ndarray:
        waveform = wav.astype(np.float32)
        if sr != 32000:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=32000)

        if self._sed is not None:
            # panns SoundEventDetection.inference → framewise [batch, T, C] or similar
            try:
                framewise = self._sed.inference(waveform[None, :])
                arr = np.asarray(framewise)
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.ndim != 2:
                    raise RuntimeError(f"Unexpected SED shape {arr.shape}")
                # Interpolate to ~10 ms if coarser.
                return self._to_10ms(arr)
            except Exception as exc:
                LOG.warning("SED inference failed (%s); empty", exc)
                return np.zeros((1, max(1, len(self.classes))), dtype=np.float32)

        # Windowed clip tagger at 100 ms hop / 1.0 s window.
        return self._windowed(waveform)

    def _to_10ms(self, framewise: np.ndarray) -> np.ndarray:
        # Assume native ~10-32 ms; resample time axis to 10 ms grid length estimate.
        T, C = framewise.shape
        if T <= 1:
            return framewise.astype(np.float32)
        # Leave as-is; infer.py uses backend.frame_hop_s for time mapping.
        # Detect hop from duration if we knew it — keep array, set hop from model.
        self.frame_hop_s = max(0.01, self.native_resolution_s / 16.0) if T > 100 else 0.032
        return framewise.astype(np.float32)

    def _windowed(self, waveform_32k: np.ndarray) -> np.ndarray:
        assert self._tagger is not None
        hop_s = 0.1
        win_s = 1.0
        self.frame_hop_s = hop_s
        n = int(np.ceil(len(waveform_32k) / 32000 / hop_s))
        C = len(self.classes) or 527
        out = np.zeros((max(n, 1), C), dtype=np.float32)
        for i in range(n):
            start = int(i * hop_s * 32000)
            end = min(len(waveform_32k), start + int(win_s * 32000))
            if end - start < 32000 // 4:
                break
            chunk = waveform_32k[start:end]
            if len(chunk) < int(win_s * 32000):
                chunk = np.pad(chunk, (0, int(win_s * 32000) - len(chunk)))
            cw, _ = self._tagger.inference(chunk[None, :])
            vec = np.asarray(cw).reshape(-1)
            out[i, : len(vec)] = vec.astype(np.float32)
        return out
