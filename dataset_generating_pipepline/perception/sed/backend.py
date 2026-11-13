"""SED backend protocol and registry."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SEDBackend(Protocol):
    name: str
    frame_hop_s: float
    native_resolution_s: float
    classes: list[str]

    def predict(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Return frame probabilities shaped [T, C] in 0..1."""
        ...


def get_backend(name: str) -> SEDBackend | None:
    name = (name or "panns_cnn14_sed").strip().lower()
    if name in {"panns_cnn14_sed", "panns", "cnn14"}:
        from perception.sed.panns_sed import PANNsSED

        try:
            return PANNsSED()
        except Exception:
            from perception.sed.windowed_tagger import WindowedTaggerSED

            return WindowedTaggerSED()
    if name in {"windowed_tagger", "tagger", "off", "none"}:
        if name in {"off", "none"}:
            return None
        from perception.sed.windowed_tagger import WindowedTaggerSED

        return WindowedTaggerSED()
    from perception.sed.windowed_tagger import WindowedTaggerSED

    return WindowedTaggerSED()
