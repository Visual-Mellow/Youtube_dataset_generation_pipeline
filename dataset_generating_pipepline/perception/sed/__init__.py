"""Frame-level sound event detection (Layer-0 v2).

Pretrained SED backbones have coarse native time resolution (often 100-160 ms).
SED provides class and approximate extent; DSP refinement (onsets_v2) owns exact
edges. Never treat SED boundaries as millisecond-accurate.
"""
from __future__ import annotations

from perception.sed.infer import run_sed
from perception.sed.events import extract_events, gate_with_onsets

__all__ = ["run_sed", "extract_events", "gate_with_onsets"]
