"""Layer 0 — local, deterministic perception.

DSP owns time. This package extracts cuts, motion peaks, audio onsets,
beat grids, speech word timestamps, optional Demucs stems, and AudioSet
frame tags, then merges them into a single ``anchors.json`` timeline.
"""
from __future__ import annotations

from perception.anchors import build_anchors, load_or_build_anchors

__all__ = ["build_anchors", "load_or_build_anchors"]
