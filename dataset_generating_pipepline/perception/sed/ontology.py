"""Map AudioSet labels → pipeline audio_type + acoustic group."""
from __future__ import annotations

from typing import Any

# Acoustic groups drive band selection for refinement and max_event_ms.
GROUP_BANDS = {
    "impact": ["hf", "mid"],
    "tonal": ["lowmid", "lf"],
    "noise_burst": ["hf", "mid", "lowmid"],
    "continuous_bed": ["lf", "lowmid"],
    "speech": ["lowmid", "mid"],
}

# Substring → (audio_type, group)
_RULES: list[tuple[str, str, str]] = [
    ("speech", "DIALOGUE", "speech"),
    ("conversation", "DIALOGUE", "speech"),
    ("narration", "DIALOGUE", "speech"),
    ("male speech", "DIALOGUE", "speech"),
    ("female speech", "DIALOGUE", "speech"),
    ("music", "MUSIC", "continuous_bed"),
    ("singing", "MUSIC", "tonal"),
    ("musical", "MUSIC", "tonal"),
    ("drum", "SFX", "impact"),
    ("gunshot", "SFX", "impact"),
    ("explosion", "SFX", "impact"),
    ("bang", "SFX", "impact"),
    ("boom", "SFX", "impact"),
    ("slap", "SFX", "impact"),
    ("knock", "SFX", "impact"),
    ("glass", "SFX", "impact"),
    ("shatter", "SFX", "impact"),
    ("clap", "SFX", "impact"),
    ("click", "SFX", "impact"),
    ("footstep", "SFX", "impact"),
    ("walk", "SFX", "impact"),
    ("typewriter", "SFX", "impact"),
    ("typing", "SFX", "impact"),
    ("whoosh", "SFX", "noise_burst"),
    ("wind", "AMBIENCE", "continuous_bed"),
    ("rain", "AMBIENCE", "continuous_bed"),
    ("water", "AMBIENCE", "noise_burst"),
    ("engine", "SFX", "continuous_bed"),
    ("vehicle", "SFX", "continuous_bed"),
    ("car", "SFX", "continuous_bed"),
    ("crowd", "AMBIENCE", "continuous_bed"),
    ("silence", "SFX", "noise_burst"),
]


def map_label(label: str) -> dict[str, Any]:
    low = (label or "").lower()
    for needle, audio_type, group in _RULES:
        if needle in low:
            return {
                "label": label,
                "audio_type": audio_type,
                "group": group,
                "bands": list(GROUP_BANDS.get(group, ["mid", "hf"])),
            }
    return {
        "label": label,
        "audio_type": "SFX",
        "group": "noise_burst",
        "bands": list(GROUP_BANDS["noise_burst"]),
    }


def class_thresholds(overrides: dict[str, float] | None = None) -> dict[str, float]:
    base = {
        "Speech": 0.35,
        "Music": 0.4,
        "Gunshot, gunfire": 0.25,
        "Explosion": 0.25,
        "Glass": 0.3,
        "Walk, footsteps": 0.3,
    }
    if overrides:
        base.update({str(k): float(v) for k, v in overrides.items()})
    return base
