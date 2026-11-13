"""Pipeline configuration for LOW / HIGH signal-first cue annotation.

Load defaults from ``pipeline_config.json``, then apply CLI or CSV overrides.
A ``ProcessingProfile`` is a resolved view of one level (low or high) with
every numeric / model setting ready to use.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.schemas import *

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "pipeline_config.json"


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key) or {}
    return value if isinstance(value, dict) else {}

def _load_raw(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))

def _filter_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k in cls.__dataclass_fields__}


def with_level(cfg: PipelineConfig, level: ProcessingLevel | str) -> PipelineConfig:
    """Return a copy of ``cfg`` switched to another processing level's flags."""
    other = load_pipeline_config(level)
    return replace(
        cfg,
        level=other.level,
        flags=other.flags,
        perception=other.perception,
        snap=other.snap,
    )
    

def _build_perception(raw: dict[str, Any], anchor_version: int) -> PerceptionConfig:
    perc_raw = _section(raw, "perception")
    flat = {k: v for k, v in perc_raw.items() if k in PerceptionConfig.__dataclass_fields__ and k not in {
        "onsets_v2", "sed", "video_cues_v2", "fusion",
    }}
    flat["anchor_version"] = int(perc_raw.get("anchor_version", anchor_version))

    ov2_raw = _section(perc_raw, "onsets_v2")
    if "bands_hz" in ov2_raw:
        ov2_raw = dict(ov2_raw)
        ov2_raw["bands_hz"] = tuple(tuple(b) for b in ov2_raw["bands_hz"])
    onsets_v2 = OnsetsV2Config(**_filter_fields(OnsetsV2Config, ov2_raw)) if ov2_raw else OnsetsV2Config()

    sed = SedConfig(**_filter_fields(SedConfig, _section(perc_raw, "sed"))) if _section(perc_raw, "sed") else SedConfig()
    video = VideoCuesV2Config(**_filter_fields(VideoCuesV2Config, _section(perc_raw, "video_cues_v2"))) if _section(perc_raw, "video_cues_v2") else VideoCuesV2Config()
    fusion = FusionConfig(**_filter_fields(FusionConfig, _section(perc_raw, "fusion"))) if _section(perc_raw, "fusion") else FusionConfig()

    base = _filter_fields(PerceptionConfig, flat)
    return PerceptionConfig(
        **base,
        onsets_v2=onsets_v2,
        sed=sed,
        video_cues_v2=video,
        fusion=fusion,
    )


def load_pipeline_config(
    level: ProcessingLevel | str = "low",
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load ``pipeline_config.json`` and resolve a LOW or HIGH profile."""
    raw = _load_raw(config_path)
    if overrides:
        merged = dict(raw)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        raw = merged

    resolved_level: ProcessingLevel = "high" if str(level).strip().lower() == "high" else "low"
    anchor_version = int(raw.get("anchor_version", _section(raw, "perception").get("anchor_version", 2)))

    models = ModelConfig(**_filter_fields(ModelConfig, _section(raw, "models")))
    segmentation = SegmentationConfig(**_filter_fields(SegmentationConfig, _section(raw, "segmentation")))
    windows = WindowConfig(**_filter_fields(WindowConfig, _section(raw, "windows")))
    snap = SnapConfig(**_filter_fields(SnapConfig, _section(raw, "snap")))
    perception = _build_perception(raw, anchor_version)

    level_raw = _section(_section(raw, "levels"), resolved_level)
    flags = LevelFlags(**_filter_fields(LevelFlags, level_raw))

    # Level-specific SED backend override.
    if flags.sed_backend:
        perception = replace(
            perception,
            sed=replace(perception.sed, backend=flags.sed_backend),
        )
    if not flags.video_cues_v2:
        perception = replace(
            perception,
            video_cues_v2=replace(perception.video_cues_v2, enabled=False),
        )
    if flags.snap_v2 and snap.version < 2:
        snap = replace(snap, version=2)
    if not flags.snap_v2:
        snap = replace(snap, version=1)

    return PipelineConfig(
        level=resolved_level,
        anchor_version=anchor_version,
        models=models,
        segmentation=segmentation,
        windows=windows,
        snap=snap,
        perception=perception,
        flags=flags,
        apply_av_offset=bool(raw.get("apply_av_offset", False)),
    )


config = load_pipeline_config()