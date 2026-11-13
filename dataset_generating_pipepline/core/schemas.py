from dataclasses import dataclass, field
from typing import Any, Literal
from core.constants import *

ProcessingLevel = Literal["low", "high"]

@dataclass(frozen=True)
class SegmentationConfig:
    """How source videos are cut into dataset clips."""

    mode: str = "scene"  # "scene" | "fixed"
    target_clip_s: float = 30.0
    max_clip_s: float = 40.0
    min_tail_s: float = 12.0
    scene_threshold: float = 27.0
    min_scene_len_frames: int = 15


@dataclass(frozen=True)
class WindowConfig:
    """Proposer sub-window sizes (HIGH level)."""

    window_ms: int = 6000
    window_overlap_ms: int = 1000
    cut_snap_tol_ms: int = 1500
    max_anchors_per_window: int = 40


@dataclass(frozen=True)
class SnapConfig:
    """Verifier / snap tolerances in milliseconds."""

    version: int = 2
    sfx_snap_ms: int = 250
    bed_snap_ms: int = 600
    audibility_tol_ms: int = 150
    min_impact_gap_ms: int = 150
    dedupe_ms: int = 40
    near_cut_tol_ms: int = 300
    assign_window_ms: int = 400
    max_cost: float = 0.85
    w_t: float = 0.45
    w_c: float = 0.30
    w_s: float = 0.15
    w_stem: float = 0.10


@dataclass(frozen=True)
class OnsetsV2Config:
    enabled: bool = True
    sr_dsp: int = 32000
    n_fft: int = 1024
    hop: int = 160
    win_length: int = 512
    bands_hz: tuple = ((0, 500), (500, 2000), (2000, 8000), (8000, 16000))
    mad_k: float = 3.5
    median_win_ms: float = 600
    min_gap_ms: float = 35
    dense_min_gap_ms: float = 20
    cross_band_merge_ms: float = 25
    refine: dict = field(default_factory=lambda: {
        "pre_ms": 120, "post_ms": 60, "env_smooth_ms": 2, "floor_db": 6, "min_snr_db": 6,
    })
    offset: dict = field(default_factory=lambda: {
        "decay_db_below_peak": 30, "max_event_ms": {"impact": 1500, "default": 3000},
    })

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sr_dsp": self.sr_dsp,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "win_length": self.win_length,
            "bands_hz": [list(b) for b in self.bands_hz],
            "mad_k": self.mad_k,
            "median_win_ms": self.median_win_ms,
            "min_gap_ms": self.min_gap_ms,
            "dense_min_gap_ms": self.dense_min_gap_ms,
            "cross_band_merge_ms": self.cross_band_merge_ms,
            "refine": dict(self.refine),
            "offset": dict(self.offset),
        }


@dataclass(frozen=True)
class SedConfig:
    enabled: bool = True
    backend: str = "panns_cnn14_sed"
    on_thr: float = 0.5
    off_thr: float = 0.3
    min_dur_ms: float = 40
    merge_gap_ms: float = 80
    gate_search_ms: float = 250
    median_frames: int = 3
    class_thresholds: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "on_thr": self.on_thr,
            "off_thr": self.off_thr,
            "min_dur_ms": self.min_dur_ms,
            "merge_gap_ms": self.merge_gap_ms,
            "gate_search_ms": self.gate_search_ms,
            "median_frames": self.median_frames,
            "class_thresholds": dict(self.class_thresholds),
        }


@dataclass(frozen=True)
class VideoCuesV2Config:
    enabled: bool = True
    use_farneback: bool = True
    use_transnet: bool = False
    # Mean pixel-intensity |Δ| floor (0–255 scale). Notebook-validated at 5.
    min_intensity_change: float = 5.0


@dataclass(frozen=True)
class FusionConfig:
    audio_cluster_ms: float = 40
    visual_attach_ms: float = 100
    w_onset: float = 0.35
    w_sed: float = 0.30
    w_snr: float = 0.15
    w_visual: float = 0.15
    w_beat: float = 0.05


@dataclass(frozen=True)
class PerceptionConfig:
    """Layer-0 signal-processing knobs."""

    sample_rate: int = 16000
    onset_hop_ms: int = 10
    onset_delta: float = 0.6
    onset_min_gap_ms: int = 100
    tagger_topk: int = 5
    whisper_model: str = "small"
    gemini_video_fps: int = 5
    anchor_fps: int = 15
    anchor_version: int = 2
    onsets_v2: OnsetsV2Config = field(default_factory=OnsetsV2Config)
    sed: SedConfig = field(default_factory=SedConfig)
    video_cues_v2: VideoCuesV2Config = field(default_factory=VideoCuesV2Config)
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass(frozen=True)
class ModelConfig:
    """Gemini model names used by each LLM stage."""

    explain: str = "gemini-3-flash-preview"
    proposer: str = "gemini-3-flash-preview"
    judge: str = "gemini-3.1-pro-preview"


@dataclass(frozen=True)
class LevelFlags:
    """Feature toggles that differ between LOW and HIGH."""

    audio_only_pass: bool = False
    llm_judge: bool = False
    separation: bool = False
    per_window_proposer: bool = False
    sed_backend: str = "windowed_tagger"
    video_cues_v2: bool = True
    snap_v2: bool = True
    description: str = ""


@dataclass(frozen=True)
class PipelineConfig:
    """Fully resolved pipeline config for one processing level."""

    level: ProcessingLevel = "low"
    anchor_version: int = 2
    models: ModelConfig = field(default_factory=ModelConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    snap: SnapConfig = field(default_factory=SnapConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    flags: LevelFlags = field(default_factory=LevelFlags)
    apply_av_offset: bool = False
    raw_dir: Path = DEFAULT_RAW
    videos_dir: Path = DEFAULT_VIDEOS
    clips_dir: Path = DEFAULT_CLIPS
    clip_index_path: Path = DEFAULT_INDEX
    extract_dir: Path = DEFAULT_EXTRACTED
    queries_path: Path = DEFAULT_QUERIES
    meta_dir: Path = DEFAULT_METADATA

    @property
    def is_high(self) -> bool:
        return self.level == "high"
