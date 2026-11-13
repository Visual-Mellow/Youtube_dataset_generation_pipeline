import os
import sys
import logging
from pathlib import Path


# core/constants.py → package root → repository root
PIPELINE_DIR = Path(__file__).resolve().parents[1]
RL_ROOT = PIPELINE_DIR.parent

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

# librosa/numba otherwise try to write caches inside site-packages.
os.environ.setdefault("NUMBA_CACHE_DIR", str(RL_ROOT / ".numba_cache"))


LOG = logging.getLogger("visualmellow.youtube_links")

COLUMNS = [
    "youtube_url",
    "video_id",
    "status",
    "error",
    "n_clips",
    "clips_dir",
    "processed_at",
    "extract_status",
    "extract_error",
    "processing_level",
]
PROCESSED = {"processed", "done", "complete", "completed"}
EXTRACTED = {"extracted", "done", "complete", "completed"}
DEFAULT_CSV = RL_ROOT / "Dataset" / "metadata" / "youtube_links.csv"
DEFAULT_RAW = RL_ROOT / "Dataset" / "raw_videos"
DEFAULT_VIDEOS = RL_ROOT / "Dataset" / "videos"
DEFAULT_EXTRACTED = RL_ROOT / "Dataset" / "extracted"
DEFAULT_QUERIES = RL_ROOT / "Dataset" / "metadata" / "queries.json"
DEFAULT_INDEX = RL_ROOT / "Dataset" / "metadata" / "clip_index.csv"
INDEX_COLUMNS = [
    "video_id",
    "clip_index",
    "filename",
    "video_path",
    "source_start_ms",
    "source_end_ms",
    "youtube_url",
]

DEFAULT_CLIPS = RL_ROOT / "Dataset" / "clips"
DEFAULT_METADATA = RL_ROOT / "Dataset" / "metadata"
