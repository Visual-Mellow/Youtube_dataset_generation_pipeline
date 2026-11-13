import csv
import os
import shutil
from pathlib import Path
from typing import Any
from core.constants import *

def is_processed(status: str) -> bool:
    return status.strip().lower() in PROCESSED

def is_extracted(status: str) -> bool:
    return status.strip().lower() in EXTRACTED

def get_video_id(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]

def upsert_clip_index(path: Path, video_id: str, youtube_url: str, published: list[dict[str, Any]]) -> None:
    """Replace this video's rows in clip_index.csv and leave every other video alone."""
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = [{k: (v or "") for k, v in row.items() if k} for row in csv.DictReader(f)]
    kept = [row for row in existing if row.get("video_id") != video_id]
    for item in published:
        kept.append({**item, "youtube_url": youtube_url})
    save_rows(path, kept, INDEX_COLUMNS)

def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read the link sheet. Missing file is created with a header and no rows."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_rows(path, [], COLUMNS)
        LOG.info("Created %s. Add a youtube_url on each row, then run this again.", path)
        return [], list(COLUMNS)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [{k: (v or "") for k, v in row.items() if k} for row in reader]
    for name in COLUMNS:
        if name not in fieldnames:
            fieldnames.append(name)
    return rows, fieldnames


def save_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Replace the sheet atomically so a killed run cannot leave a half-written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def publish_clips(video_id: str, folder: Path, clips: list[dict[str, Any]], videos_dir: Path) -> list[dict[str, Any]]:
    """Hard-link each raw clip to Dataset/videos/<youtube_id>_clip<index>.mp4."""
    videos_dir.mkdir(parents=True, exist_ok=True)
    published = []
    for clip in clips:
        index = int(str(clip["clip_id"]).split("_")[-1])
        src = folder / clip["filename"]
        filename = f"{video_id}_clip{index}.mp4"
        dest = videos_dir / filename
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        published.append({
            "video_id": video_id,
            "clip_index": str(index),
            "filename": filename,
            "video_path": f"Dataset/videos/{filename}",
            "source_start_ms": str(clip.get("source_start_ms", "")),
            "source_end_ms": str(clip.get("source_end_ms", "")),
        })
    return published


