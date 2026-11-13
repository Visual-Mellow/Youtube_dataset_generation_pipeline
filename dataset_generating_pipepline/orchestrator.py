from pathlib import Path
from typing import Any, Callable

from core.constants import *
from utils.loaders import load_rows, save_rows
from utils.downloaders import download_error_text

from video_helpers.video_extraction import extract_video
from utils.ui_log import UILogWriter, render_html
from utils.loaders import is_processed, is_extracted, publish_clips, upsert_clip_index
import video_helpers.download_and_clip as acquire

def process_links(
    csv_path: Path,
    *,
    raw_dir: Path,
    videos_dir: Path,
    clip_index_path: Path,
    extract_dir: Path,
    queries_path: Path,
    clip_duration: int,
    force: bool,
    processing_level: str = "low",
    acquire_fn: Callable[..., dict[str, Any]] = acquire.acquire_url,
    extract_fn: Callable[..., int] = extract_video,
) -> dict[str, int]:
    """Download missing videos, then extract features and cues for clips that are not extracted yet."""
    rows, fieldnames = load_rows(csv_path)
    counts = {"skipped": 0, "acquired": 0, "download_error": 0, "empty": 0, "extracted": 0, "extract_error": 0}
    if not rows:
        return counts
    batch_log = UILogWriter(
        csv_path.parent / "youtube_links.ui.log",
        fresh=force,
        meta={"title": "YouTube links batch", "csv": str(csv_path), "processing_level": processing_level},
    )
    batch_log.step_begin("batch", "Process youtube_links.csv", inputs={"row_count": len(rows)})
    for row in rows:
        url = row.get("youtube_url", "").strip()
        if not url:
            counts["empty"] += 1
            continue
        # Per-row level override from CSV, else CLI default.
        row_level = (row.get("processing_level") or processing_level or "low").strip().lower()
        if row_level not in {"low", "high"}:
            row_level = "low"
        downloaded = is_processed(row.get("status", "")) and not force
        if downloaded and is_extracted(row.get("extract_status", "")) and not force:
            LOG.info("Skipping video that is already downloaded and extracted %s", url)
            batch_log.message(f"Skipped (already done): {url}")
            counts["skipped"] += 1
            continue
        video_step = f"video:{row.get('video_id') or url}"
        batch_log.step_begin(video_step, f"Row: {url}", inputs={"row": {k: row.get(k, "") for k in COLUMNS}})
        if not downloaded:
            LOG.info("Acquiring %s", url)
            batch_log.step_begin(f"{video_step}:download", "Download + clip", inputs={"url": url, "clip_duration": clip_duration})
            try:
                metadata = acquire_fn(url, raw_dir, clip_duration, force)
            except Exception as exc:
                message = download_error_text(exc)
                row["status"] = "download_error"
                row["error"] = message
                row["video_id"] = row.get("video_id") or acquire.youtube_id_from_url(url) or ""
                row["processed_at"] = acquire.now()
                save_rows(csv_path, rows, fieldnames)
                counts["download_error"] += 1
                LOG.error("%s | %s", message, url)
                batch_log.step_end(f"{video_step}:download", message, outputs={"error": message})
                batch_log.step_end(video_step, "download_error")
                continue
            video_id = str(metadata["video_id"])
            clips = list(metadata.get("clips") or [])
            folder = raw_dir / video_id
            published = publish_clips(video_id, folder, clips, videos_dir)
            upsert_clip_index(clip_index_path, video_id, url, published)
            row["video_id"] = video_id
            row["status"] = "processed"
            row["error"] = ""
            row["n_clips"] = str(len(published))
            row["clips_dir"] = f"Dataset/raw_videos/{video_id}"
            row["processed_at"] = acquire.now()
            row["processing_level"] = row_level
            save_rows(csv_path, rows, fieldnames)
            counts["acquired"] += 1
            LOG.info("Downloaded %s: %d clips in %s and Dataset/videos/", video_id, len(published), row["clips_dir"])
            batch_log.step_end(
                f"{video_step}:download",
                f"{len(published)} clips",
                outputs={"video_id": video_id, "n_clips": len(published), "clips_dir": row["clips_dir"]},
            )
        video_id = row.get("video_id") or acquire.youtube_id_from_url(url) or ""
        if not video_id:
            row["extract_status"] = "extract_error"
            row["extract_error"] = "clip extraction failed: missing video id"
            save_rows(csv_path, rows, fieldnames)
            counts["extract_error"] += 1
            LOG.error("clip extraction failed: missing video id | %s", url)
            batch_log.step_end(video_step, "missing video_id")
            continue
        LOG.info("Extracting features and cues for %s (level=%s)", video_id, row_level)
        batch_log.step_begin(
            f"{video_step}:extract",
            f"Extract {video_id}",
            inputs={"level": row_level, "extract_dir": str(extract_dir)},
        )
        try:
            try:
                n_queries = extract_fn(
                    video_id, raw_dir, extract_dir, queries_path, force,
                    processing_level=row_level,
                )
            except TypeError:
                # Test doubles may not accept processing_level kwarg.
                n_queries = extract_fn(video_id, raw_dir, extract_dir, queries_path, force)
        except Exception as exc:
            message = str(exc).splitlines()[-1][:500]
            row["extract_status"] = "extract_error"
            row["extract_error"] = message
            save_rows(csv_path, rows, fieldnames)
            counts["extract_error"] += 1
            LOG.error("clip extraction failed for %s: %s", video_id, message)
            batch_log.step_end(f"{video_step}:extract", message, outputs={"error": message})
            batch_log.step_end(video_step, "extract_error")
            continue
        row["video_id"] = video_id
        row["extract_status"] = "extracted"
        row["extract_error"] = ""
        row["processing_level"] = row_level
        save_rows(csv_path, rows, fieldnames)
        counts["extracted"] += 1
        LOG.info("Extracted %s: %d grounding queries written", video_id, n_queries)
        batch_log.step_end(
            f"{video_step}:extract",
            f"{n_queries} queries",
            outputs={"n_queries": n_queries, "per_clip_ui_logs": f"Dataset/extracted/features/{video_id}/*/annotations/{row_level}/pipeline.ui.log"},
        )
        batch_log.step_end(video_step, "ok")
    batch_log.step_end("batch", "finished", outputs=counts)
    try:
        render_html(batch_log.path)
    except Exception as exc:
        LOG.debug("Could not render batch UI log HTML: %s", exc)
    return counts
