#!/usr/bin/env python3
"""Acquire every unprocessed YouTube URL listed in Dataset/metadata/youtube_links.csv.

A row whose status is processed is not downloaded again. After the clips exist,
each clip is run through feature and cue extraction unless extract_status is
already extracted. If YouTube refuses the download, the row is marked
download_error, the reason is loggerged, and the next URL is tried.

Clips and extracted records are stored as:

    Dataset/raw_videos/<youtube_id>/source.mp4
    Dataset/raw_videos/<youtube_id>/clip_000.mp4
    Dataset/videos/<youtube_id>_clip0.mp4
    Dataset/extracted/features/<youtube_id>/clip_000/
    Dataset/metadata/queries.json   (Gemini cues, merged in for these clips)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from core.constants import *
from utils.loaders import load_rows, save_rows
from orchestrator import process_links
from tests.pipeline_self_test import self_test
from utils.logger import setup_logger

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="YouTube URL to add to the links CSV, then process with every other unfinished row",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--clip-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--clip-duration", type=int, default=30)
    parser.add_argument(
        "--processing-level", choices=("low", "high"), default="low",
        help="Default annotation level; overridden by CSV processing_level column",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract rows already marked done")
    parser.add_argument("--self-test", action="store_true", help="Run the skip/error/naming checks and exit")
    parser.add_argument("--logger-level", default="INFO")
    args = parser.parse_args(argv)
    LOGGER = setup_logger(args.logger_level)
    
    if args.self_test:
        return self_test()
    
    if args.clip_duration <= 0:
        parser.error("--clip-duration must be a positive number of seconds")
    
    if args.url:
        rows, fieldnames = load_rows(args.csv)
        known = {row.get("youtube_url", "").strip() for row in rows}
        added = 0
        for url in args.url:
            url = url.strip()
            if not url or url in known:
                continue
            rows.append({"youtube_url": url, "processing_level": args.processing_level})
            known.add(url)
            added += 1
        if added:
            save_rows(args.csv, rows, fieldnames)
            LOGGER.info("Added %d new link(s) to %s", added, args.csv)
    
    LOGGER.info("Reading YouTube links from %s", args.csv)
    
    counts = process_links(
        args.csv,
        raw_dir=args.raw_dir,
        videos_dir=args.videos_dir,
        clip_index_path=args.clip_index,
        extract_dir=args.extract_dir,
        queries_path=args.queries,
        clip_duration=args.clip_duration,
        force=args.force,
        processing_level=args.processing_level,
    )
    LOGGER.info(
        "Finished youtube_links: %d downloaded, %d extracted, %d skipped, %d download errors, %d extract errors",
        counts["acquired"], counts["extracted"], counts["skipped"], counts["download_error"], counts["extract_error"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
