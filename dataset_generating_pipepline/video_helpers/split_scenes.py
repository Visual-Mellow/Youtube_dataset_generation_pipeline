#!/usr/bin/env python3
"""
split_scenes.py

Standalone helper: split raw flat MP4s into scene clips using PySceneDetect.
The main YouTube pipeline uses ``video_helpers.segment_clips`` instead; this
script is for flat ``Dataset/raw_videos/*.mp4`` layouts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from typing import List, Set, Tuple

from scenedetect import ContentDetector, SceneManager, open_video

from core.config import config
from utils.loaders import get_video_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("scene_splitter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw videos into semantic scene clips."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Unused legacy flag kept for CLI compatibility.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="ContentDetector threshold (default: pipeline_config segmentation.scene_threshold)",
    )
    return parser.parse_args()


def detect_scenes_pyscenedetect(
    video_path: str, threshold: float
) -> List[Tuple[float, float]]:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    logger.info(
        "Analyzing %s for scene changes (threshold=%s)...",
        video_path,
        threshold,
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]


def extract_clip(
    input_path: str, output_path: str, start_sec: float, end_sec: float
) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-avoid_negative_ts",
        "make_zero",
        "-loglevel",
        "error",
        output_path,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return True
        logger.error("FFmpeg split failed: %s", res.stderr)
        return False
    except Exception as e:
        logger.error("Failed to execute FFmpeg clip extraction: %s", e)
        return False


def _existing_parent_ids(meta_dir: str) -> Set[str]:
    """Return parent video IDs that already have metadata under meta_dir."""
    parents: Set[str] = set()
    if not os.path.isdir(meta_dir):
        return parents
    for name in os.listdir(meta_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(meta_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            parent = data.get("parent_video_id")
            if parent:
                parents.add(str(parent))
        except (OSError, json.JSONDecodeError):
            continue
    return parents


def run_splitter(*, threshold: float | None = None) -> None:
    """Split raw videos in ``config.raw_dir``; skip already-processed parents."""
    raw_dir = config.raw_dir
    clips_dir = config.clips_dir
    meta_dir = config.meta_dir
    seg = config.segmentation
    scene_threshold = float(threshold if threshold is not None else seg.scene_threshold)
    min_clip_duration = float(seg.min_tail_s)
    max_clip_duration = float(seg.max_clip_s)

    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    if not os.path.isdir(raw_dir):
        logger.error("Raw videos directory does not exist: %s", raw_dir)
        return

    processed_parents = _existing_parent_ids(str(meta_dir))
    if processed_parents:
        logger.info(
            "Found %d parent video(s) already represented in metadata — will skip them.",
            len(processed_parents),
        )

    raw_videos: List[str] = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".mp4"):
            continue
        video_id = os.path.splitext(name)[0]
        if video_id in processed_parents:
            logger.info(
                "Skipping %s: already processed (clips exist in metadata).",
                name,
            )
            continue
        raw_videos.append(os.path.join(raw_dir, name))

    logger.info("Found %d raw video(s) to process.", len(raw_videos))
    if not raw_videos:
        return

    total_clips = 0

    for idx, video_path in enumerate(raw_videos, start=1):
        video_id = get_video_id(video_path)
        parent_video_path = f"{video_id}.mp4"

        logger.info("[%d/%d] Processing video: %s", idx, len(raw_videos), video_id)

        try:
            scenes = detect_scenes_pyscenedetect(video_path, scene_threshold)
        except Exception as e:
            logger.error("Scene detection failed for %s: %s", video_id, e)
            continue

        valid_count = 0
        for s_idx, (start_sec, end_sec) in enumerate(scenes, start=1):
            duration = end_sec - start_sec
            if not (min_clip_duration <= duration <= max_clip_duration):
                logger.debug(
                    "  Skipping scene %d: duration %.2fs outside [%.1f, %.1f]s.",
                    s_idx,
                    duration,
                    min_clip_duration,
                    max_clip_duration,
                )
                continue

            clip_id = f"{video_id}_{s_idx:04d}"
            clip_filepath = os.path.join(clips_dir, f"{clip_id}.mp4")
            meta_filepath = os.path.join(meta_dir, f"{clip_id}.json")

            if os.path.isfile(clip_filepath) and os.path.isfile(meta_filepath):
                logger.debug("  Clip %s already exists — skipping extract.", clip_id)
                valid_count += 1
                total_clips += 1
                continue

            logger.info(
                "  Extracting scene %04d (%.2fs -> %.2fs, duration: %.2fs)...",
                s_idx,
                start_sec,
                end_sec,
                duration,
            )
            if not extract_clip(video_path, clip_filepath, start_sec, end_sec):
                logger.error("  Failed to extract scene %04d from %s.", s_idx, video_id)
                continue

            clip_meta = {
                "clip_id": clip_id,
                "audio_id": f"audio_{clip_id}",
                "parent_video_id": video_id,
                "parent_video_path": parent_video_path,
                "duration": round(duration, 3),
            }
            try:
                with open(meta_filepath, "w", encoding="utf-8") as f:
                    json.dump(clip_meta, f, indent=2, ensure_ascii=False)
                valid_count += 1
                total_clips += 1
            except Exception as e:
                logger.error("  Failed to save metadata for clip %s: %s", clip_id, e)

        logger.info(
            "Finished %s: %d valid clip(s). Parent path: %s",
            video_id,
            valid_count,
            parent_video_path,
        )

    logger.info(
        "Scene splitter completed. Generated/maintained %d clip(s) total.",
        total_clips,
    )


def main() -> None:
    args = parse_args()
    run_splitter(threshold=args.threshold)


if __name__ == "__main__":
    main()
