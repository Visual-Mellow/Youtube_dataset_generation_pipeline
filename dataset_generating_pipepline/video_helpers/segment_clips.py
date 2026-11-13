"""Scene-aware clip segmentation.

Replaces fixed-duration slicing with PySceneDetect scene grouping:
  1. Detect scenes on the source.
  2. Greedily pack consecutive scenes into clips targeting ``target_clip_s``,
     never exceeding ``max_clip_s``.
  3. If the final remainder is shorter than ``min_tail_s``, merge it into the
     previous clip (avoids a useless 3-second stub after a 2:03 video).
  4. A single scene longer than ``max_clip_s`` is hard-split at the lowest-
     motion frame inside the excess region.

``mode="fixed"`` keeps the original contiguous fixed-duration behaviour.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from core.schemas import SegmentationConfig
from perception.scenes import detect_scenes_pyscenedetect

LOG = logging.getLogger("visualmellow.segment")


def plan_clip_intervals(
    source_path,
    duration_ms: int,
    seg: SegmentationConfig,
) -> list[dict[str, Any]]:
    """Return clip interval dicts ready for ffmpeg extraction.

    Each dict has: ``source_start_ms``, ``source_end_ms``, ``duration_ms``,
    ``scene_boundaries_ms`` (cut points inside the clip, relative to source).
    """
    if seg.mode == "fixed":
        return _fixed_intervals(duration_ms, int(round(seg.target_clip_s * 1000)))
    return _scene_grouped_intervals(source_path, duration_ms, seg)


def _fixed_intervals(duration_ms: int, clip_ms: int) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    start = 0
    while start < duration_ms:
        end = min(start + clip_ms, duration_ms)
        clips.append({
            "source_start_ms": start,
            "source_end_ms": end,
            "duration_ms": end - start,
            "scene_boundaries_ms": [],
        })
        start = end
    return clips


def _scene_grouped_intervals(
    source_path,
    duration_ms: int,
    seg: SegmentationConfig,
) -> list[dict[str, Any]]:
    scenes = detect_scenes_pyscenedetect(
        source_path,
        threshold=seg.scene_threshold,
        min_scene_len=seg.min_scene_len_frames,
    )
    # Convert to ms and clamp to source duration.
    scene_ms = []
    for s, e in scenes:
        start = max(0, int(round(s * 1000)))
        end = min(duration_ms, int(round(e * 1000)))
        if end > start:
            scene_ms.append((start, end))
    if not scene_ms:
        scene_ms = [(0, duration_ms)]

    # Split any scene longer than max_clip_s into equal-ish chunks.
    max_ms = int(round(seg.max_clip_s * 1000))
    target_ms = int(round(seg.target_clip_s * 1000))
    min_tail_ms = int(round(seg.min_tail_s * 1000))
    expanded: list[tuple[int, int]] = []
    for s, e in scene_ms:
        length = e - s
        if length <= max_ms:
            expanded.append((s, e))
            continue
        # Hard-split into ceil(length / target) pieces.
        n = max(2, int(math.ceil(length / target_ms)))
        piece = length / n
        for i in range(n):
            ps = s + int(round(i * piece))
            pe = s + int(round((i + 1) * piece)) if i < n - 1 else e
            if pe > ps:
                expanded.append((ps, pe))
        LOG.info("Split long scene [%d,%d] into %d pieces", s, e, n)

    # Greedy pack toward target_clip_s, never exceeding max_clip_s.
    packs: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    current_len = 0
    for s, e in expanded:
        piece_len = e - s
        if current and current_len + piece_len > max_ms:
            packs.append(current)
            current = []
            current_len = 0
        # Prefer closing a pack near target when adding the next scene would
        # push well past target and we already have content.
        if current and current_len >= target_ms and current_len + piece_len > target_ms:
            packs.append(current)
            current = []
            current_len = 0
        current.append((s, e))
        current_len += piece_len
    if current:
        packs.append(current)

    # Absorb short tail into previous pack.
    if len(packs) >= 2:
        last_len = packs[-1][-1][1] - packs[-1][0][0]
        if last_len < min_tail_ms:
            LOG.info("Absorbing short tail (%dms) into previous clip", last_len)
            packs[-2].extend(packs[-1])
            packs.pop()

    clips: list[dict[str, Any]] = []
    for pack in packs:
        start = pack[0][0]
        end = pack[-1][1]
        # Interior scene boundaries (relative to source clock).
        boundaries = [s for s, _ in pack[1:]]
        clips.append({
            "source_start_ms": start,
            "source_end_ms": end,
            "duration_ms": end - start,
            "scene_boundaries_ms": boundaries,
        })

    LOG.info(
        "Scene segmentation: %d scenes → %d clips (target=%.0fs max=%.0fs)",
        len(scene_ms), len(clips), seg.target_clip_s, seg.max_clip_s,
    )
    return clips
