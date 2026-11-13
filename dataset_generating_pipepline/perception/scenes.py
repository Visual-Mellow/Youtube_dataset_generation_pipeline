"""Visual cut + motion-peak detection.

Two complementary detectors:

1. **PySceneDetect ContentDetector** — HSV-based cut detection with a
   minimum scene length. Preferred for clip segmentation.
2. **ffmpeg scene_score** — per-frame mean-absolute frame difference at
   ``anchor_fps``. Cuts are frames with score ≥ ``cut_score``; motion peaks
   are local maxima of the non-cut scores above ``mean + 1.5σ``.

Ported and cleaned from ``visual_high_thinking._pick_anchors``.
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

LOG = logging.getLogger("visualmellow.perception.scenes")

_CUT_SCORE = 0.35
_MIN_MOTION_SCORE = 0.012
_MIN_ANCHOR_GAP_MS = 180
_MAX_ANCHORS = 600


@dataclass
class VisualAnchor:
    """A visual cut or motion peak at millisecond resolution."""

    ms: int
    kind: str  # "cut" | "motion"
    strength: float


def detect_scenes_pyscenedetect(
    video_path: Path | str,
    *,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> list[tuple[float, float]]:
    """Return ``[(start_sec, end_sec), ...]`` scene intervals via ContentDetector.

    Falls back to a single full-length scene if PySceneDetect is missing or
    finds no cuts (so callers always get at least one interval).
    """
    video_path = Path(video_path)
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError:
        LOG.warning("PySceneDetect not installed; treating whole video as one scene")
        return _fallback_full_scene(video_path)

    try:
        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        manager.detect_scenes(video)
        scenes = manager.get_scene_list()
        if not scenes:
            return _fallback_full_scene(video_path)
        result = [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
        LOG.info("PySceneDetect: %d scenes in %s (threshold=%.1f)", len(result), video_path.name, threshold)
        return result
    except Exception as exc:
        LOG.warning("PySceneDetect failed (%s); falling back to full scene", exc)
        return _fallback_full_scene(video_path)


def _fallback_full_scene(video_path: Path) -> list[tuple[float, float]]:
    """Probe duration and return a single [0, duration] scene."""
    try:
        raw = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        duration = float(raw.stdout.strip() or 0)
    except Exception:
        duration = 0.0
    if duration <= 0:
        duration = 1.0
    return [(0.0, duration)]


def extract_visual_anchors(
    video_path: Path | str,
    *,
    fps: int = 15,
    cut_score: float = _CUT_SCORE,
) -> list[VisualAnchor]:
    """ffmpeg scene_score → cut + motion anchors at ~``1000/fps`` ms resolution."""
    video_path = Path(video_path)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as meta:
        meta_path = meta.name
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path), "-an",
            "-vf", f"fps={fps},scale=160:-2,select='gte(scene,0)',metadata=print:file={meta_path}",
            "-f", "null", "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=False)
        if res.returncode != 0:
            LOG.warning("Anchor extraction ffmpeg failed: %s", (res.stderr or b"")[:300])
            return []
        text = Path(meta_path).read_text(encoding="utf-8", errors="ignore")
        scores = _parse_scene_scores(text)
    except Exception as exc:
        LOG.warning("Visual anchor extraction failed: %s", exc)
        return []
    finally:
        try:
            Path(meta_path).unlink()
        except OSError:
            pass

    anchors = _pick_anchors(scores, cut_score=cut_score)
    LOG.info(
        "Visual anchors: %d (%d cuts, %d motion) from %d frames",
        len(anchors),
        sum(1 for a in anchors if a.kind == "cut"),
        sum(1 for a in anchors if a.kind == "motion"),
        len(scores),
    )
    return anchors


def _parse_scene_scores(text: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    t_ms: Optional[int] = None
    for line in text.splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            t_ms = int(round(float(m.group(1)) * 1000))
            continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m and t_ms is not None:
            out.append((t_ms, float(m.group(1))))
    return out


def _pick_anchors(
    scores: Sequence[tuple[int, float]],
    *,
    cut_score: float = _CUT_SCORE,
) -> list[VisualAnchor]:
    if len(scores) < 3:
        return []
    non_cut = [s for _, s in scores if s < cut_score]
    if non_cut:
        mean = sum(non_cut) / len(non_cut)
        var = sum((s - mean) ** 2 for s in non_cut) / len(non_cut)
        thr = max(_MIN_MOTION_SCORE, mean + 1.5 * var ** 0.5)
    else:
        thr = _MIN_MOTION_SCORE

    raw: list[VisualAnchor] = []
    for i, (ms, s) in enumerate(scores):
        if s >= cut_score:
            raw.append(VisualAnchor(ms, "cut", s))
            continue
        prev_s = scores[i - 1][1] if i > 0 else 0.0
        next_s = scores[i + 1][1] if i + 1 < len(scores) else 0.0
        if s >= thr and s >= prev_s and s >= next_s:
            raw.append(VisualAnchor(ms, "motion", s))

    # Enforce minimum gap; cuts always win over motion.
    merged: list[VisualAnchor] = []
    for a in sorted(raw, key=lambda x: x.ms):
        if merged and a.ms - merged[-1].ms < _MIN_ANCHOR_GAP_MS:
            last = merged[-1]
            if last.kind == "cut" and a.kind != "cut":
                continue
            if a.kind == "cut" or a.strength > last.strength:
                merged[-1] = a
            continue
        merged.append(a)

    if len(merged) > _MAX_ANCHORS:
        cuts = [a for a in merged if a.kind == "cut"]
        motion = sorted((a for a in merged if a.kind != "cut"), key=lambda x: -x.strength)
        merged = sorted(cuts + motion[: max(0, _MAX_ANCHORS - len(cuts))], key=lambda x: x.ms)
    return merged


def visual_anchors_as_dicts(anchors: list[VisualAnchor]) -> list[dict[str, Any]]:
    """Serialise VisualAnchor list for JSON."""
    return [{"t_ms": a.ms, "kind": a.kind, "strength": a.strength, "stem": "visual"} for a in anchors]
