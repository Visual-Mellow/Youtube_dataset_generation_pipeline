"""Clip timebase probing, A/V start delta, and per-frame PTS maps.

Never compute ``frame_index / fps`` for perception timestamps — always map
through the PTS array when available.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("visualmellow.perception.timebase")


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate in {"0/0", "N/A"}:
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            n, d = float(num), float(den)
            return n / d if d else None
        except Exception:
            return None
    try:
        return float(rate)
    except Exception:
        return None


def probe(clip: Path | str) -> dict[str, Any]:
    """Return timebase metadata via ffprobe JSON."""
    clip = Path(clip)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", str(clip),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {clip}: {proc.stderr[-500:]}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None) or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None) or {}

    audio_start = float(audio.get("start_time") or 0.0)
    video_start = float(video.get("start_time") or 0.0)
    r_fps = _parse_rate(video.get("r_frame_rate"))
    avg_fps = _parse_rate(video.get("avg_frame_rate"))
    is_vfr = False
    if r_fps and avg_fps and r_fps > 0:
        is_vfr = abs(r_fps - avg_fps) / r_fps > 0.02

    sr = int(float(audio.get("sample_rate") or 0) or 0)
    duration = float(fmt.get("duration") or video.get("duration") or audio.get("duration") or 0.0)
    av_start_delta_ms = (audio_start - video_start) * 1000.0
    if abs(av_start_delta_ms) > 20:
        LOG.warning(
            "A/V start_time delta %.1f ms for %s (audio=%.4f video=%.4f)",
            av_start_delta_ms, clip.name, audio_start, video_start,
        )

    return {
        "audio_start": audio_start,
        "video_start": video_start,
        "av_start_delta_ms": av_start_delta_ms,
        "fps": avg_fps or r_fps,
        "r_frame_rate": r_fps,
        "avg_frame_rate": avg_fps,
        "is_vfr": is_vfr,
        "sr": sr,
        "duration": duration,
        "duration_ms": int(round(duration * 1000)) if duration else 0,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
    }


def extract_frame_pts(
    clip: Path | str,
    *,
    out_path: Path | None = None,
    max_frames: int = 20000,
) -> np.ndarray:
    """Return float64 array of video frame PTS in seconds (stream timebase).

    Prefer ``ffprobe -show_frames`` packet PTS. Falls back to constant fps grid.
    """
    clip = Path(clip)
    meta = probe(clip)
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pts_time,pkt_pts_time",
        "-of", "csv=p=0",
        str(clip),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    times: list[float] = []
    if proc.returncode == 0 and proc.stdout:
        for line in proc.stdout.splitlines():
            line = line.strip().strip(",")
            if not line or line == "N/A":
                continue
            # csv may be "pts_time" or "pts_time,pkt_pts_time"
            part = line.split(",")[0]
            try:
                times.append(float(part))
            except Exception:
                continue
            if len(times) >= max_frames:
                break

    if len(times) < 2:
        fps = meta.get("fps") or 30.0
        n = int(round((meta.get("duration") or 1.0) * fps))
        times = [i / fps for i in range(max(n, 1))]
        LOG.warning("PTS extract sparse for %s; using constant fps grid (%.3f)", clip.name, fps)

    # Normalize so first frame is 0 (shared t=0 after A/V alignment).
    arr = np.asarray(times, dtype=np.float64)
    arr = arr - arr[0]
    # Shift by -video_start relative? probe already uses stream start; keep relative to first frame.
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)
    return arr


def frame_index_to_ms(frame_pts_s: np.ndarray, index: int) -> float:
    """Map frame index → milliseconds via PTS (not fps)."""
    if frame_pts_s is None or len(frame_pts_s) == 0:
        return 0.0
    i = int(np.clip(index, 0, len(frame_pts_s) - 1))
    return float(frame_pts_s[i] * 1000.0)


def build_timebase_doc(
    clip: Path | str,
    *,
    perception_dir: Path | None = None,
    extract_pts: bool = True,
) -> dict[str, Any]:
    """Probe + optional PTS dump; return dict for anchors.json ``timebase``."""
    clip = Path(clip)
    meta = probe(clip)
    pts_path = None
    n_frames = None
    if extract_pts and perception_dir is not None:
        pts_file = Path(perception_dir) / "frame_pts_s.npy"
        pts = extract_frame_pts(clip, out_path=pts_file)
        pts_path = str(pts_file)
        n_frames = int(len(pts))
    elif extract_pts:
        pts = extract_frame_pts(clip)
        n_frames = int(len(pts))
    else:
        pts = None

    return {
        **meta,
        "frame_pts_path": pts_path,
        "n_frames": n_frames,
        "t0_policy": "first_video_frame_pts_zero",
    }
