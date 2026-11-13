"""Video cues v2: frame-diff, Farneback motion, cuts, intensity changes — via PTS.

Video anchors are supporting evidence only; they never move audio-derived times.
Legacy ``motion`` kind is emitted as an alias of ``motion_spike`` for compat.

``intensity_change`` anchors follow the notebook logic in
``data_analysis/video_audio_analysis.ipynb``: per-frame mean pixel intensity,
then contiguous runs where ``|Δ intensity| >= min_intensity_change`` (default 5).
These are preferred over peaks/troughs for LLM visual grounding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("visualmellow.perception.video_cues")


def _parabolic_peak(y_prev: float, y: float, y_next: float) -> float:
    """Sub-sample peak offset in [-0.5, 0.5] frames."""
    denom = (y_prev - 2 * y + y_next)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y_prev - y_next) / denom


def detect_intensity_changes(
    frames_gray: list[np.ndarray],
    frame_pts_s: np.ndarray,
    *,
    min_change: float = 5.0,
) -> list[dict[str, Any]]:
    """Detect frame changes from mean pixel-intensity jumps (notebook logic).

    For each grayscale frame compute mean intensity in [0, 255]. Contiguous
    runs where ``|Δ mean| >= min_change`` become one ``intensity_change``
    anchor at the run's mean time (not peak/trough extrema).
    """
    if len(frames_gray) < 2 or len(frame_pts_s) < 2:
        return []
    n = min(len(frames_gray), len(frame_pts_s))
    means = np.asarray(
        [float(np.mean(frames_gray[i].astype(np.float64))) for i in range(n)],
        dtype=np.float64,
    )
    t = np.asarray(frame_pts_s[:n], dtype=np.float64)
    dy = np.diff(means)
    abs_dy = np.abs(dy)
    # Fixed floor (notebook/user-validated): |Δ mean intensity| ≥ min_change (default 5).
    thr = float(min_change)

    anchors: list[dict[str, Any]] = []
    i = 0
    while i < len(dy):
        if abs_dy[i] < thr:
            i += 1
            continue
        j = i
        while j + 1 < len(dy) and abs_dy[j + 1] >= thr:
            j += 1
        start_f, end_f = int(i), int(j + 1)
        start_s, end_s = float(t[start_f]), float(t[end_f])
        mean_s = 0.5 * (start_s + end_s)
        delta = float(means[end_f] - means[start_f])
        t_ms_f = mean_s * 1000.0
        anchors.append({
            "kind": "intensity_change",
            "stem": "visual",
            "t_ms": int(round(t_ms_f)),
            "t_ms_f": round(t_ms_f, 1),
            "t_end_ms_f": round(end_s * 1000.0, 1),
            "strength": abs(delta),
            "strength_norm": min(1.0, abs(delta) / 40.0),
            "delta": round(delta, 2),
            "abs_delta": round(abs(delta), 2),
            "direction": "up" if delta >= 0 else "down",
            "intensity_before": round(float(means[start_f]), 2),
            "intensity_after": round(float(means[end_f]), 2),
            "start_frame": start_f,
            "end_frame": end_f,
            "timing_source": "visual",
            "min_change": thr,
        })
        i = j + 1

    LOG.info(
        "Intensity changes: %d events (|Δ|≥%.1f) from %d frames",
        len(anchors), thr, n,
    )
    return anchors


def extract_video_cues_v2(
    clip_path: Path | str,
    *,
    frame_pts_s: np.ndarray | None = None,
    use_farneback: bool = True,
    use_transnet: bool = False,
    width: int = 160,
    min_intensity_change: float = 5.0,
) -> dict[str, Any]:
    """Return motion/cut/intensity-change anchors + motion energy for avsync."""
    import cv2

    clip_path = Path(clip_path)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        LOG.warning("Cannot open video %s", clip_path)
        return {"anchors": [], "motion_times_ms": [], "motion_energy": [], "available": False}

    frames_gray: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (width, max(1, int(frame.shape[0] * width / max(frame.shape[1], 1)))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)
    cap.release()

    n = len(frames_gray)
    if n < 3:
        return {"anchors": [], "motion_times_ms": [], "motion_energy": [], "available": False}

    if frame_pts_s is None or len(frame_pts_s) < n:
        # Fallback constant grid — should be rare if timebase ran.
        fps = 30.0
        try:
            from perception.timebase import probe

            fps = float(probe(clip_path).get("fps") or 30.0)
        except Exception:
            pass
        frame_pts_s = np.arange(n) / fps
    else:
        frame_pts_s = np.asarray(frame_pts_s[:n], dtype=np.float64)

    # Frame MAD difference.
    diffs = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diffs[i] = float(np.mean(np.abs(frames_gray[i].astype(np.float32) - frames_gray[i - 1].astype(np.float32)))) / 255.0

    # Optical flow energy (Farneback).
    flow_e = np.zeros(n, dtype=np.float64)
    if use_farneback:
        try:
            prev = frames_gray[0]
            for i in range(1, n):
                flow = cv2.calcOpticalFlowFarneback(
                    prev, frames_gray[i], None,
                    0.5, 3, 15, 3, 5, 1.2, 0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_e[i] = float(np.mean(mag))
                prev = frames_gray[i]
        except Exception as exc:
            LOG.warning("Farneback failed: %s", exc)
            flow_e = diffs.copy()
    else:
        flow_e = diffs.copy()

    energy = 0.5 * (diffs / (diffs.max() + 1e-9) + flow_e / (flow_e.max() + 1e-9))
    d_energy = np.diff(energy, prepend=energy[0])

    anchors: list[dict[str, Any]] = []

    # Cuts from large MAD spikes (cross-check with scenes later).
    cut_thr = float(np.mean(diffs) + 2.5 * (np.std(diffs) + 1e-9))
    for i in range(1, n - 1):
        if diffs[i] >= cut_thr and diffs[i] >= diffs[i - 1] and diffs[i] >= diffs[i + 1]:
            off = _parabolic_peak(diffs[i - 1], diffs[i], diffs[i + 1])
            t_s = float(frame_pts_s[i]) + off * (frame_pts_s[min(i + 1, n - 1)] - frame_pts_s[i])
            anchors.append({
                "kind": "cut",
                "stem": "visual",
                "t_ms": int(round(t_s * 1000)),
                "t_ms_f": round(t_s * 1000, 1),
                "strength": float(diffs[i]),
                "timing_source": "visual",
                "cut_confirmed_by": ["frame_diff"],
            })

    # Motion spikes / stops.
    spike_thr = float(np.mean(d_energy) + 2.0 * (np.std(d_energy) + 1e-9))
    stop_thr = float(np.mean(d_energy) - 2.0 * (np.std(d_energy) + 1e-9))
    for i in range(1, n - 1):
        t_base = float(frame_pts_s[i])
        if d_energy[i] >= spike_thr and d_energy[i] >= d_energy[i - 1] and d_energy[i] >= d_energy[i + 1]:
            off = _parabolic_peak(d_energy[i - 1], d_energy[i], d_energy[i + 1])
            t_s = t_base + off * (frame_pts_s[min(i + 1, n - 1)] - frame_pts_s[i])
            item = {
                "kind": "motion_spike",
                "stem": "visual",
                "t_ms": int(round(t_s * 1000)),
                "t_ms_f": round(t_s * 1000, 1),
                "strength": float(d_energy[i]),
                "timing_source": "visual",
            }
            anchors.append(item)
            # Compat alias.
            anchors.append({**item, "kind": "motion"})
        if d_energy[i] <= stop_thr and d_energy[i] <= d_energy[i - 1] and d_energy[i] <= d_energy[i + 1]:
            # sudden stop: large negative derivative after high velocity
            if energy[i - 1] > float(np.median(energy) + np.std(energy)):
                off = _parabolic_peak(-d_energy[i - 1], -d_energy[i], -d_energy[i + 1])
                t_s = t_base + off * (frame_pts_s[min(i + 1, n - 1)] - frame_pts_s[i])
                anchors.append({
                    "kind": "motion_stop",
                    "stem": "visual",
                    "t_ms": int(round(t_s * 1000)),
                    "t_ms_f": round(t_s * 1000, 1),
                    "strength": float(-d_energy[i]),
                    "timing_source": "visual",
                })

    # Flash: brief luminance spike.
    means = np.array([f.mean() for f in frames_gray], dtype=np.float64)
    d_mean = np.diff(means, prepend=means[0])
    flash_thr = float(np.mean(np.abs(d_mean)) + 3.0 * (np.std(d_mean) + 1e-9))
    for i in range(1, n - 1):
        if abs(d_mean[i]) >= flash_thr and abs(d_mean[i]) >= abs(d_mean[i - 1]) and abs(d_mean[i]) >= abs(d_mean[i + 1]):
            t_s = float(frame_pts_s[i])
            anchors.append({
                "kind": "flash",
                "stem": "visual",
                "t_ms": int(round(t_s * 1000)),
                "t_ms_f": round(t_s * 1000, 1),
                "strength": float(abs(d_mean[i])),
                "timing_source": "visual",
            })

    # Optional TransNetV2 cross-check.
    if use_transnet:
        try:
            _apply_transnet(clip_path, anchors, frame_pts_s)
        except Exception as exc:
            LOG.info("TransNetV2 skipped: %s", exc)

    # Mean pixel-intensity changes (notebook): preferred visual change signal for LLM.
    intensity_anchors = detect_intensity_changes(
        frames_gray, frame_pts_s, min_change=float(min_intensity_change),
    )
    anchors.extend(intensity_anchors)

    # Cross-check with existing ffmpeg scene_score cuts.
    try:
        from perception.scenes import extract_visual_anchors

        legacy = extract_visual_anchors(clip_path)
        cut_ms = {int(a.ms) for a in legacy if a.kind == "cut"}
        for a in anchors:
            if a["kind"] != "cut":
                continue
            for cms in cut_ms:
                if abs(cms - int(a["t_ms"])) <= 100:
                    a.setdefault("cut_confirmed_by", []).append("ffmpeg_scene")
                    break
    except Exception:
        pass

    times_ms = [float(frame_pts_s[i] * 1000) for i in range(n)]
    LOG.info(
        "Video cues v2: %d anchors from %d frames (intensity_change=%d)",
        len(anchors), n, sum(1 for a in anchors if a.get("kind") == "intensity_change"),
    )
    return {
        "available": True,
        "anchors": anchors,
        "motion_times_ms": times_ms,
        "motion_energy": energy.tolist(),
        "n_frames": n,
        "n_intensity_changes": sum(1 for a in anchors if a.get("kind") == "intensity_change"),
        "min_intensity_change": float(min_intensity_change),
    }


def _apply_transnet(clip_path: Path, anchors: list[dict], frame_pts_s: np.ndarray) -> None:
    """Mark cuts confirmed by TransNetV2 when the package is installed."""
    # Soft optional dependency.
    import importlib

    mod = importlib.import_module("transnetv2_pytorch")
    # API varies; best-effort.
    model = getattr(mod, "TransNetV2", None)
    if model is None:
        return
    LOG.info("TransNetV2 available — confirmation pass not fully wired for all APIs; marking flag only")
    for a in anchors:
        if a.get("kind") == "cut":
            a.setdefault("cut_confirmed_by", []).append("transnetv2_available")
