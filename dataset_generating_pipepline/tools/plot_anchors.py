"""Debug plot: waveform, onsets, SED, motion, snapped cues → perception/debug.png

  python -m tools.plot_anchors \\
    --features-dir ../Dataset/extracted/features \\
    --video-id g-6g2uEjF1s --clip-id clip_001 --level high
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--level", default="high")
    ap.add_argument("--raw-videos", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = args.features_dir / args.video_id / args.clip_id
    anchors = json.loads((base / "perception" / "anchors.json").read_text(encoding="utf-8"))
    cues_path = base / "annotations" / args.level / "cue_annotations.json"
    cues = json.loads(cues_path.read_text(encoding="utf-8")) if cues_path.is_file() else {"cues": []}

    clip = Path(anchors.get("clip_path") or "")
    if not clip.is_file():
        raw_root = args.raw_videos or (args.features_dir.parent.parent / "raw_videos")
        clip = raw_root / args.video_id / f"{args.clip_id}.mp4"

    from perception.audio_io import decode_mono

    sr = 16000
    wav = decode_mono(clip, sample_rate=sr) if clip.is_file() else np.zeros(sr, dtype=np.float32)
    t = np.arange(len(wav)) / sr

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(t, wav, color="#444", lw=0.4)
    axes[0].set_ylabel("wav")
    for a in anchors.get("anchors") or []:
        if a.get("kind") not in {"audio_onset", "sed_event", "speech_start"}:
            continue
        x = float(a.get("t_ms_f", a.get("t_ms") or 0)) / 1000.0
        color = {"audio_onset": "#d62728", "sed_event": "#2ca02c", "speech_start": "#1f77b4"}.get(a["kind"], "#000")
        axes[0].axvline(x, color=color, alpha=0.5, lw=0.8)

    # Per-band proxy: strength scatter
    axes[1].set_ylabel("onset strength")
    for a in anchors.get("anchors") or []:
        if a.get("kind") != "audio_onset":
            continue
        x = float(a.get("t_ms_f", a.get("t_ms") or 0)) / 1000.0
        y = float(a.get("strength_norm") or a.get("strength") or 0)
        axes[1].vlines(x, 0, y, color="#d62728", alpha=0.7)

    axes[2].set_ylabel("SED / score")
    for a in anchors.get("anchors") or []:
        if a.get("kind") == "sed_event" or a.get("sed"):
            x = float(a.get("t_ms_f", a.get("t_ms") or 0)) / 1000.0
            y = float(a.get("prob") or (a.get("sed") or [{}])[0].get("prob") or a.get("score") or 0)
            axes[2].vlines(x, 0, y, color="#2ca02c", alpha=0.7)

    axes[3].set_ylabel("cues / motion")
    for a in anchors.get("anchors") or []:
        if a.get("kind") in {"motion", "motion_spike", "motion_stop", "cut"}:
            x = float(a.get("t_ms_f", a.get("t_ms") or 0)) / 1000.0
            axes[3].axvline(x, color="#9467bd" if "motion" in str(a.get("kind")) else "#8c564b", alpha=0.4)
    for c in cues.get("cues") or []:
        s = float(c.get("start_time_ms") or 0) / 1000.0
        e = float(c.get("end_time_ms") or s + 0.05) / 1000.0
        axes[3].axvspan(s, e, color="#17becf", alpha=0.25)

    axes[3].set_xlabel("time (s)")
    title = f"{args.video_id}/{args.clip_id} anchors_v{anchors.get('anchor_version', '?')} n={anchors.get('counts', {}).get('total')}"
    fig.suptitle(title)
    fig.tight_layout()
    out = args.out or (base / "perception" / "debug.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
