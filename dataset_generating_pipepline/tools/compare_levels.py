#!/usr/bin/env python3
"""Compare LOW vs HIGH annotations for one or more clips.

Usage::

    python -m tools.compare_levels \\
        --features-dir Dataset/extracted/features \\
        --video-id 2l8dNRfZit0 --clip-id clip_000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(level: str, ann_dir: Path) -> dict:
    cue_doc = _load(ann_dir / "cue_annotations.json") or {}
    cues = cue_doc.get("cues") or []
    rejected = cue_doc.get("rejected_cues") or []
    vlog = cue_doc.get("verifier_log") or []
    types = Counter(str(c.get("audio_type") or "?") for c in cues)
    snaps = [abs(int(c["snap_delta_ms"])) for c in cues if c.get("snap_delta_ms") is not None]
    actions = Counter(e.get("action") for e in vlog)
    llm_calls = cue_doc.get("llm_call_count")
    if llm_calls is None:
        llm_path = ann_dir / "llm_calls.jsonl"
        if llm_path.is_file():
            llm_calls = sum(1 for l in llm_path.read_text(encoding="utf-8").splitlines() if l.strip())
    return {
        "level": level,
        "n_cues": len(cues),
        "n_rejected": len(rejected),
        "types": dict(types),
        "mean_abs_snap_ms": round(sum(snaps) / len(snaps), 1) if snaps else None,
        "max_abs_snap_ms": max(snaps) if snaps else None,
        "n_snapped": len(snaps),
        "verifier_actions": dict(actions),
        "llm_calls": llm_calls or 0,
        "cues": cues,
    }


def compare_clip(features_dir: Path, video_id: str, clip_id: str) -> None:
    base = features_dir / video_id / clip_id
    print(f"\n######## {video_id}/{clip_id} ########")
    summaries = {}
    for level in ("low", "high"):
        ann = base / "annotations" / level
        if not ann.is_dir():
            print(f"  [{level}] missing ({ann})")
            continue
        s = summarise(level, ann)
        summaries[level] = s
        print(
            f"  [{level}] cues={s['n_cues']} rejected={s['n_rejected']} "
            f"types={s['types']} llm_calls={s['llm_calls']} "
            f"snap_mean={s['mean_abs_snap_ms']} snap_max={s['max_abs_snap_ms']}"
        )
        print(f"         verifier={s['verifier_actions']}")

    if "low" in summaries and "high" in summaries:
        low_starts = {(c.get("audio_type"), int(c.get("start_time_ms") or 0) // 250)
                      for c in summaries["low"]["cues"]}
        high_starts = {(c.get("audio_type"), int(c.get("start_time_ms") or 0) // 250)
                       for c in summaries["high"]["cues"]}
        only_high = high_starts - low_starts
        only_low = low_starts - high_starts
        print(f"  bucket-diff (±250ms): only_high≈{len(only_high)} only_low≈{len(only_low)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--clip-id", action="append", default=[])
    args = parser.parse_args(argv)

    if not args.video_id:
        # Auto-discover video dirs that have annotations.
        for vdir in sorted(args.features_dir.iterdir() if args.features_dir.is_dir() else []):
            if not vdir.is_dir():
                continue
            for cdir in sorted(vdir.iterdir()):
                if (cdir / "annotations").is_dir():
                    compare_clip(args.features_dir, vdir.name, cdir.name)
        return 0

    for vid in args.video_id:
        clips = args.clip_id or [
            p.name for p in sorted((args.features_dir / vid).iterdir())
            if p.is_dir() and (p / "annotations").is_dir()
        ]
        for cid in clips:
            compare_clip(args.features_dir, vid, cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
