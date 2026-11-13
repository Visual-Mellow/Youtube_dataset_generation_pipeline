#!/usr/bin/env python3
"""Smoke-test Layer 0 perception on one clip."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from core.config import load_pipeline_config
from perception.anchors import build_anchors


def main() -> int:
    level = sys.argv[1] if len(sys.argv) > 1 else "low"
    clip = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "../Dataset/raw_videos/2l8dNRfZit0/clip_000.mp4"
    )
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
        "../Dataset/extracted/features/2l8dNRfZit0/clip_000/perception"
    )
    cfg = load_pipeline_config(level)
    print("level", cfg.level, "separation", cfg.flags.separation)
    doc = build_anchors(clip, cfg, out_dir=out, force=True)
    print("COUNTS", doc["counts"])
    print("tempo", doc.get("tempo_bpm"), "asr", doc["asr"].get("available"),
          "words", doc["asr"].get("word_count"))
    print("tagger", doc["tagger"].get("available"), doc["tagger"].get("clip_topk")[:3])
    for a in doc["anchors"]:
        t = a["t_ms"]
        if a["kind"] in ("audio_onset", "speech_start", "music_boundary", "cut") and (
            4800 <= t <= 7000 or 15000 <= t <= 16500 or 20800 <= t <= 25000
        ):
            print(
                a.get("id"), t, "ms", a["kind"],
                "stem=" + str(a.get("stem")),
                "beat=" + str(a.get("on_music_beat")),
                "tags=" + str((a.get("tagger_topk") or [])[:2]),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
