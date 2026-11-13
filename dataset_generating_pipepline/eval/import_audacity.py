"""Convert an Audacity label track back into gold JSON.

Only rows whose label does NOT start with ANCHOR| or CUE| are treated as
gold (hand-edited). Rows starting with GOLD| are also accepted.

Usage:

  python -m eval.import_audacity \\
    --labels eval/gold_templates/g-6g2uEjF1s_clip_001_high.txt \\
    --clip-id g-6g2uEjF1s/clip_001 \\
    --out eval/gold/g-6g2uEjF1s_clip_001.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_audacity(path: Path) -> list[dict]:
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            parts = line.split()
        if len(parts) < 3:
            continue
        start_s, end_s, label = float(parts[0]), float(parts[1]), "\t".join(parts[2:]).strip()
        if label.startswith("ANCHOR|") or label.startswith("CUE|"):
            # Skip machine templates unless prefixed GOLD|
            continue
        if label.startswith("GOLD|"):
            label = label[5:]
        events.append({
            "label": label,
            "onset_ms": round(start_s * 1000, 1),
            "offset_ms": round(end_s * 1000, 1),
        })
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--video-type", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    events = parse_audacity(args.labels)
    doc = {
        "clip_id": args.clip_id,
        "video_id": args.video_id,
        "video_type": args.video_type,
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(events)} gold events → {args.out}")


if __name__ == "__main__":
    main()
