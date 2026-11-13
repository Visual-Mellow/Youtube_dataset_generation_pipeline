"""Dump anchors + cues into an Audacity label track for hand-labeling.

Usage (from dataset_generating_pipepline/):

  python -m eval.make_gold_template \\
    --features-dir ../Dataset/extracted/features \\
    --video-id g-6g2uEjF1s --clip-id clip_001 --level high
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_labels(
    anchors_doc: dict,
    cue_doc: dict | None,
) -> list[tuple[float, float, str]]:
    """Return Audacity rows as (start_s, end_s, label)."""
    rows: list[tuple[float, float, str]] = []
    for a in anchors_doc.get("anchors") or []:
        t0 = float(a.get("t_ms_f") if a.get("t_ms_f") is not None else a.get("t_ms") or 0) / 1000.0
        t1_raw = a.get("t_end_ms_f")
        if t1_raw is None:
            t1_raw = a.get("end_ms")
        t1 = float(t1_raw) / 1000.0 if t1_raw is not None else t0 + 0.05
        if t1 <= t0:
            t1 = t0 + 0.05
        kind = a.get("kind", "anchor")
        stem = a.get("stem") or ""
        label = f"ANCHOR|{kind}|{stem}|{a.get('id', '')}"
        rows.append((t0, t1, label))

    if cue_doc:
        for c in cue_doc.get("cues") or []:
            t0 = float(c.get("start_time_ms") or 0) / 1000.0
            t1 = float(c.get("end_time_ms") or (c.get("start_time_ms") or 0) + (c.get("duration_ms") or 50)) / 1000.0
            if t1 <= t0:
                t1 = t0 + 0.05
            cls = (c.get("audio_class") or c.get("audio_prompt") or "cue").replace("\t", " ")[:80]
            label = f"CUE|{c.get('audio_type', '')}|{cls}"
            rows.append((t0, t1, label))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def write_audacity(path: Path, rows: list[tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for start, end, label in rows:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--level", default="high", choices=("low", "high"))
    ap.add_argument("--out", type=Path, default=None, help="Output .txt (default: eval/gold_templates/...)")
    args = ap.parse_args()

    base = args.features_dir / args.video_id / args.clip_id
    anchors_path = base / "perception" / "anchors.json"
    cues_path = base / "annotations" / args.level / "cue_annotations.json"
    if not anchors_path.is_file():
        raise SystemExit(f"Missing anchors: {anchors_path}")

    anchors = _load(anchors_path)
    cues = _load(cues_path) if cues_path.is_file() else None
    rows = build_labels(anchors, cues)

    out = args.out
    if out is None:
        root = Path(__file__).resolve().parent
        out = root / "gold_templates" / f"{args.video_id}_{args.clip_id}_{args.level}.txt"
    write_audacity(out, rows)
    print(f"Wrote {len(rows)} labels → {out}")


if __name__ == "__main__":
    main()
