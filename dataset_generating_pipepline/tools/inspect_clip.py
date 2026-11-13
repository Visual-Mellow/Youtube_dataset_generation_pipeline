#!/usr/bin/env python3
"""Print a merged timeline for one clip and export Audacity label tracks.

Usage::

    python -m tools.inspect_clip \\
        --features-dir Dataset/extracted/features \\
        --video-id 2l8dNRfZit0 --clip-id clip_000 --level low
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def export_audacity_labels(cues: list[dict], out_path: Path) -> None:
    """Audacity label track: start_s\\tend_s\\tlabel."""
    lines = []
    for c in sorted(cues, key=lambda x: int(x.get("start_time_ms") or 0)):
        start = int(c.get("start_time_ms") or 0) / 1000.0
        end = int(c.get("end_time_ms") or start * 1000 + 100) / 1000.0
        label = f"{c.get('audio_type')}|{c.get('audio_class') or ''}"[:80]
        lines.append(f"{start:.3f}\t{end:.3f}\t{label}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote Audacity labels → {out_path} ({len(lines)} rows)")


def print_timeline(anchors_doc: dict | None, cues: list[dict], rejected: list[dict], vlog: list[dict]) -> None:
    print("\n=== ANCHORS (Layer 0) ===")
    if not anchors_doc:
        print("(no anchors.json)")
    else:
        print(f"counts={anchors_doc.get('counts')} separation={anchors_doc.get('separation_used')} "
              f"tempo={anchors_doc.get('tempo_bpm')}")
        for a in (anchors_doc.get("anchors") or [])[:40]:
            extra = ""
            if a.get("transcript"):
                extra = f' "{a["transcript"][:40]}"'
            elif a.get("tagger_topk"):
                extra = " tags=" + ",".join(n for n, _ in a["tagger_topk"][:2])
            print(f"  {a.get('id'):5s} {int(a['t_ms']):6d}ms  {a.get('kind'):14s}  "
                  f"stem={a.get('stem','?'):8s}  str={float(a.get('strength') or 0):.2f}{extra}")
        n = len(anchors_doc.get("anchors") or [])
        if n > 40:
            print(f"  … ({n - 40} more)")

    print("\n=== FINAL CUES ===")
    for c in sorted(cues, key=lambda x: int(x.get("start_time_ms") or 0)):
        prop = c.get("proposed") or {}
        delta = c.get("snap_delta_ms")
        delta_s = f" snap={delta:+d}ms" if delta is not None else ""
        prop_s = ""
        if prop and prop.get("start_time_ms") != c.get("start_time_ms"):
            prop_s = f" (proposed {prop.get('start_time_ms')}ms)"
        print(
            f"  {c.get('cue_id'):12s} {c.get('audio_type'):8s} "
            f"{int(c.get('start_time_ms') or 0):6d}-{int(c.get('end_time_ms') or 0):6d}ms "
            f"aud={c.get('audibility','?'):18s}{delta_s}{prop_s}"
        )
        print(f"               {c.get('audio_class')}")

    if rejected:
        print(f"\n=== REJECTED ({len(rejected)}) ===")
        for c in rejected[:20]:
            print(f"  {c.get('cue_id')}: {c.get('reject_reason') or c.get('audio_class')}")

    if vlog:
        print(f"\n=== VERIFIER LOG ({len(vlog)} actions) ===")
        by_action: dict[str, int] = {}
        for e in vlog:
            by_action[e.get("action", "?")] = by_action.get(e.get("action", "?"), 0) + 1
        print("  summary:", by_action)
        for e in vlog[:15]:
            print(f"  [{e.get('actor')}] {e.get('action')} {e.get('cue_id')}: {e.get('reason')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--level", choices=("low", "high"), default="low")
    parser.add_argument("--export-labels", action="store_true", default=True)
    args = parser.parse_args(argv)

    base = args.features_dir / args.video_id / args.clip_id
    ann = base / "annotations" / args.level
    anchors = _load(base / "perception" / "anchors.json")
    cue_doc = _load(ann / "cue_annotations.json") or {}
    rejected_doc = _load(ann / "rejected_cues.json") or {}
    cues = cue_doc.get("cues") or []
    rejected = rejected_doc.get("cues") or cue_doc.get("rejected_cues") or []
    vlog = cue_doc.get("verifier_log") or []
    vlog_path = ann / "verifier_log.jsonl"
    if vlog_path.is_file() and not vlog:
        vlog = [json.loads(l) for l in vlog_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"Clip {args.video_id}/{args.clip_id} level={args.level}")
    print(f"  perception: {base / 'perception'}")
    print(f"  annotations: {ann}")
    print_timeline(anchors, cues, rejected, vlog)

    if args.export_labels and cues:
        export_audacity_labels(cues, ann / f"labels_{args.level}.txt")
        if anchors:
            # Also export anchors as a second label track for manual review.
            anchor_cues = [
                {
                    "start_time_ms": a["t_ms"],
                    "end_time_ms": a.get("end_ms") or (int(a["t_ms"]) + 100),
                    "audio_type": a.get("kind"),
                    "audio_class": a.get("stem") or a.get("kind"),
                }
                for a in anchors.get("anchors") or []
                if a.get("kind") in {"audio_onset", "speech_start", "music_boundary", "cut"}
            ]
            export_audacity_labels(anchor_cues, ann / f"anchors_{args.level}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
