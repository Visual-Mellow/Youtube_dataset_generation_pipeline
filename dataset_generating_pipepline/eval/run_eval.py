"""Evaluate Layer-0 anchors against gold labels.

  python -m eval.run_eval \\
    --features-dir ../Dataset/extracted/features \\
    --gold-dir eval/gold \\
    --level high --anchor-version 1
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.metrics import aggregate, summarize_clip


AUDIO_KINDS = {"audio_onset", "speech_start", "sed_event", "music_boundary"}


def _load_gold(gold_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not gold_dir.is_dir():
        return out
    for path in sorted(gold_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        cid = str(doc.get("clip_id") or path.stem)
        out[cid] = doc
        # Also index bare clip stem variants.
        if "/" in cid:
            out[cid.split("/")[-1]] = doc
            out[cid.replace("/", "_")] = doc
    return out


def _pred_from_anchors(anchors_doc: dict[str, Any], *, audio_only: bool = True) -> list[dict[str, Any]]:
    preds = []
    for a in anchors_doc.get("anchors") or []:
        kind = a.get("kind")
        if audio_only and kind not in AUDIO_KINDS:
            continue
        preds.append({
            "label": kind,
            "onset_ms": a.get("t_ms_f", a.get("t_ms")),
            "offset_ms": a.get("t_end_ms_f", a.get("end_ms")),
            "kind": kind,
            **{k: a.get(k) for k in ("id", "stem", "score") if k in a},
        })
    return preds


def _resolve_clip_keys(gold_key: str, features_dir: Path) -> tuple[str, str] | None:
    """Map gold clip_id → (video_id, clip_id)."""
    if "/" in gold_key:
        vid, cid = gold_key.split("/", 1)
        if (features_dir / vid / cid).is_dir():
            return vid, cid
    if "_" in gold_key:
        # video_clip_000 style
        parts = gold_key.rsplit("_", 1)
        if len(parts) == 2 and (features_dir / parts[0] / parts[1]).is_dir():
            return parts[0], parts[1]
        # video_id_clip_001
        for vid in features_dir.iterdir() if features_dir.is_dir() else []:
            if not vid.is_dir():
                continue
            for cand in (gold_key, gold_key.replace(vid.name + "_", "")):
                if (vid / cand).is_dir():
                    return vid.name, cand
                if gold_key.startswith(vid.name + "_"):
                    rest = gold_key[len(vid.name) + 1 :]
                    if (vid / rest).is_dir():
                        return vid.name, rest
    # Search features for matching clip folder name.
    if features_dir.is_dir():
        for vid in features_dir.iterdir():
            if (vid / gold_key).is_dir():
                return vid.name, gold_key
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--gold-dir", type=Path, default=Path(__file__).resolve().parent / "gold")
    ap.add_argument("--level", default="high", choices=("low", "high"))
    ap.add_argument("--anchor-version", type=int, default=1, choices=(1, 2))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()

    gold = _load_gold(args.gold_dir)
    if not gold:
        print(f"No gold labels found in {args.gold_dir}")
        print("Create templates with: python -m eval.make_gold_template ...")
        print("Import hand labels with: python -m eval.import_audacity ...")
        # Still write an empty baseline so CI/scripts have a reproducible artifact.
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        empty = {
            "timestamp": stamp,
            "level": args.level,
            "anchor_version": args.anchor_version,
            "n_gold_files": 0,
            "message": "no gold",
            "aggregate": aggregate([]),
        }
        out = args.out_dir / f"{stamp}_v{args.anchor_version}_{args.level}.json"
        out.write_text(json.dumps(empty, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote empty baseline → {out}")
        return

    clip_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, gdoc in gold.items():
        # Prefer canonical clip_id keys only once.
        cid = str(gdoc.get("clip_id") or key)
        if cid in seen:
            continue
        seen.add(cid)
        resolved = _resolve_clip_keys(cid, args.features_dir)
        if resolved is None:
            print(f"SKIP (features missing): {cid}")
            continue
        vid, clip = resolved
        anchors_path = args.features_dir / vid / clip / "perception" / "anchors.json"
        if not anchors_path.is_file():
            print(f"SKIP (no anchors): {vid}/{clip}")
            continue
        anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
        # Soft check: schema / version metadata for reporting.
        schema = anchors.get("schema_version")
        pred = _pred_from_anchors(anchors)
        summary = summarize_clip(pred, gdoc.get("events") or [])
        summary["clip_id"] = f"{vid}/{clip}"
        summary["schema_version"] = schema
        summary["n_pred"] = len(pred)
        summary["n_gold"] = len(gdoc.get("events") or [])
        clip_rows.append(summary)
        onset = summary["onset"]
        print(
            f"{vid}/{clip}: matched={onset['n_matched']}/{onset['n_gold']} "
            f"med={onset['median_err_ms']} p50%={onset['pct_within_50']:.1f} "
            f"F1@50={summary['pr50']['f1']:.3f}"
        )

    agg = aggregate(clip_rows)
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "timestamp": stamp,
        "level": args.level,
        "anchor_version": args.anchor_version,
        "features_dir": str(args.features_dir),
        "gold_dir": str(args.gold_dir),
        "aggregate": agg,
        "clips": clip_rows,
    }
    out = args.out_dir / f"{stamp}_v{args.anchor_version}_{args.level}.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
