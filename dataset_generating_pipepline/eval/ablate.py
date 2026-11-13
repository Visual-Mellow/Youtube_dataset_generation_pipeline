"""Ablation toggles for Layer-0 v2 (M7).

Runs synthetic + optional gold eval under named configurations.

  python -m eval.ablate --synthetic-only
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ABLATIONS = {
    "onsets_v1": {"anchor_version": 1, "onsets_v2": False, "sed": False, "video_v2": False, "snap_v2": False},
    "onsets_v2": {"anchor_version": 2, "onsets_v2": True, "sed": False, "video_v2": False, "snap_v2": False},
    "onsets_v2_refine": {"anchor_version": 2, "onsets_v2": True, "refine": True, "sed": False},
    "onsets_v2_sed": {"anchor_version": 2, "onsets_v2": True, "sed": True, "video_v2": False},
    "full_v2": {"anchor_version": 2, "onsets_v2": True, "sed": True, "video_v2": True, "snap_v2": True},
}


def run_synthetic() -> dict[str, Any]:
    """Import and execute the synthetic onset suite; return summary dict."""
    import sys

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tests.test_synthetic import run_all

    return run_all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "timestamp": stamp,
        "ablations": list(ABLATIONS.keys()),
        "configs": ABLATIONS,
        "synthetic": None,
        "note": "Full gold ablations require labeled eval/gold and rebuild with toggles.",
    }
    if args.synthetic_only or True:
        try:
            report["synthetic"] = run_synthetic()
        except Exception as exc:
            report["synthetic"] = {"error": str(exc)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{stamp}_ablate.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report.get("synthetic") or {}, indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
