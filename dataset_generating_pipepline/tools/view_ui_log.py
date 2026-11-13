#!/usr/bin/env python3
"""Render a VisualMellow ``pipeline.ui.log`` as collapsible HTML and open in a browser."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from utils.ui_log import open_log_in_browser, render_html  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="Path to pipeline.ui.log (or batch .ui.log)")
    parser.add_argument("--out", type=Path, default=None, help="HTML output path (default: alongside the .log)")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in your default browser",
    )
    args = parser.parse_args(argv)
    if not args.log_file.is_file():
        parser.error(f"log file not found: {args.log_file}")
    html_path = render_html(args.log_file, out_path=args.out)
    print(html_path)
    if args.open:
        open_log_in_browser(args.log_file, html_path=html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
