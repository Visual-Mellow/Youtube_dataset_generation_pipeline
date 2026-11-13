#!/usr/bin/env python3
"""Run the YouTube dataset pipeline from this repository.

Add links in Dataset/metadata/youtube_links.csv, or pass them directly:

    python run.py
    python run.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "dataset_generating_pipepline"
TARGET = PACKAGE / "run_youtube_links.py"

# Ensure package-local imports (core.*, utils.*, …) resolve when launched via runpy.
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    sys.argv[0] = str(TARGET)
    raise SystemExit(runpy.run_path(str(TARGET), run_name="__main__"))
