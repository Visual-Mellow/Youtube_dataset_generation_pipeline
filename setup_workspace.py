#!/usr/bin/env python3
"""Bootstrap a freshly cloned checkout so the pipeline can run.

Creates gitignored Dataset folders, seed metadata files, and (optionally)
installs Python dependencies. Safe to re-run — never overwrites a non-empty
youtube_links.csv or an existing .env.

Usage:
    python setup_workspace.py
    python setup_workspace.py --install
    python setup_workspace.py --install --check
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CSV_COLUMNS = [
    "youtube_url",
    "video_id",
    "status",
    "error",
    "n_clips",
    "clips_dir",
    "processed_at",
    "extract_status",
    "extract_error",
    "processing_level",
]
INDEX_COLUMNS = [
    "video_id",
    "clip_index",
    "filename",
    "video_path",
    "source_start_ms",
    "source_end_ms",
    "youtube_url",
]

DIRS = [
    ROOT / "Dataset" / "metadata",
    ROOT / "Dataset" / "raw_videos",
    ROOT / "Dataset" / "videos",
    ROOT / "Dataset" / "clips",
    ROOT / "Dataset" / "extracted" / "features",
    ROOT / ".numba_cache",
]


def _write_csv_header(path: Path, columns: list[str]) -> str:
    if path.exists():
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
    return "created"


def _write_json_if_missing(path: Path, value) -> str:
    if path.exists():
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return "created"


def ensure_dirs() -> list[str]:
    created: list[str] = []
    for path in DIRS:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path.relative_to(ROOT)))
    return created


def ensure_seed_files() -> dict[str, str]:
    meta = ROOT / "Dataset" / "metadata"
    return {
        "youtube_links.csv": _write_csv_header(meta / "youtube_links.csv", CSV_COLUMNS),
        "clip_index.csv": _write_csv_header(meta / "clip_index.csv", INDEX_COLUMNS),
        "queries.json": _write_json_if_missing(meta / "queries.json", []),
    }


def ensure_env() -> str:
    env_path = ROOT / ".env"
    example = ROOT / ".env.example"
    if env_path.exists():
        return "exists"
    if example.is_file():
        shutil.copy2(example, env_path)
        return "created_from_example"
    env_path.write_text(
        "# Copy from .env.example and set GEMINI_API_KEY or GOOGLE_API_KEY.\n"
        "GEMINI_API_KEY=\n",
        encoding="utf-8",
    )
    return "created_stub"


def check_ffmpeg() -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        try:
            out = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=False,
            )
            first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "ffmpeg ok"
            return True, first
        except OSError as exc:
            return False, str(exc)
    missing = [name for name, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not path]
    return False, f"missing on PATH: {', '.join(missing)}"


def install_requirements() -> int:
    req = ROOT / "requirements.txt"
    if not req.is_file():
        print(f"ERROR: {req} not found", file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "pip", "install", "-U", "pip", "wheel"]
    print("+", " ".join(cmd))
    code = subprocess.call(cmd)
    if code:
        return code
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Also run: python -m pip install -r requirements.txt",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print ffmpeg / .env readiness after creating folders",
    )
    args = parser.parse_args(argv)

    print(f"Workspace root: {ROOT}")
    created = ensure_dirs()
    if created:
        print("Created directories:")
        for item in created:
            print(f"  + {item}")
    else:
        print("Directories already present.")

    seeds = ensure_seed_files()
    for name, status in seeds.items():
        print(f"  {name}: {status}")

    env_status = ensure_env()
    print(f"  .env: {env_status}")

    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

    if args.install:
        print("\nInstalling Python dependencies…")
        code = install_requirements()
        if code:
            return code
        print("pip install finished.")

    if args.check or not args.install:
        ok, detail = check_ffmpeg()
        print(f"\nffmpeg: {'OK' if ok else 'MISSING'} ({detail})")
        env_path = ROOT / ".env"
        key_set = False
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() in {"GEMINI_API_KEY", "GOOGLE_API_KEY"} and value.strip():
                    key_set = True
                    break
        print(f"API key in .env: {'set' if key_set else 'not set yet — edit .env'}")

    print(
        "\nNext steps:\n"
        "  1. Edit .env and set GEMINI_API_KEY (or GOOGLE_API_KEY)\n"
        "  2. Install deps if you have not:  python setup_workspace.py --install\n"
        "     (or: python -m pip install -r requirements.txt)\n"
        "  3. Add a URL to Dataset/metadata/youtube_links.csv, or run:\n"
        '       python run.py --url "https://www.youtube.com/watch?v=VIDEO_ID"\n'
        "  4. Optional sanity check:  python run.py --self-test\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
