#!/usr/bin/env python3
"""Acquire every unprocessed YouTube URL listed in Dataset/metadata/youtube_links.csv.

A row whose status is processed is not downloaded again. After the clips exist,
each clip is run through feature and cue extraction unless extract_status is
already extracted. If YouTube refuses the download, the row is marked
download_error, the reason is logged, and the next URL is tried.

Clips and extracted records are stored as:

    Dataset/raw_videos/<youtube_id>/source.mp4
    Dataset/raw_videos/<youtube_id>/clip_000.mp4
    Dataset/videos/<youtube_id>_clip0.mp4
    Dataset/extracted/features/<youtube_id>/clip_000/
    Dataset/metadata/queries.json   (Gemini cues, merged in for these clips)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

PIPELINE_DIR = Path(__file__).resolve().parent
RL_ROOT = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import download_and_clip as acquire  # noqa: E402
import build_dataset  # noqa: E402

LOG = logging.getLogger("visualmellow.youtube_links")

COLUMNS = [
    "youtube_url",
    "video_id",
    "status",
    "error",
    "n_clips",
    "clips_dir",
    "processed_at",
    "extract_status",
    "extract_error",
]
PROCESSED = {"processed", "done", "complete", "completed"}
EXTRACTED = {"extracted", "done", "complete", "completed"}
DEFAULT_CSV = RL_ROOT / "Dataset" / "metadata" / "youtube_links.csv"
DEFAULT_RAW = RL_ROOT / "Dataset" / "raw_videos"
DEFAULT_VIDEOS = RL_ROOT / "Dataset" / "videos"
DEFAULT_EXTRACTED = RL_ROOT / "Dataset" / "extracted"
DEFAULT_QUERIES = RL_ROOT / "Dataset" / "metadata" / "queries.json"
DEFAULT_INDEX = RL_ROOT / "Dataset" / "metadata" / "clip_index.csv"
INDEX_COLUMNS = [
    "video_id",
    "clip_index",
    "filename",
    "video_path",
    "source_start_ms",
    "source_end_ms",
    "youtube_url",
]


def download_error_text(exc: BaseException) -> str:
    """One CSV-sized reason. The batch log uses this same sentence."""
    raw = str(exc).strip()
    line = raw.splitlines()[-1] if raw else exc.__class__.__name__
    while True:
        if line.lower().startswith("video is not downloadable:"):
            line = line.split(":", 1)[1].strip()
            continue
        if line.startswith("ERROR:"):
            line = line.removeprefix("ERROR:").strip()
            continue
        break
    if len(line) > 300:
        line = line[:300]
    return f"video is not downloadable: {line}"


def is_processed(status: str) -> bool:
    return status.strip().lower() in PROCESSED


def is_extracted(status: str) -> bool:
    return status.strip().lower() in EXTRACTED


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read the link sheet. Missing file is created with a header and no rows."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_rows(path, [], COLUMNS)
        LOG.info("Created %s. Add a youtube_url on each row, then run this again.", path)
        return [], list(COLUMNS)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [{k: (v or "") for k, v in row.items() if k} for row in reader]
    for name in COLUMNS:
        if name not in fieldnames:
            fieldnames.append(name)
    return rows, fieldnames


def save_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Replace the sheet atomically so a killed run cannot leave a half-written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def publish_clips(video_id: str, folder: Path, clips: list[dict[str, Any]], videos_dir: Path) -> list[dict[str, Any]]:
    """Hard-link each raw clip to Dataset/videos/<youtube_id>_clip<index>.mp4."""
    videos_dir.mkdir(parents=True, exist_ok=True)
    published = []
    for clip in clips:
        index = int(str(clip["clip_id"]).split("_")[-1])
        src = folder / clip["filename"]
        filename = f"{video_id}_clip{index}.mp4"
        dest = videos_dir / filename
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        published.append({
            "video_id": video_id,
            "clip_index": str(index),
            "filename": filename,
            "video_path": f"Dataset/videos/{filename}",
            "source_start_ms": str(clip.get("source_start_ms", "")),
            "source_end_ms": str(clip.get("source_end_ms", "")),
        })
    return published


def upsert_clip_index(path: Path, video_id: str, youtube_url: str, published: list[dict[str, Any]]) -> None:
    """Replace this video's rows in clip_index.csv and leave every other video alone."""
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = [{k: (v or "") for k, v in row.items() if k} for row in csv.DictReader(f)]
    kept = [row for row in existing if row.get("video_id") != video_id]
    for item in published:
        kept.append({**item, "youtube_url": youtube_url})
    save_rows(path, kept, INDEX_COLUMNS)


def _modality(event_type: str) -> str:
    if event_type.upper() in {"SFX", "AMBIENCE", "MUSIC", "NARRATOR", "MOVIE_BGM"}:
        return "audio"
    return "visual"


def export_grounding_queries(dataset_jsonl: Path, queries_path: Path, youtube_id: str) -> int:
    """Merge this video's Gemini cues into queries.json under <id>_clip<index>."""
    if not dataset_jsonl.is_file():
        return 0
    fresh: list[dict[str, Any]] = []
    touched: set[str] = set()
    for line in dataset_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("video_id")) != youtube_id:
            continue
        clip_index = int(str(row["clip_id"]).split("_")[-1])
        grounding_id = f"{youtube_id}_clip{clip_index}"
        touched.add(grounding_id)
        duration_sec = float(row["clip_duration_ms"]) / 1000.0
        story = ""
        artifact = (row.get("gemini_explanation") or {}).get("artifact")
        if artifact:
            explanation_path = dataset_jsonl.parent / artifact
            if explanation_path.is_file():
                story = str(json.loads(explanation_path.read_text(encoding="utf-8")).get("response", {}).get("video_content") or "")
        source_url = str((row.get("source_video") or {}).get("youtube_url") or "")
        for cue in row.get("audio_cues") or []:
            event_type = str(cue.get("audio_type") or "unknown").upper()
            fresh.append({
                "video_id": grounding_id,
                "video_path": f"Dataset/videos/{grounding_id}.mp4",
                "event_text": str(cue.get("audio_class") or cue.get("audio_prompt") or "").strip(),
                "event_type": event_type,
                "modality": _modality(event_type),
                "start_sec": int(cue["start_time_ms"]) / 1000.0,
                "duration_sec": int(cue["duration_ms"]) / 1000.0,
                "video_duration_sec": round(duration_sec, 3),
                "story": story,
                "source_url": source_url,
                "label_source": "gemini_pseudo",
            })
    existing: list[dict[str, Any]] = []
    if queries_path.is_file():
        existing = json.loads(queries_path.read_text(encoding="utf-8"))
    kept = [
        record for record in existing
        if not (record.get("video_id") in touched and record.get("label_source") == "gemini_pseudo")
    ]
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text(json.dumps(kept + fresh, indent=2) + "\n", encoding="utf-8")
    return len(fresh)


def extract_failure_text(raw_dir: Path, video_id: str) -> str:
    """Read the per-clip ledger so the CSV records why extraction stopped."""
    status_path = raw_dir / video_id / "processing_status.json"
    if not status_path.is_file():
        return "clip extraction failed"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "clip extraction failed"
    messages = []
    for clip_id, clip in (status.get("clips") or {}).items():
        if clip.get("status") == "failed" and clip.get("error"):
            messages.append(f"{clip_id}: {clip['error']}")
    text = "; ".join(messages) if messages else "clip extraction failed"
    return text[:500]


def extract_video(video_id: str, raw_dir: Path, extract_dir: Path, queries_path: Path, force: bool) -> int:
    """Run VideoMAE + Gemini cue extraction for one downloaded source, then export queries."""
    argv = [
        "--input-dir", str(raw_dir),
        "--output-dir", str(extract_dir),
        "--video-id", video_id,
        "--device", "auto",
    ]
    if force:
        argv.append("--force")
    code = build_dataset.main(argv)
    if code != 0:
        raise RuntimeError(extract_failure_text(raw_dir, video_id))
    return export_grounding_queries(extract_dir / "dataset.jsonl", queries_path, video_id)


def process_links(
    csv_path: Path,
    *,
    raw_dir: Path,
    videos_dir: Path,
    clip_index_path: Path,
    extract_dir: Path,
    queries_path: Path,
    clip_duration: int,
    force: bool,
    acquire_fn: Callable[..., dict[str, Any]] = acquire.acquire_url,
    extract_fn: Callable[..., int] = extract_video,
) -> dict[str, int]:
    """Download missing videos, then extract features and cues for clips that are not extracted yet."""
    rows, fieldnames = load_rows(csv_path)
    counts = {"skipped": 0, "acquired": 0, "download_error": 0, "empty": 0, "extracted": 0, "extract_error": 0}
    if not rows:
        return counts
    for row in rows:
        url = row.get("youtube_url", "").strip()
        if not url:
            counts["empty"] += 1
            continue
        downloaded = is_processed(row.get("status", "")) and not force
        if downloaded and is_extracted(row.get("extract_status", "")) and not force:
            LOG.info("Skipping video that is already downloaded and extracted %s", url)
            counts["skipped"] += 1
            continue
        if not downloaded:
            LOG.info("Acquiring %s", url)
            try:
                metadata = acquire_fn(url, raw_dir, clip_duration, force)
            except Exception as exc:
                message = download_error_text(exc)
                row["status"] = "download_error"
                row["error"] = message
                row["video_id"] = row.get("video_id") or acquire.youtube_id_from_url(url) or ""
                row["processed_at"] = acquire.now()
                save_rows(csv_path, rows, fieldnames)
                counts["download_error"] += 1
                LOG.error("%s | %s", message, url)
                continue
            video_id = str(metadata["video_id"])
            clips = list(metadata.get("clips") or [])
            folder = raw_dir / video_id
            published = publish_clips(video_id, folder, clips, videos_dir)
            upsert_clip_index(clip_index_path, video_id, url, published)
            row["video_id"] = video_id
            row["status"] = "processed"
            row["error"] = ""
            row["n_clips"] = str(len(published))
            row["clips_dir"] = f"Dataset/raw_videos/{video_id}"
            row["processed_at"] = acquire.now()
            save_rows(csv_path, rows, fieldnames)
            counts["acquired"] += 1
            LOG.info("Downloaded %s: %d clips in %s and Dataset/videos/", video_id, len(published), row["clips_dir"])
        video_id = row.get("video_id") or acquire.youtube_id_from_url(url) or ""
        if not video_id:
            row["extract_status"] = "extract_error"
            row["extract_error"] = "clip extraction failed: missing video id"
            save_rows(csv_path, rows, fieldnames)
            counts["extract_error"] += 1
            LOG.error("clip extraction failed: missing video id | %s", url)
            continue
        LOG.info("Extracting features and cues for %s", video_id)
        try:
            n_queries = extract_fn(video_id, raw_dir, extract_dir, queries_path, force)
        except Exception as exc:
            message = str(exc).splitlines()[-1][:500]
            row["extract_status"] = "extract_error"
            row["extract_error"] = message
            save_rows(csv_path, rows, fieldnames)
            counts["extract_error"] += 1
            LOG.error("clip extraction failed for %s: %s", video_id, message)
            continue
        row["video_id"] = video_id
        row["extract_status"] = "extracted"
        row["extract_error"] = ""
        save_rows(csv_path, rows, fieldnames)
        counts["extracted"] += 1
        LOG.info("Extracted %s: %d grounding queries written", video_id, n_queries)
    return counts


def self_test() -> int:
    """Check skip, error recording, and clip naming without touching YouTube."""
    with tempfile.TemporaryDirectory(prefix="yt_links_test_") as td:
        root = Path(td)
        csv_path = root / "youtube_links.csv"
        raw_dir = root / "raw_videos"
        videos_dir = root / "videos"
        index_path = root / "clip_index.csv"
        rows = [
            {"youtube_url": "https://www.youtube.com/watch?v=alreadyok1", "status": "processed", "extract_status": "extracted", "video_id": "alreadyok1"},
            {"youtube_url": "https://www.youtube.com/watch?v=needextract", "status": "processed", "video_id": "needextract"},
            {"youtube_url": "https://www.youtube.com/watch?v=o-Ikkh5oxuo", "status": ""},
            {"youtube_url": "https://youtu.be/newvideo01", "status": "pending"},
        ]
        save_rows(csv_path, rows, COLUMNS)
        extracted_ids: list[str] = []

        def fake_acquire(url: str, output_dir: Path, clip_duration: int, force: bool) -> dict[str, Any]:
            if "o-Ikkh5oxuo" in url:
                raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
            if "alreadyok1" in url or "needextract" in url:
                raise AssertionError("a downloaded row was downloaded again")
            video_id = "newvideo01"
            folder = output_dir / video_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "clip_000.mp4").write_bytes(b"clip-bytes")
            (folder / "clip_001.mp4").write_bytes(b"clip-bytes-2")
            return {
                "video_id": video_id,
                "clips": [
                    {"clip_id": "clip_000", "filename": "clip_000.mp4", "source_start_ms": 0, "source_end_ms": 30000},
                    {"clip_id": "clip_001", "filename": "clip_001.mp4", "source_start_ms": 30000, "source_end_ms": 45000},
                ],
            }

        def fake_extract(video_id: str, raw_dir: Path, extract_dir: Path, queries_path: Path, force: bool) -> int:
            extracted_ids.append(video_id)
            if video_id == "needextract":
                raise RuntimeError("Gemini explanation failed")
            return 3

        counts = process_links(
            csv_path,
            raw_dir=raw_dir,
            videos_dir=videos_dir,
            clip_index_path=index_path,
            extract_dir=root / "extracted",
            queries_path=root / "queries.json",
            clip_duration=30,
            force=False,
            acquire_fn=fake_acquire,
            extract_fn=fake_extract,
        )
        loaded, _ = load_rows(csv_path)
        by_url = {row["youtube_url"]: row for row in loaded}
        ok = by_url["https://www.youtube.com/watch?v=alreadyok1"]
        pending_extract = by_url["https://www.youtube.com/watch?v=needextract"]
        blocked = by_url["https://www.youtube.com/watch?v=o-Ikkh5oxuo"]
        fresh = by_url["https://youtu.be/newvideo01"]
        problems = []
        expected_counts = {"skipped": 1, "acquired": 1, "download_error": 1, "empty": 0, "extracted": 1, "extract_error": 1}
        if counts != expected_counts:
            problems.append(f"counts {counts}")
        if ok["status"] != "processed" or ok["extract_status"] != "extracted":
            problems.append("fully processed row was rewritten")
        if "alreadyok1" in extracted_ids:
            problems.append("extracted row was extracted again")
        if pending_extract["extract_status"] != "extract_error" or "Gemini" not in pending_extract["extract_error"]:
            problems.append(f"pending extract row {pending_extract}")
        if blocked["status"] != "download_error":
            problems.append(f"blocked status is {blocked['status']}")
        if "HTTP Error 403" not in blocked["error"] or not blocked["error"].startswith("video is not downloadable:"):
            problems.append(f"blocked error is {blocked['error']}")
        if blocked["video_id"] != "o-Ikkh5oxuo":
            problems.append(f"blocked video_id is {blocked['video_id']}")
        if blocked.get("extract_status"):
            problems.append("download error was sent to extraction")
        if fresh["status"] != "processed" or fresh["extract_status"] != "extracted" or fresh["n_clips"] != "2" or fresh["video_id"] != "newvideo01":
            problems.append(f"fresh row {fresh}")
        if extracted_ids != ["needextract", "newvideo01"]:
            problems.append(f"extract order {extracted_ids}")
        expected = [
            videos_dir / "newvideo01_clip0.mp4",
            videos_dir / "newvideo01_clip1.mp4",
        ]
        for path in expected:
            if not path.is_file() or path.read_bytes()[:4] != b"clip":
                problems.append(f"missing published clip {path.name}")
        if problems:
            for item in problems:
                LOG.error("self-test failed: %s", item)
            return 1
    LOG.info("self-test passed: downloaded clips are extracted, 403 stays a download_error, clips named <id>_clip<index>.mp4")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="YouTube URL to add to the links CSV, then process with every other unfinished row",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--clip-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--clip-duration", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract rows already marked done")
    parser.add_argument("--self-test", action="store_true", help="Run the skip/error/naming checks and exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        return self_test()
    if args.clip_duration <= 0:
        parser.error("--clip-duration must be a positive number of seconds")
    if args.url:
        rows, fieldnames = load_rows(args.csv)
        known = {row.get("youtube_url", "").strip() for row in rows}
        added = 0
        for url in args.url:
            url = url.strip()
            if not url or url in known:
                continue
            rows.append({"youtube_url": url})
            known.add(url)
            added += 1
        if added:
            save_rows(args.csv, rows, fieldnames)
            LOG.info("Added %d new link(s) to %s", added, args.csv)
    LOG.info("Reading YouTube links from %s", args.csv)
    counts = process_links(
        args.csv,
        raw_dir=args.raw_dir,
        videos_dir=args.videos_dir,
        clip_index_path=args.clip_index,
        extract_dir=args.extract_dir,
        queries_path=args.queries,
        clip_duration=args.clip_duration,
        force=args.force,
    )
    LOG.info(
        "Finished youtube_links: %d downloaded, %d extracted, %d skipped, %d download errors, %d extract errors",
        counts["acquired"], counts["extracted"], counts["skipped"], counts["download_error"], counts["extract_error"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
