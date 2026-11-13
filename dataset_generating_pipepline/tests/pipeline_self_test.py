import tempfile
from pathlib import Path
from typing import Any

from core.constants import *
from utils.loaders import save_rows, load_rows
from orchestrator import process_links


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

