#!/usr/bin/env python3
"""Download one YouTube video and split it into contiguous, non-overlapping clips."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

LOG = logging.getLogger("visualmellow.acquire")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Dataset" / "raw_videos"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}


def now() -> str:
    """Return the current UTC time in an ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON through a temporary file so interrupted writes do not corrupt metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run(command: list[str]) -> str:
    """Run an external command and include its stderr when it fails."""
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr[-3000:]}")
    return proc.stdout.strip()


def probe(path: Path) -> dict[str, Any]:
    """Read video dimensions, duration, frame rate, and audio presence with ffprobe."""
    raw = run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,duration", "-of", "json", str(path)])
    data = json.loads(raw)
    streams = data.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"No readable video stream in {path}")
    stream = video_streams[0]
    duration_s = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"Invalid video duration for {path}")
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = None
    if rate and rate != "0/0":
        a, _, b = rate.partition("/")
        fps = float(a) / float(b or 1)
    return {"duration_seconds": duration_s, "duration_ms": round(duration_s * 1000),
            "width": int(stream["width"]) if stream.get("width") else None,
            "height": int(stream["height"]) if stream.get("height") else None,
            "fps": fps, "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def identify(url: str) -> tuple[str, dict[str, Any]]:
    """Ask yt-dlp for canonical video metadata and a safe video ID."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required. Install with: python -m pip install yt-dlp") from exc
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    vid = str(info.get("id") or "").strip()
    if not vid or not re.fullmatch(r"[A-Za-z0-9_-]+", vid):
        raise ValueError("Downloader did not return a safe YouTube video ID")
    return vid, info


def _download_progress_hook():
    """Create a yt-dlp hook that prints throttled progress on complete terminal lines."""
    last_reported: dict[str, int] = {}

    def report_progress(status: dict[str, Any]) -> None:
        """Print each download's progress at ten-percent intervals and on completion."""
        filename = Path(status.get("filename") or "download").name
        downloaded = int(status.get("downloaded_bytes") or 0)
        total = int(status.get("total_bytes") or status.get("total_bytes_estimate") or 0)
        if status.get("status") == "finished":
            print(f"Downloaded {filename} ({downloaded / (1024 * 1024):.1f} MiB)", flush=True)
            return
        if status.get("status") != "downloading":
            return

        percent = (100 * downloaded / total) if total else 0
        progress_bucket = int(percent // 10) * 10 if total else downloaded // (10 * 1024 * 1024)
        if progress_bucket <= last_reported.get(filename, -1):
            return
        last_reported[filename] = progress_bucket

        size = f"{downloaded / (1024 * 1024):.1f} MiB"
        if total:
            size += f"/{total / (1024 * 1024):.1f} MiB"
        speed = status.get("speed")
        speed_text = f" at {speed / (1024 * 1024):.1f} MiB/s" if speed else ""
        eta = status.get("eta")
        eta_text = f", ETA {int(eta)}s" if eta is not None else ""
        percent_text = f"{percent:.0f}%" if total else "in progress"
        print(f"Downloading {filename}: {percent_text} ({size}){speed_text}{eta_text}", flush=True)

    return report_progress


def youtube_id_from_url(url: str) -> str | None:
    """Extract a video ID from common YouTube URL formats without a network request."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    candidate = None
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        candidate = parse_qs(parsed.query).get("v", [None])[0]
        if not candidate:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                candidate = parts[1]
    return candidate if candidate and re.fullmatch(r"[A-Za-z0-9_-]{6,20}", candidate) else None


def _download_attempts() -> list[dict[str, Any]]:
    """YouTube often returns HTTP 403 for one player client and serves another.

    Progressive formats (a single mp4 URL) fail less often than separate
    video+audio DASH streams, so those are tried first.
    """
    common = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 2,
        "windowsfilenames": True,
        "merge_output_format": "mp4",
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }
    # (player clients, format selector)
    ladder = [
        (["android_vr", "web_safari", "android"], "b[ext=mp4]/b"),
        (["tv_embedded", "mweb"], "18/22/b"),
        (["web", "android"], "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/best"),
    ]
    attempts = []
    for clients, fmt in ladder:
        opts = dict(common)
        opts["format"] = fmt
        opts["extractor_args"] = {"youtube": {"player_client": clients}}
        attempts.append(opts)
    return attempts


def download(url: str, folder: Path, force: bool, video_id: str, info: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Reuse a valid local source or download and atomically install a new MP4."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "source.mp4"
    if target.exists() and not force:
        meta = probe(target)
        LOG.debug("Reusing valid source: %s", target)
        return target, {**info, **meta, "id": video_id, "reused": True}
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required. Install with: python -m pip install yt-dlp") from exc

    last_error: Exception | None = None
    downloaded_info: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=f"{video_id}_", dir=str(folder)) as td:
        temp = Path(td)
        for attempt, opts in enumerate(_download_attempts(), start=1):
            for stale in temp.glob("source.*"):
                stale.unlink()
            opts = {
                **opts,
                "outtmpl": str(temp / "source.%(ext)s"),
                "progress_hooks": [_download_progress_hook()],
            }
            clients = opts.get("extractor_args", {}).get("youtube", {}).get("player_client")
            LOG.info("Download attempt %d for %s via %s", attempt, video_id, clients)
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    downloaded_info = ydl.extract_info(url, download=True) or {}
                    candidate = Path(ydl.prepare_filename(downloaded_info))
                    candidates = [candidate, *temp.glob("source.*")]
                    source = next((p for p in candidates if p.exists() and p.suffix.lower() in VIDEO_EXTS), None)
                    if source is None:
                        raise RuntimeError("yt-dlp completed but no video file was created")
                staged = temp / "source.mp4"
                if source.resolve() != staged.resolve():
                    run(["ffmpeg", "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac",
                         "-movflags", "+faststart", str(staged)])
                    source = staged
                probe(source)
                replace = folder / "source.mp4.new"
                shutil.copy2(source, replace)
                os.replace(replace, target)
            except Exception as exc:
                last_error = exc
                LOG.warning("Download attempt %d failed for %s: %s", attempt, video_id, exc)
                continue
            last_error = None
            break
    if last_error is not None or not target.exists():
        raise RuntimeError(f"video is not downloadable: {last_error}") from last_error
    return target, {**downloaded_info, **probe(target), "id": video_id, "reused": False}


def write_clips(source: Path, folder: Path, clip_seconds: int, duration_ms: int, force: bool = False) -> list[dict[str, Any]]:
    """Create or reuse contiguous clips and return their exact source intervals."""
    folder.mkdir(parents=True, exist_ok=True)
    clip_ms = clip_seconds * 1000
    # Avoid floating-point boundary drift; the final endpoint is the probed source endpoint.
    clips: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < duration_ms:
        end = min(start + clip_ms, duration_ms)
        clip_id = f"clip_{index:03d}"
        name = f"{clip_id}.mp4"
        path = folder / name
        actual_ms = end - start
        reusable = False
        if path.exists() and not force:
            try:
                reusable = abs(probe(path)["duration_ms"] - actual_ms) <= 250
            except Exception:
                reusable = False
        if not reusable:
            # Seeking before input keeps extraction fast; FFmpeg accurately decodes from the seek point.
            run(["ffmpeg", "-y", "-ss", f"{start / 1000:.3f}", "-i", str(source), "-t", f"{actual_ms / 1000:.3f}",
                 "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-c:a", "aac", "-movflags", "+faststart", str(path)])
        clip_info = probe(path)
        if abs(clip_info["duration_ms"] - actual_ms) > 250:
            raise RuntimeError(f"Clip duration mismatch for {clip_id}: wanted {actual_ms}ms, got {clip_info['duration_ms']}ms")
        clips.append({"clip_id": clip_id, "filename": name, "source_start_ms": start,
                      "source_end_ms": end, "duration_ms": actual_ms})
        start = end
        index += 1
    return clips


def update_ledger(path: Path, video_id: str, clips: list[dict[str, Any]]) -> None:
    """Preserve statuses for unchanged clip intervals and initialize new clips as pending."""
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8")).get("clips", {})
        except (OSError, json.JSONDecodeError):
            LOG.warning("Ignoring invalid prior ledger %s", path)
    new: dict[str, Any] = {}
    for clip in clips:
        cid = clip["clip_id"]
        prior = old.get(cid, {})
        if prior.get("source_start_ms") != clip["source_start_ms"] or prior.get("source_end_ms") != clip["source_end_ms"]:
            prior = {}
        new[cid] = {"status": prior.get("status", "pending"), "attempts": int(prior.get("attempts", 0)),
                    "started_at": prior.get("started_at"), "completed_at": prior.get("completed_at"),
                    "error": prior.get("error"), "dataset_path": prior.get("dataset_path"),
                    "source_start_ms": clip["source_start_ms"], "source_end_ms": clip["source_end_ms"]}
    atomic_json(path, {"schema_version": "1.0", "video_id": video_id, "updated_at": now(), "clips": new})


def acquire_url(url: str, output_dir: Path, clip_duration: int, force: bool = False) -> dict[str, Any]:
    """Download one URL, split it into clips, and return the acquisition record.

    Raises when the video cannot be downloaded or clipped. Callers that walk a
    CSV should catch that and keep going.
    """
    if clip_duration <= 0:
        raise ValueError("clip_duration must be a positive number of seconds")
    video_id = youtube_id_from_url(url)
    info: dict[str, Any] = {}
    if video_id:
        old_meta_path = output_dir / video_id / "video_metadata.json"
        source_path = output_dir / video_id / "source.mp4"
        if source_path.is_file() and not force and old_meta_path.exists():
            try:
                info = json.loads(old_meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}
    if not video_id:
        video_id, info = identify(url)
    folder = output_dir / video_id
    prior_metadata = {}
    if (folder / "video_metadata.json").exists() and not force:
        try:
            prior_metadata = json.loads((folder / "video_metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_metadata = {}
    source, info = download(url, folder, force, video_id, info)
    source_meta = probe(source)
    if not source_meta["has_audio"]:
        raise ValueError(f"Downloaded source has no audio stream: {source}")
    clips = write_clips(source, folder, clip_duration, source_meta["duration_ms"], force)
    metadata = {"video_id": video_id, "youtube_url": url,
                "title": info.get("title"), "source_duration_ms": source_meta["duration_ms"],
                "fps": source_meta["fps"], "width": source_meta["width"], "height": source_meta["height"],
                "downloaded_filename": source.name,
                "download_timestamp": prior_metadata.get("download_timestamp") if info.get("reused") else now(),
                "processed_at": now(), "clip_duration_ms": clip_duration * 1000, "clips": clips}
    metadata["download_timestamp"] = info.get("download_timestamp") or now()
    metadata["source_reused"] = bool(info.get("reused"))
    atomic_json(folder / "video_metadata.json", metadata)
    update_ledger(folder / "processing_status.json", video_id, clips)
    LOG.info("Acquired %s: %d contiguous clips (%ds target), directory %s", video_id, len(clips), clip_duration, folder)
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Parse acquisition options, download one source, split it, and persist its ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-duration", type=int, default=30, help="Clip length in seconds (default: 30)")
    parser.add_argument("--force", action="store_true", help="Redownload source and recreate clips")
    args = parser.parse_args(argv)
    if args.clip_duration <= 0:
        parser.error("--clip-duration must be a positive number of seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        acquire_url(args.url, args.output_dir, args.clip_duration, args.force)
        return 0
    except Exception as exc:
        LOG.error("Acquisition failed: video is not downloadable: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
