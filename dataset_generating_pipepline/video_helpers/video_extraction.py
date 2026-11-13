import json
from pathlib import Path
from typing import Any

import build_dataset
from core.constants import *


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


def extract_video(
    video_id: str,
    raw_dir: Path,
    extract_dir: Path,
    queries_path: Path,
    force: bool,
    *,
    processing_level: str = "low",
) -> int:
    """Run VideoMAE + signal-first cue extraction for one downloaded source, then export queries."""
    argv = [
        "--input-dir", str(raw_dir),
        "--output-dir", str(extract_dir),
        "--video-id", video_id,
        "--device", "auto",
        "--processing-level", processing_level,
    ]
    if force:
        argv.append("--force")
    code = build_dataset.main(argv)
    if code != 0:
        raise RuntimeError(extract_failure_text(raw_dir, video_id))
    return export_grounding_queries(extract_dir / "dataset.jsonl", queries_path, video_id)

