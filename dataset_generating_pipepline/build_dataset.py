#!/usr/bin/env python3
"""Build resumable VisualMellow clip records with VideoMAE features and Gemini pseudo-labels."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("visualmellow.dataset")
PIPELINE_VERSION = "1.0.0"
VIDEO_TYPES = {"ACTION", "DIALOGUE", "ESTABLISHING", "CHASE", "FIGHT", "REVEAL", "EMOTIONAL", "HORROR_TENSION", "TRAVEL", "OTHER"}
AUDIO_TYPES = {"SFX", "AMBIENCE", "MUSIC", "MOVIE_BGM", "NARRATOR"}
EVENT_TYPES = {"INSTANT", "EVENT", "CONTINUOUS", "SEGMENT", "GLOBAL"}
ALIGNMENT_RELATIONS = {"SYNCHRONOUS", "ANTICIPATORY", "REACTIVE", "CONTEXTUAL", "TRANSITIONAL"}
VIDEO_MODEL_DEFAULT = "MCG-NJU/videomae-base"
GEMINI_MODEL_DEFAULT = "gemini-3-flash-preview"
TOKEN_WINDOW_S = 4.0
TOKEN_FRAMES = 16
LOGGER_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_VIDEOMAE_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}

DATASET_EXPLANATION_SUFFIX = r"""

Dataset addition: classify the clip with exactly one video_type from
ACTION, DIALOGUE, ESTABLISHING, CHASE, FIGHT, REVEAL, EMOTIONAL,
HORROR_TENSION, TRAVEL, OTHER. Include video_type in the same JSON object.
This is a Gemini-generated pseudo-label, not a human-verified class.
"""

CUE_ANNOTATION_PROMPT = r"""You are a cinematic sound designer annotating one short source video clip for VisualMellow temporal-alignment research.

The attached video is the ORIGINAL audiovisual clip, including its audio stream. Use both what is visible and what is audible as context. The target is a cinematic soundscape aligned to visible/narrative events, NOT a transcription of every sound in the source soundtrack. Existing audio can help disambiguate events, but do not blindly copy it or let it replace visual grounding. Do not claim a sound is audible unless supported by the source audio; distinguish observed audio from a proposed generated cinematic cue in the fields below.

Clip duration: {clip_duration_ms} milliseconds. All times must be clip-relative.
Video/scene classification and explanation:
{explanation_json}

For each useful cue return every field:
- cue_id: short unique ID within this clip, e.g. cue_001
- audio_type: SFX, AMBIENCE, MUSIC, MOVIE_BGM, or NARRATOR. Prefer no narrator for visual-only content unless voice-over is clearly useful.
- audio_class: production-compatible semantic class/description; for MOVIE_BGM use a catalog clip ID only if one is actually supplied, otherwise use a semantic music description.
- audio_prompt: this repository has no separate generation prompt field, so for SFX, AMBIENCE, MUSIC, and NARRATOR it MUST equal audio_class. For MOVIE_BGM preserve its clip ID in audio_class and use that same ID as audio_prompt unless a catalog candidate is actually supplied.
- cue_trigger: concrete visible/narrative event or condition that motivates this cue.
- alignment_relation: SYNCHRONOUS, ANTICIPATORY, REACTIVE, CONTEXTUAL, or TRANSITIONAL.
- event_type: INSTANT, EVENT, CONTINUOUS, SEGMENT, or GLOBAL.
- start_time_ms, duration_ms, end_time_ms: integer clip-relative timing, with end_time_ms = start_time_ms + duration_ms and end_time_ms <= clip duration.
- visual_evidence: concise visible evidence. Do not invent off-screen action.
- source_audio_context: what the original audio contributes, or "not clearly audible". Keep separate from the generated cue description.

Ontology guidance: INSTANT is a near-point event; EVENT is a short non-zero event; CONTINUOUS is a sustained source; SEGMENT is tied to a narrative segment; GLOBAL spans essentially the full clip. SYNCHRONOUS hits the event; ANTICIPATORY begins before it; REACTIVE follows it; CONTEXTUAL establishes/maintains the scene; TRANSITIONAL accompanies a narrative transition. Keep cues sparse and avoid duplicates. Return [] when no cue is justified.

Return JSON only, in this shape:
{{"cues":[{{"cue_id":"cue_001","audio_type":"SFX","audio_class":"...","audio_prompt":"...","cue_trigger":"...","alignment_relation":"SYNCHRONOUS","event_type":"EVENT","start_time_ms":1000,"duration_ms":800,"end_time_ms":1800,"visual_evidence":"...","source_audio_context":"..."}}]}}
"""


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: Any) -> None:
    """Atomically serialize JSON so interrupted writes do not corrupt metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(torch: Any, data: Any, path: Path) -> None:
    """Save a tensor artifact through a temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def run(command: list[str]) -> str:
    """Run an external command and raise a concise error if it fails."""
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr[-2500:]}")
    return proc.stdout.strip()


def ffprobe(path: Path) -> dict[str, Any]:
    """Read the media streams needed to validate a clip and its audio."""
    raw = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    data = json.loads(raw)
    videos = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not videos:
        raise ValueError("No video stream found")
    v = videos[0]
    seconds = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Video duration is missing or invalid")
    fps_text = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    a, _, b = fps_text.partition("/")
    fps = float(a) / float(b or 1)
    return {"duration_ms": round(seconds * 1000), "fps": fps, "width": int(v["width"]),
            "height": int(v["height"]), "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
            "video_codec": v.get("codec_name"), "pixel_format": v.get("pix_fmt"),
            "format_name": data.get("format", {}).get("format_name")}


def validate_video(path: Path, expected_duration_ms: int | None = None) -> dict[str, Any]:
    """Validate clip duration, audio presence, and first-frame decodability."""
    meta = ffprobe(path)
    if expected_duration_ms and abs(meta["duration_ms"] - expected_duration_ms) > 300:
        raise ValueError(f"Clip duration differs from metadata ({meta['duration_ms']} vs {expected_duration_ms} ms)")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install it with: python -m pip install -r requirements.txt") from exc
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV could not open the clip")
        ok, _ = cap.read()
        if not ok:
            raise ValueError("OpenCV could not decode the first frame")
    finally:
        cap.release()
    return meta


def _model_explanation_prompt() -> tuple[str, str, str]:
    """Build the existing explanation prompt with the dataset video-type taxonomy."""
    from Utils.prompts import gemini_explain_visual_story_prompt
    raw_prompt = gemini_explain_visual_story_prompt.template
    prompt = gemini_explain_visual_story_prompt.format(director_prompt="(none provided)", video_duration_ms="{duration}")
    prompt += DATASET_EXPLANATION_SUFFIX
    return prompt, "gemini_explain_visual_story_prompt+dataset_video_type_v1", raw_prompt


def _build_cue_prompt(duration_ms: int, explanation: dict[str, Any]) -> str:
    """Combine the production cue prompt with dataset-specific annotation rules."""
    from Utils.prompts import gemini_decide_audio_cues_prompt_without_narrator_video
    production_prompt = gemini_decide_audio_cues_prompt_without_narrator_video.format(
        visual_story=json.dumps(explanation, ensure_ascii=False),
        bgm_candidates="[]",
        video_duration_ms=duration_ms,
    )
    return (
        production_prompt
        + "\n\nDATASET ANNOTATION CONTRACT (extends the production cue instructions; this JSON schema takes precedence):\n"
        + CUE_ANNOTATION_PROMPT.format(
            clip_duration_ms=duration_ms,
            explanation_json=json.dumps(explanation, ensure_ascii=False),
        )
    )


def _hash_prompt(prompt: str) -> str:
    """Return a stable SHA-256 digest for a prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    """Hash a media file in chunks without loading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_local_env() -> None:
    """Load API keys from this repo's .env. Parent VisualMellow env files are optional fallbacks."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    btp = ROOT.parent
    for path in (
        ROOT / ".env",
        btp / ".env",
        btp / "visualMellow" / "backend" / ".env",
        btp / "visualMellow" / ".env",
    ):
        if path.is_file():
            load_dotenv(path, override=False)


def _gemini_json(model: str, prompt: str, video: bytes) -> Any:
    """Send the original audiovisual clip through VisualMellow's Gemini wrapper."""
    _load_local_env()
    from Utils.llm import query_gemini_multimodal
    result = query_gemini_multimodal(model, prompt, video_bytes=video, mime_type="video/mp4")
    if result is None:
        raise RuntimeError("Gemini multimodal request returned no parsed JSON")
    return result


def _load_json(path: Path) -> Any:
    """Load a UTF-8 JSON artifact from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_clip_table(video_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one video's clip metadata and persistent processing ledger."""
    meta_path = video_dir / "video_metadata.json"
    status_path = video_dir / "processing_status.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = _load_json(meta_path)
    if not status_path.is_file():
        raise FileNotFoundError(f"Missing {status_path}; run download_and_clip.py to initialize the ledger")
    status = _load_json(status_path)
    return meta, status


def _save_status(path: Path, status: dict[str, Any]) -> None:
    """Refresh the ledger timestamp and save it atomically."""
    status["updated_at"] = utc_now()
    atomic_json(path, status)


def _configure_logging(level: str) -> None:
    """Set concise pipeline logs and silence routine HTTP/client chatter."""
    logging.basicConfig(level=getattr(logging, level), format=LOGGER_FORMAT)
    for logger_name in ("httpx", "httpcore", "google_genai", "google.genai", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _video_id(video_dir: Path, metadata: dict[str, Any]) -> str:
    """Resolve a video ID from metadata, falling back to its directory name."""
    return str(metadata.get("video_id") or video_dir.name)


def _clip_paths(output_dir: Path, video_id: str, clip_id: str) -> dict[str, Path]:
    """Return the canonical output paths for one clip's artifacts."""
    base = output_dir / "features" / video_id / clip_id
    return {"base": base, "tokens": base / "video_tokens.pt", "times": base / "token_timestamps.json",
            "token_meta": base / "video_token_metadata.json", "explanation": base / "gemini_explanation.json",
            "cues": base / "cue_annotations.json", "record": base / "record.json"}


def _load_cached_tokens(paths: dict[str, Path], torch: Any, clip_sha256: str) -> tuple[Any, list[dict[str, int]], dict[str, Any]] | None:
    """Load cached VideoMAE tensors only when shape and source fingerprint match."""
    if not all(paths[k].is_file() for k in ("tokens", "times", "token_meta")):
        return None
    try:
        tokens = torch.load(paths["tokens"], map_location="cpu", weights_only=True)
        times = _load_json(paths["times"])
        meta = _load_json(paths["token_meta"])
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] != len(times):
            return None
        if list(tokens.shape) != meta.get("tensor_shape") or meta.get("clip_sha256") != clip_sha256:
            return None
        return tokens, times, meta
    except Exception:
        return None


def extract_videomae(path: Path, meta: dict[str, Any], model_name: str, device: str) -> tuple[Any, list[dict[str, int]], dict[str, Any]]:
    """Sample temporal windows and return pooled VideoMAE tokens with timestamps."""
    import cv2
    import numpy as np
    import torch
    from transformers import VideoMAEImageProcessorPil as VideoMAEImageProcessor, VideoMAEModel

    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device
    cache_key = (model_name, resolved_device)
    cached = _VIDEOMAE_CACHE.get(cache_key)
    if cached is None:
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        model = VideoMAEModel.from_pretrained(model_name).eval().to(resolved_device)
        _VIDEOMAE_CACHE[cache_key] = (torch, processor, model)
    else:
        _, processor, model = cached
    cfg = model.config
    tubelet = int(getattr(cfg, "tubelet_size", 2))
    frames_n = TOKEN_FRAMES
    temporal_n = frames_n // tubelet
    patch_size = getattr(cfg, "patch_size", 16)
    image_size = getattr(cfg, "image_size", 224)
    patch_size = int(patch_size if isinstance(patch_size, int) else patch_size[0])
    image_size = int(image_size if isinstance(image_size, int) else image_size[0])
    patches_side = image_size // patch_size
    spatial_n = patches_side * patches_side
    hidden_dim = int(getattr(cfg, "hidden_size", 0))
    if not hidden_dim:
        hidden_dim = int(getattr(model, "num_features", 0))

    duration_s = meta["duration_ms"] / 1000.0
    window = TOKEN_WINDOW_S
    starts = [float(s) for s in np.arange(0.0, max(duration_s, 0.001), window)]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not reopen clip for token extraction")
    tensors = []
    timestamps: list[dict[str, int]] = []
    try:
        for win_start in starts:
            win_end = min(duration_s, win_start + window)
            if win_end <= win_start:
                continue
            # Stay before the final frame interval. Seeking at duration-1ms often
            # requests a timestamp for which no decoded frame exists (especially
            # in short clips whose audio stream is slightly longer than video).
            frame_period = 1.0 / max(float(meta.get("fps") or 25.0), 1.0)
            sample_end = max(win_start, win_end - frame_period)
            sample_times = np.linspace(win_start, sample_end, frames_n)
            frames = []
            for t in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"VideoMAE frame decode failed at {t:.3f}s")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(resolved_device) for k, v in inputs.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"Unexpected VideoMAE output layout {tuple(hidden.shape)}")
            tokens_n = hidden.shape[1]
            if tokens_n != temporal_n * spatial_n:
                # Derive the spatial count from real model output while requiring temporal tubelet agreement.
                if tokens_n % temporal_n:
                    raise RuntimeError(f"Cannot map {tokens_n} hidden tokens to {temporal_n} temporal tubelets")
                actual_spatial = tokens_n // temporal_n
            else:
                actual_spatial = spatial_n
            pooled = hidden[0].reshape(temporal_n, actual_spatial, hidden.shape[-1]).mean(dim=1).float().cpu()
            tensors.append(pooled)
            # A pooled token covers its actual group of sampled input frames. Use group boundaries
            # midway between neighboring sample frames, clipped to this window.
            for ti in range(temporal_n):
                lo_i, hi_i = ti * tubelet, min((ti + 1) * tubelet, frames_n) - 1
                left = win_start if ti == 0 else (float(sample_times[lo_i - 1]) + float(sample_times[lo_i])) / 2
                right = win_end if ti == temporal_n - 1 else (float(sample_times[hi_i]) + float(sample_times[hi_i + 1])) / 2
                timestamps.append({"token_index": len(timestamps), "start_ms": max(0, round(left * 1000)),
                                   "end_ms": min(meta["duration_ms"], round(right * 1000))})
    finally:
        cap.release()
    if not tensors:
        raise RuntimeError("VideoMAE produced no temporal tokens")
    result = torch.cat(tensors, dim=0).contiguous()
    if result.shape[1] != 768 and model_name == VIDEO_MODEL_DEFAULT:
        LOG.warning("Default VideoMAE hidden size is %s (expected 768 for videomae-base)", result.shape[1])
    token_meta = {"model_name": model_name, "hidden_dimension": int(result.shape[1]),
                  "number_of_tokens": int(result.shape[0]), "tensor_shape": list(result.shape),
                  "sampling": {"window_seconds": window, "frames_per_window": frames_n,
                               "tubelet_size": tubelet, "temporal_tokens_per_window": temporal_n,
                               "spatial_patches_mean_pooled": actual_spatial, "window_starts_seconds": starts,
                               "mapping": "sampled-frame tubelet spans; approximate; clip-relative"},
                  "fps": meta["fps"], "clip_duration_ms": meta["duration_ms"], "device": resolved_device,
                  "transformers_model_type": getattr(cfg, "model_type", None),
                  "transformers_config": cfg.to_dict() if hasattr(cfg, "to_dict") else str(cfg),
                  "created_at": utc_now()}
    return result, timestamps, token_meta


def _cached_explanation(path: Path, prompt_hash: str, model_name: str, clip_sha256: str) -> dict[str, Any] | None:
    """Reuse an explanation only when model, prompt, clip, and schema still match."""
    try:
        data = _load_json(path)
        response = data["response"]
        required = ("video_content", "scene_gravity", "scene_interpretation", "audio_asthetics")
        if (all(isinstance(response.get(k), str) for k in required)
                and response.get("video_type") in VIDEO_TYPES
                and data.get("prompt_sha256") == prompt_hash
                and data.get("model") == model_name
                and data.get("clip_sha256") == clip_sha256
                and data.get("gemini_received_audio") is True):
            return data
    except Exception:
        pass
    return None


def _cached_cues(path: Path, prompt_hash: str, duration_ms: int, model_name: str, clip_sha256: str) -> dict[str, Any] | None:
    """Reuse annotations only when provenance, timing, prompt, and clip still match."""
    try:
        data = _load_json(path)
        if (data.get("prompt_sha256") != prompt_hash
                or data.get("annotation_type") != "pseudo_ground_truth"
                or data.get("model") != model_name
                or data.get("clip_sha256") != clip_sha256
                or data.get("gemini_received_audio") is not True):
            return None
        validate_cues(data.get("cues"), duration_ms)
        return data
    except Exception:
        return None


def cache_is_current(paths: dict[str, Path], *, clip_path: Path, duration_ms: int,
                     video_model: str, gemini_model: str) -> bool:
    """Check whether all expensive artifacts match the current input and settings."""
    try:
        record = _load_json(paths["record"])
        if record.get("schema_version") != "1.0.0" or not isinstance(record.get("source_video"), dict):
            return False
        token_meta = _load_json(paths["token_meta"])
        sampling = token_meta.get("sampling", {})
        clip_sha256 = _hash_file(clip_path)
        if (token_meta.get("model_name") != video_model
                or token_meta.get("clip_sha256") != clip_sha256
                or sampling.get("window_seconds") != TOKEN_WINDOW_S
                or sampling.get("frames_per_window") != TOKEN_FRAMES):
            return False
        explanation_prompt, _, _ = _model_explanation_prompt()
        explanation_prompt = explanation_prompt.replace("{duration}", str(duration_ms))
        explanation = _cached_explanation(paths["explanation"], _hash_prompt(explanation_prompt), gemini_model, clip_sha256)
        if explanation is None:
            return False
        cue_prompt = _build_cue_prompt(duration_ms, explanation["response"])
        return _cached_cues(paths["cues"], _hash_prompt(cue_prompt), duration_ms, gemini_model, clip_sha256) is not None
    except Exception:
        return False


def normalize_cues(response: Any, duration_ms: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize Gemini cue fields, clamp times, and attach pseudo-label provenance."""
    if isinstance(response, dict):
        raw_cues = response.get("cues") or response.get("audio_cues") or response.get("results")
    elif isinstance(response, list):
        raw_cues = response
    else:
        raise ValueError("Gemini cue result must be an object or list")
    if not isinstance(raw_cues, list):
        raise ValueError("Gemini cue response is missing a cues list")
    cues = []
    normalizations = []
    for i, raw in enumerate(raw_cues, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Cue {i} is not an object")
        typ = str(raw.get("audio_type", "")).upper()
        if typ not in AUDIO_TYPES:
            raise ValueError(f"Cue {i} has invalid audio_type {typ!r}")
        start_raw = raw.get("start_time_ms")
        duration_raw = raw.get("duration_ms")
        end_raw = raw.get("end_time_ms")
        if start_raw is None or duration_raw is None:
            raise ValueError(f"Cue {i} is missing start_time_ms or duration_ms")
        start = int(round(float(start_raw)))
        dur = int(round(float(duration_raw)))
        start0, dur0 = start, dur
        start = min(max(0, start), max(duration_ms - 1, 0))
        dur = min(max(1, dur), duration_ms - start)
        end = start + dur
        changed = start != start0 or dur != dur0 or (end_raw is not None and int(round(float(end_raw))) != end)
        if changed:
            normalizations.append({"cue_index": i - 1, "from": {"start_time_ms": start0, "duration_ms": dur0, "end_time_ms": end_raw},
                                   "to": {"start_time_ms": start, "duration_ms": dur, "end_time_ms": end},
                                   "reason": "clamped_to_clip_or_end_recomputed"})
        audio_class = str(raw.get("audio_class") or "").strip()
        audio_prompt = audio_class if str(raw.get("audio_type", "")).upper() != "MOVIE_BGM" else str(raw.get("audio_prompt") or audio_class).strip()
        fields = {"audio_type": typ, "audio_class": audio_class, "audio_prompt": audio_prompt,
                  "cue_trigger": str(raw.get("cue_trigger") or "").strip(),
                  "alignment_relation": str(raw.get("alignment_relation") or "").upper(),
                  "event_type": str(raw.get("event_type") or "").upper(),
                  "visual_evidence": str(raw.get("visual_evidence") or "").strip(),
                  "source_audio_context": str(raw.get("source_audio_context") or "not clearly audible").strip()}
        if not fields["audio_class"] or not fields["audio_prompt"] or not fields["cue_trigger"]:
            raise ValueError(f"Cue {i} must have audio_class, audio_prompt, and cue_trigger")
        if fields["alignment_relation"] not in ALIGNMENT_RELATIONS:
            raise ValueError(f"Cue {i} has invalid alignment_relation {fields['alignment_relation']!r}")
        if fields["event_type"] not in EVENT_TYPES:
            raise ValueError(f"Cue {i} has invalid event_type {fields['event_type']!r}")
        cue_id = str(raw.get("cue_id") or f"cue_{i:03d}")
        if any(c["cue_id"] == cue_id for c in cues):
            cue_id = f"{cue_id}_{i:03d}"
        cues.append({"cue_id": cue_id, **fields, "start_time_ms": start, "duration_ms": dur, "end_time_ms": end,
                     "temporal_annotation": {"start_time_ms": start, "duration_ms": dur, "end_time_ms": end,
                                             "source": "gemini", "annotation_type": "pseudo_ground_truth"},
                     "source": "gemini", "annotation_type": "pseudo_ground_truth"})
    validate_cues(cues, duration_ms)
    return cues, normalizations


def validate_cues(cues: Any, duration_ms: int) -> None:
    """Reject cues with invalid fields, provenance, or clip-relative time ranges."""
    if not isinstance(cues, list):
        raise ValueError("cues must be a list")
    required = ("cue_id", "audio_type", "audio_class", "audio_prompt", "cue_trigger", "alignment_relation", "event_type",
                "start_time_ms", "duration_ms", "end_time_ms", "source", "annotation_type")
    for i, cue in enumerate(cues):
        if not isinstance(cue, dict) or any(key not in cue for key in required):
            raise ValueError(f"Cue {i} lacks required dataset fields")
        start, duration, end = int(cue["start_time_ms"]), int(cue["duration_ms"]), int(cue["end_time_ms"])
        if start < 0 or duration <= 0 or end != start + duration or end > duration_ms:
            raise ValueError(f"Cue {i} has invalid time interval {start}+{duration}={end}, clip={duration_ms}")
        if cue["source"] != "gemini" or cue["annotation_type"] != "pseudo_ground_truth":
            raise ValueError(f"Cue {i} provenance must be Gemini pseudo-ground-truth")
        ann = cue.get("temporal_annotation", {})
        if (ann.get("source"), ann.get("annotation_type")) != ("gemini", "pseudo_ground_truth"):
            raise ValueError(f"Cue {i} temporal annotation provenance is invalid")


def _artifact_provenance(model: str, prompt_name: str, prompt_hash: str, clip: Path,
                         clip_sha256: str, gemini_received_audio: bool) -> dict[str, Any]:
    """Create shared provenance fields for generated explanation and cue files."""
    return {"model": model, "prompt_name": prompt_name, "prompt_sha256": prompt_hash,
            "generation_timestamp": utc_now(), "gemini_received_audio": gemini_received_audio,
            "source_clip": str(clip), "clip_sha256": clip_sha256}


def process_clip(video_dir: Path, clip_entry: dict[str, Any], output_dir: Path, status_path: Path, status_doc: dict[str, Any],
                 *, video_model: str, gemini_model: str, device: str, max_retries: int, force: bool) -> str:
    """Resume or process one clip while tracking status and caching each stage."""
    video_id = str(status_doc.get("video_id") or video_dir.name)
    clip_id = str(clip_entry["clip_id"])
    clip_path = video_dir / str(clip_entry["filename"])
    expected_ms = int(clip_entry["duration_ms"])
    source_video_metadata = _load_json(video_dir / "video_metadata.json")
    paths = _clip_paths(output_dir, video_id, clip_id)
    paths["base"].mkdir(parents=True, exist_ok=True)
    ledger = status_doc.setdefault("clips", {}).setdefault(clip_id, {"status": "pending", "attempts": 0})
    if ledger.get("status") == "processing":
        # The global dataset lock proves no older builder is still alive.
        LOG.warning("Recovering interrupted processing state for %s/%s", video_id, clip_id)
        ledger.update(status="pending", error="Recovered stale processing status after previous run stopped")
        _save_status(status_path, status_doc)
    if ledger.get("status") == "completed" and not force:
        try:
            validate_completed(paths, output_dir, video_id, clip_id)
            completed_video_meta = validate_video(clip_path, expected_ms)
            if not completed_video_meta["has_audio"]:
                raise ValueError("Completed clip no longer has its original audio stream")
            # Prompts and cue bounds use the encoded clip's probed duration. Use
            # that same value for cache validation, since stream durations can
            # differ slightly from the requested source interval at frame edges.
            if cache_is_current(paths, clip_path=clip_path, duration_ms=completed_video_meta["duration_ms"],
                                video_model=video_model, gemini_model=gemini_model):
                return "skipped"
            LOG.debug("[%s/%s] cached artifacts are stale; refreshing only invalid stages", video_id, clip_id)
            ledger.update(status="pending", error="Cached artifacts do not match current model/prompt versions")
        except Exception as exc:
            LOG.warning("Completed ledger has invalid artifacts for %s/%s; retrying: %s", video_id, clip_id, exc)
            ledger.update(status="pending", error=f"Invalid completed artifacts: {exc}")
    ledger["status"] = "processing"
    ledger["attempts"] = int(ledger.get("attempts", 0)) + 1
    ledger["started_at"] = utc_now()
    ledger["completed_at"] = None
    ledger["error"] = None
    ledger["dataset_path"] = str(paths["base"])
    _save_status(status_path, status_doc)
    try:
        video_meta = validate_video(clip_path, expected_ms)
        if not video_meta["has_audio"]:
            raise ValueError("Source clip has no audio stream; this dataset requires original audiovisual Gemini input")
        clip_sha256 = _hash_file(clip_path)
        LOG.debug("[%s/%s] video validated (%dms, audio present)", video_id, clip_id, video_meta["duration_ms"])

        import torch
        cached_tokens = None if force else _load_cached_tokens(paths, torch, clip_sha256)
        tokens_reused = cached_tokens is not None
        if cached_tokens is None:
            tokens, token_times, token_meta = None, None, None
            for attempt in range(1, max_retries + 1):
                try:
                    tokens, token_times, token_meta = extract_videomae(clip_path, video_meta, video_model, device)
                    break
                except Exception:
                    if attempt >= max_retries:
                        raise
                    LOG.exception("VideoMAE attempt %d/%d failed for %s/%s", attempt, max_retries, video_id, clip_id)
                    time.sleep(min(2 ** (attempt - 1), 10))
            assert tokens is not None and token_times is not None and token_meta is not None
            token_meta["clip_sha256"] = clip_sha256
            atomic_torch_save(torch, tokens, paths["tokens"])
            atomic_json(paths["times"], token_times)
            atomic_json(paths["token_meta"], token_meta)
        else:
            tokens, token_times, token_meta = cached_tokens
            if token_meta.get("model_name") != video_model and not force:
                # A different requested model invalidates only the token stage.
                tokens_reused = False
                tokens, token_times, token_meta = extract_videomae(clip_path, video_meta, video_model, device)
                atomic_torch_save(torch, tokens, paths["tokens"])
                atomic_json(paths["times"], token_times)
                atomic_json(paths["token_meta"], token_meta)
        token_action = "reused" if tokens_reused else "generated"
        LOG.info("%s/%s: Video tokens %s (%d x %d)",
                 video_id, clip_id, token_action, tokens.shape[0], tokens.shape[1])

        explanation_prompt, explanation_prompt_name, _ = _model_explanation_prompt()
        explanation_prompt = explanation_prompt.replace("{duration}", str(video_meta["duration_ms"]))
        explanation_hash = _hash_prompt(explanation_prompt)
        explanation = None if force else _cached_explanation(paths["explanation"], explanation_hash, gemini_model, clip_sha256)
        explanation_reused = explanation is not None
        if explanation is None:
            video_bytes = clip_path.read_bytes()
            last = None
            for attempt in range(1, max_retries + 1):
                try:
                    raw = _gemini_json(gemini_model, explanation_prompt, video_bytes)
                    if not isinstance(raw, dict):
                        raise ValueError("Gemini explanation must be a JSON object")
                    for key in ("video_content", "scene_gravity", "scene_interpretation", "audio_asthetics"):
                        if not isinstance(raw.get(key), str) or not raw[key].strip():
                            raise ValueError(f"Gemini explanation missing non-empty {key}")
                    video_type = str(raw.get("video_type", "")).upper()
                    if video_type not in VIDEO_TYPES:
                        raise ValueError(f"Gemini explanation has invalid video_type {video_type!r}")
                    raw["video_type"] = video_type
                    explanation = {**_artifact_provenance(gemini_model, explanation_prompt_name, explanation_hash, clip_path, clip_sha256, True),
                                   "video_duration_ms": video_meta["duration_ms"], "video_type": video_type,
                                   "video_type_source": "gemini", "response": raw}
                    atomic_json(paths["explanation"], explanation)
                    break
                except Exception as exc:
                    last = exc
                    if attempt >= max_retries:
                        raise
                    LOG.warning("Gemini explanation attempt %d/%d failed: %s", attempt, max_retries, exc)
                    time.sleep(min(2 ** (attempt - 1), 10))
            if explanation is None:
                raise RuntimeError(f"Gemini explanation failed: {last}")
        explanation_action = "reused" if explanation_reused else "generated"
        LOG.info("%s/%s: Gemini explanation %s (%s)",
                 video_id, clip_id, explanation_action, explanation["video_type"])

        cue_prompt_name = "gemini_decide_audio_cues_prompt_without_narrator_video+visualmellow_dsp_cue_annotation_v1"
        cue_prompt = _build_cue_prompt(video_meta["duration_ms"], explanation["response"])
        cue_hash = _hash_prompt(cue_prompt)
        cue_artifact = None if force else _cached_cues(paths["cues"], cue_hash, video_meta["duration_ms"], gemini_model, clip_sha256)
        cues_reused = cue_artifact is not None
        if cue_artifact is None:
            video_bytes = clip_path.read_bytes()
            last = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = _gemini_json(gemini_model, cue_prompt, video_bytes)
                    cues, normalized = normalize_cues(result, video_meta["duration_ms"])
                    cue_artifact = {**_artifact_provenance(gemini_model, cue_prompt_name, cue_hash, clip_path, clip_sha256, True),
                                    "annotation_type": "pseudo_ground_truth", "source": "gemini",
                                    "normalizations": normalized, "cues": cues}
                    atomic_json(paths["cues"], cue_artifact)
                    break
                except Exception as exc:
                    last = exc
                    if attempt >= max_retries:
                        raise
                    LOG.warning("Gemini cue attempt %d/%d failed: %s", attempt, max_retries, exc)
                    time.sleep(min(2 ** (attempt - 1), 10))
            if cue_artifact is None:
                raise RuntimeError(f"Gemini cue annotation failed: {last}")
        cues_action = "reused" if cues_reused else "generated"
        LOG.info("%s/%s: Cue annotations %s (%d cues)",
                 video_id, clip_id, cues_action, len(cue_artifact["cues"]))

        record = {"schema_version": "1.0.0", "pipeline_version": PIPELINE_VERSION,
                  "video_id": video_id, "clip_id": clip_id,
                  "source_start_ms": int(clip_entry["source_start_ms"]), "source_end_ms": int(clip_entry["source_end_ms"]),
                  "clip_duration_ms": video_meta["duration_ms"],
                  "source_video": {k: source_video_metadata.get(k)
                                   for k in ("video_id", "youtube_url", "title", "source_duration_ms", "fps", "width", "height")},
                  "video": {"path": str(clip_path), **video_meta},
                  "features": {"video_tokens": str(paths["tokens"].relative_to(output_dir)),
                               "token_timestamps": str(paths["times"].relative_to(output_dir)),
                               "token_metadata": str(paths["token_meta"].relative_to(output_dir))},
                  "gemini_explanation": {"artifact": str(paths["explanation"].relative_to(output_dir)),
                                        "video_type": explanation["video_type"], "video_type_source": "gemini",
                                        "gemini_received_audio": True},
                  "audio_cues": cue_artifact["cues"],
                  "annotation_provenance": {"source": "gemini", "annotation_type": "pseudo_ground_truth",
                                             "human_verified": False, "gemini_received_audio": True,
                                             "model": gemini_model, "cue_prompt_name": cue_prompt_name,
                                             "cue_prompt_sha256": cue_hash, "generated_at": cue_artifact["generation_timestamp"],
                                             "normalizations": cue_artifact["normalizations"]}}
        atomic_json(paths["record"], record)
        _upsert_jsonl(output_dir / "dataset.jsonl", record)
        validate_record(record, video_meta["duration_ms"], paths, output_dir)
        _update_splits_and_metadata(output_dir, video_dir.parent, video_id, video_model, gemini_model,
                                    int(source_video_metadata.get("clip_duration_ms") or expected_ms))
        validate_completed(paths, output_dir, video_id, clip_id)
        ledger["status"] = "completed"
        ledger["completed_at"] = utc_now()
        ledger["error"] = None
        _save_status(status_path, status_doc)
        LOG.debug("[%s/%s] dataset record written; status=completed", video_id, clip_id)
        return "processed"
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            ledger["status"] = "pending"
            ledger["error"] = "Interrupted; will resume using cached stage artifacts"
        else:
            ledger["status"] = "failed"
            ledger["error"] = f"{type(exc).__name__}: {exc}"[:4000]
            ledger["failed_at"] = utc_now()
        _save_status(status_path, status_doc)
        if isinstance(exc, KeyboardInterrupt):
            raise
        LOG.exception("[%s/%s] ✗ failed: %s", video_id, clip_id, exc)
        return "failed"


def validate_record(record: dict[str, Any], duration_ms: int, paths: dict[str, Path], output_dir: Path) -> None:
    """Validate one dataset row and every artifact it references."""
    if not isinstance(record.get("source_video"), dict):
        raise ValueError("Dataset record is missing source-video metadata")
    if abs((int(record.get("source_end_ms", -1)) - int(record.get("source_start_ms", -1))) - duration_ms) > 300:
        raise ValueError("Record source interval does not match clip duration")
    validate_cues(record.get("audio_cues"), duration_ms)
    for key in ("tokens", "times", "token_meta", "explanation", "cues"):
        if not paths[key].is_file() or paths[key].stat().st_size == 0:
            raise ValueError(f"Required artifact absent/empty: {paths[key]}")
    import torch
    tokens = torch.load(paths["tokens"], map_location="cpu", weights_only=True)
    times = _load_json(paths["times"])
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[1] != _load_json(paths["token_meta"])["hidden_dimension"]:
        raise ValueError("VideoMAE tensor has invalid dimensions")
    if len(times) != tokens.shape[0] or any(not (0 <= int(t["start_ms"]) < int(t["end_ms"]) <= duration_ms) for t in times):
        raise ValueError("Token-to-time mapping is invalid")
    explanation = _load_json(paths["explanation"])
    response = explanation.get("response")
    if (not isinstance(response, dict)
            or any(not isinstance(response.get(k), str) or not response[k].strip()
                   for k in ("video_content", "scene_gravity", "scene_interpretation", "audio_asthetics"))
            or response.get("video_type") not in VIDEO_TYPES
            or explanation.get("gemini_received_audio") is not True):
        raise ValueError("Gemini explanation artifact is invalid or lacks audiovisual provenance")
    cue_artifact = _load_json(paths["cues"])
    if cue_artifact.get("annotation_type") != "pseudo_ground_truth" or cue_artifact.get("source") != "gemini" or cue_artifact.get("gemini_received_audio") is not True:
        raise ValueError("Cue annotation provenance is invalid")
    validate_cues(cue_artifact.get("cues"), duration_ms)
    if cue_artifact.get("cues") != record.get("audio_cues"):
        raise ValueError("Cue artifact and dataset record disagree")
    provenance = record.get("annotation_provenance", {})
    if provenance.get("source") != "gemini" or provenance.get("annotation_type") != "pseudo_ground_truth" or provenance.get("human_verified") is not False:
        raise ValueError("Dataset record provenance is invalid")
    if not paths["record"].is_file() or not paths["record"].stat().st_size:
        raise ValueError("Dataset record was not written")
    if not any(json.loads(line).get("video_id") == record["video_id"] and json.loads(line).get("clip_id") == record["clip_id"]
               for line in (output_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()):
        raise ValueError("Dataset JSONL record is missing")


def validate_completed(paths: dict[str, Path], output_dir: Path, video_id: str, clip_id: str) -> None:
    """Validate the saved record and required files before treating a clip as complete."""
    for key in ("tokens", "times", "token_meta", "explanation", "cues", "record"):
        if not paths[key].is_file() or not paths[key].stat().st_size:
            raise ValueError(f"Missing artifact {key}")
    record = _load_json(paths["record"])
    validate_record(record, int(record["clip_duration_ms"]), paths, output_dir)
    if record.get("video_id") != video_id or record.get("clip_id") != clip_id:
        raise ValueError("Record identity mismatch")


def _upsert_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Atomically replace or append one clip record in the JSONL dataset index."""
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                old = json.loads(line)
                rows[(old.get("video_id"), old.get("clip_id"))] = old
            except json.JSONDecodeError:
                LOG.warning("Ignoring malformed existing JSONL row")
    rows[(record["video_id"], record["clip_id"])] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for key in sorted(rows):
            f.write(json.dumps(rows[key], ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _split(video_id: str) -> str:
    """Assign a source video deterministically to a dataset split."""
    # Stable SHA-256 bucketing keeps all clips from the same source video together.
    x = int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if x < 80 else "validation" if x < 90 else "test"


def _update_splits_and_metadata(output_dir: Path, input_dir: Path, current_video_id: str, video_model: str, gemini_model: str, clip_duration_ms: int) -> None:
    """Rebuild source-level split lists and aggregate dataset metadata."""
    ids = set()
    dataset = output_dir / "dataset.jsonl"
    if dataset.exists():
        for line in dataset.read_text(encoding="utf-8").splitlines():
            try:
                ids.add(str(json.loads(line)["video_id"]))
            except Exception:
                continue
    ids.add(current_video_id)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()] if dataset.exists() else []
    total_source_clips = 0
    for source_dir in (p for p in input_dir.iterdir() if p.is_dir()):
        try:
            source_metadata = _load_json(source_dir / "video_metadata.json")
            total_source_clips += len(source_metadata.get("clips", []))
            ids.add(str(source_metadata.get("video_id") or source_dir.name))
        except Exception:
            continue
    # Source clips include pending/failed clips; records count completed clips.
    total_source_clips = max(total_source_clips, len(rows))
    splits = {name: sorted(v for v in ids if _split(v) == name) for name in ("train", "validation", "test")}
    atomic_json(output_dir / "splits.json", splits)
    atomic_json(output_dir / "dataset_metadata.json", {"dataset_version": "1.0.0", "pipeline_version": PIPELINE_VERSION,
                "created_at": utc_now(), "clip_duration_ms": clip_duration_ms, "overlap_ms": 0,
                "video_model": video_model, "gemini_model": gemini_model, "gemini_received_audio": True,
                "annotation_type": "pseudo_ground_truth", "annotation_source": "Gemini",
                "number_of_source_videos": len(ids), "number_of_clips": total_source_clips,
                "number_of_completed_clips": len(rows), "split_policy": "stable SHA-256 bucket by source video_id (80/10/10)"})


def _pid_is_alive(pid: int) -> bool:
    """Check whether a process ID exists using the current platform's native API."""
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) reports WinError 87 rather than ESRCH for missing PIDs.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        from ctypes import wintypes
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5  # Access denied means the process exists.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(output_dir: Path) -> Path:
    """Acquire the single-builder lock and recover it only if its owner has exited."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".build_dataset.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # A killed process cannot run its finally block. Remove the lock only
        # after checking the recorded PID is no longer alive, then acquire once.
        owner = None
        try:
            owner = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            owner = None
        if owner is not None:
            pid = int(owner.get("pid", -1))
            if _pid_is_alive(pid):
                raise RuntimeError(f"Another dataset builder is still running (pid={pid}, lock={lock})")
        else:
            try:
                if time.time() - lock.stat().st_mtime < 30:
                    raise RuntimeError(f"Unverifiable recent dataset lock: {lock}")
            except FileNotFoundError:
                pass
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Another dataset builder acquired the lock: {lock}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
    return lock


def discover(input_dir: Path, video_filter: str | None, clip_filter: str | None) -> list[tuple[Path, dict[str, Any], Path, dict[str, Any]]]:
    """Find valid clips under the input directory and apply optional ID filters."""
    found = []
    for video_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if video_filter and video_dir.name != video_filter:
            continue
        try:
            metadata, status = _read_clip_table(video_dir)
        except Exception as exc:
            LOG.exception("Skipping invalid video directory %s: %s", video_dir, exc)
            continue
        clips = metadata.get("clips")
        if not isinstance(clips, list):
            raise ValueError(f"Invalid clips array in {video_dir / 'video_metadata.json'}")
        status_path = video_dir / "processing_status.json"
        for clip in clips:
            if clip_filter and str(clip.get("clip_id")) != clip_filter:
                continue
            found.append((video_dir, clip, status_path, status))
    return found


def main(argv: list[str] | None = None) -> int:
    """Parse build options, process discovered clips, and update dataset summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "Dataset" / "raw_videos")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Dataset" / "extracted")
    parser.add_argument("--video-id")
    parser.add_argument("--clip-id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-model", default=VIDEO_MODEL_DEFAULT)
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_VIDEO_EXPLAIN_MODEL", GEMINI_MODEL_DEFAULT))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    _configure_logging(args.log_level)
    _load_local_env()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    lock = None
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    try:
        lock = acquire_lock(args.output_dir)
        items = discover(args.input_dir, args.video_id, args.clip_id)
        if args.video_id and not items:
            raise ValueError(f"No clips found for video ID {args.video_id}")
        if args.clip_id and not items:
            raise ValueError(f"No clips found for clip ID {args.clip_id}")
        total = len(items)
        if not items:
            raise RuntimeError("No clips discovered under input directory")
        # Group clips so normal progress output summarizes videos instead of listing every clip.
        videos: dict[str, list[tuple[Path, dict[str, Any], Path, dict[str, Any]]]] = {}
        for item in items:
            videos.setdefault(item[0].name, []).append(item)

        total_videos = len(videos)
        videos_skipped = 0
        for video_id, video_items in videos.items():
            video_counts = {"processed": 0, "skipped": 0, "failed": 0}
            for video_dir, clip, status_path, status in video_items:
                outcome = process_clip(video_dir, clip, args.output_dir, status_path, status,
                                       video_model=args.video_model, gemini_model=args.gemini_model,
                                       device=args.device, max_retries=args.max_retries, force=args.force)
                counts[outcome] += 1
                video_counts[outcome] += 1

            if video_counts["skipped"] == len(video_items):
                videos_skipped += 1

        LOG.info(
            "Dataset build complete: skipped %d/%d videos and %d/%d clips; processed %d clips; failed %d.",
            videos_skipped, total_videos, counts["skipped"], total, counts["processed"], counts["failed"],
        )
        return 1 if counts["failed"] else 0
    except KeyboardInterrupt:
        LOG.warning("Interrupted. Completed artifacts are cached; processing clip will be retried next run.")
        return 130
    except Exception:
        LOG.exception("Dataset build could not start")
        return 1
    finally:
        if lock:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
