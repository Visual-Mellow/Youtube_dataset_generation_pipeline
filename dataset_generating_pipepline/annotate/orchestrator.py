"""Annotate one clip end-to-end: explain → propose → verify → write artifacts.

All Gemini traffic goes through ``_llm_call`` so every request lands in
``llm_calls.jsonl`` with stage, model, prompt hash, latency, and a pointer
to the raw response file under ``llm/``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.schemas import PipelineConfig
from annotate.prompts import EXPLANATION_PROMPT
from annotate.proposer import propose_cues
from annotate.verifier import verify_cues
from utils.ui_log import UILogWriter

LOG = logging.getLogger("visualmellow.annotate.orchestrator")

VIDEO_TYPES = {
    "ACTION", "DIALOGUE", "ESTABLISHING", "CHASE", "FIGHT",
    "REVEAL", "EMOTIONAL", "HORROR_TENSION", "TRAVEL", "OTHER",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class _ClipLogger:
    """Tee INFO-level stage lines into ``pipeline.log`` and optional ``pipeline.ui.log``."""

    def __init__(self, path: Path, *, ui_log: UILogWriter | None = None):
        self.path = path
        self.ui = ui_log
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def info(self, msg: str, *args, step_id: str | None = None) -> None:
        line = msg % args if args else msg
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        text = f"[{stamp}] {line}"
        LOG.info("%s", line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        if self.ui is not None:
            self.ui.message(text, step_id=step_id)


def annotate_clip(
    *,
    clip_path: Path,
    duration_ms: int,
    anchors_doc: dict[str, Any],
    cfg: PipelineConfig,
    out_dir: Path,
    clip_sha256: str,
    force: bool = False,
    query_fn: Callable | None = None,
    ui_log: UILogWriter | None = None,
    ui_log_fresh: bool = True,
) -> dict[str, Any]:
    """Run explanation + proposer + verifier for one clip at ``cfg.level``.

    Writes under ``out_dir`` (typically
    ``features/<vid>/<clip>/annotations/<level>/``):

    - gemini_explanation.json
    - cue_annotations.json
    - rejected_cues.json
    - verifier_log.jsonl
    - llm_calls.jsonl
    - llm/<stage>_<n>.json  (raw responses)
    - pipeline.log
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ui_log is None:
        ui_log = UILogWriter(
            out_dir / "pipeline.ui.log",
            fresh=ui_log_fresh,
            meta={
                "clip_path": str(clip_path),
                "clip_sha256": clip_sha256,
                "duration_ms": duration_ms,
                "level": cfg.level,
                "stage": "annotate",
            },
        )
    plog = _ClipLogger(out_dir / "pipeline.log", ui_log=ui_log)
    llm_log_path = out_dir / "llm_calls.jsonl"
    llm_dir = out_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)

    # Resume: reuse explanation + cues when provenance still matches.
    if not force:
        cached = _try_reuse(out_dir, clip_sha256, cfg)
        if cached is not None:
            plog.info("Reusing cached annotations for level=%s", cfg.level, step_id="annotate")
            ui_log.message(f"Reusing cached annotations for level={cfg.level}")
            return cached

    # Fresh llm log for this run.
    if llm_log_path.exists() and force:
        llm_log_path.unlink()

    clip_bytes = clip_path.read_bytes()
    call_counter = {"n": 0}

    def llm_call(
        stage: str,
        model: str,
        prompt: str,
        *,
        video_bytes: bytes | None = None,
        audio_bytes: bytes | None = None,
        mime_type: str = "video/mp4",
        video_fps: int | None = None,
    ) -> Any:
        return _logged_llm_call(
            stage=stage,
            model=model,
            prompt=prompt,
            video_bytes=video_bytes,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            video_fps=video_fps,
            llm_log_path=llm_log_path,
            llm_dir=llm_dir,
            call_counter=call_counter,
            query_fn=query_fn,
            plog=plog,
            ui_log=ui_log,
        )

    ui_log.step_begin(
        "annotate",
        f"Annotate clip (level={cfg.level})",
        inputs={
            "clip_path": str(clip_path),
            "clip_sha256": clip_sha256,
            "duration_ms": duration_ms,
            "anchor_counts": (anchors_doc.get("counts") if isinstance(anchors_doc, dict) else None),
            "models": {"explain": cfg.models.explain, "proposer": cfg.models.proposer},
            "flags": {
                "per_window_proposer": cfg.flags.per_window_proposer,
                "audio_only_pass": cfg.flags.audio_only_pass,
                "llm_judge": cfg.flags.llm_judge,
            },
        },
    )

    # ----- explanation ------------------------------------------------------
    t0 = time.perf_counter()
    explanation_prompt = EXPLANATION_PROMPT.format(
        video_duration_ms=duration_ms,
        director_prompt="(none provided)",
    )
    ui_log.step_begin(
        "explanation",
        f"Gemini explanation ({cfg.models.explain})",
        inputs={"prompt_template": "EXPLANATION_PROMPT", "duration_ms": duration_ms, "clip_bytes": len(clip_bytes)},
    )
    expl_path = out_dir / "gemini_explanation.json"
    explanation_doc = None
    if expl_path.is_file() and not force:
        try:
            explanation_doc = json.loads(expl_path.read_text(encoding="utf-8"))
            if explanation_doc.get("clip_sha256") != clip_sha256:
                explanation_doc = None
        except Exception:
            explanation_doc = None

    if explanation_doc is None:
        plog.info("Explanation: calling %s", cfg.models.explain, step_id="explanation")
        ui_log.data_block("explanation", "input", "Full prompt", explanation_prompt)
        raw = llm_call(
            "explanation", cfg.models.explain, explanation_prompt,
            video_bytes=clip_bytes, mime_type="video/mp4",
            video_fps=cfg.perception.gemini_video_fps,
        )
        if not isinstance(raw, dict):
            raise RuntimeError("Gemini explanation must be a JSON object")
        for key in ("video_content", "scene_gravity", "scene_interpretation", "audio_asthetics"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise RuntimeError(f"Gemini explanation missing non-empty {key}")
        video_type = str(raw.get("video_type") or "OTHER").upper()
        if video_type not in VIDEO_TYPES:
            video_type = "OTHER"
        raw["video_type"] = video_type
        explanation_doc = {
            "model": cfg.models.explain,
            "prompt_name": "dataset_explanation_audible_v2",
            "prompt_sha256": _hash_text(explanation_prompt),
            "generation_timestamp": _utc_now(),
            "gemini_received_audio": True,
            "source_clip": str(clip_path),
            "clip_sha256": clip_sha256,
            "video_duration_ms": duration_ms,
            "video_type": video_type,
            "video_type_source": "gemini",
            "processing_level": cfg.level,
            "response": raw,
        }
        _atomic_json(expl_path, explanation_doc)
        ui_log.file_written("explanation", expl_path, detail={"video_type": video_type})
    else:
        ui_log.message("Reused cached gemini_explanation.json", step_id="explanation")
        ui_log.data_block("explanation", "output", "Cached explanation", explanation_doc.get("response"))
    plog.info("Explanation done (%s) in %.1fs", explanation_doc["video_type"], time.perf_counter() - t0, step_id="explanation")
    ui_log.step_end(
        "explanation",
        f"video_type={explanation_doc['video_type']}",
        outputs=explanation_doc.get("response"),
    )

    # ----- propose ----------------------------------------------------------
    t1 = time.perf_counter()
    plog.info(
        "Proposer: level=%s per_window=%s audio_only=%s",
        cfg.level, cfg.flags.per_window_proposer, cfg.flags.audio_only_pass,
        step_id="proposer",
    )
    ui_log.step_begin(
        "proposer",
        "Cue proposer",
        inputs={
            "level": cfg.level,
            "per_window": cfg.flags.per_window_proposer,
            "audio_only_pass": cfg.flags.audio_only_pass,
            "explanation_video_type": explanation_doc["video_type"],
            "anchor_total": (anchors_doc.get("counts") or {}).get("total"),
        },
    )
    proposed = propose_cues(
        clip_path=clip_path,
        clip_bytes=clip_bytes,
        duration_ms=duration_ms,
        explanation=explanation_doc["response"],
        anchors_doc=anchors_doc,
        cfg=cfg,
        llm_call=llm_call,
    )
    plog.info("Proposer returned %d raw cues in %.1fs", len(proposed), time.perf_counter() - t1, step_id="proposer")
    ui_log.step_end("proposer", f"{len(proposed)} raw cues", outputs=proposed)

    # ----- verify -----------------------------------------------------------
    t2 = time.perf_counter()
    plog.info("Verifier: judge=%s", cfg.flags.llm_judge, step_id="verifier")
    ui_log.step_begin(
        "verifier",
        "Cue verifier",
        inputs={"llm_judge": cfg.flags.llm_judge, "raw_cue_count": len(proposed)},
    )
    verified = verify_cues(
        proposed,
        anchors_doc=anchors_doc,
        explanation=explanation_doc["response"],
        duration_ms=duration_ms,
        cfg=cfg,
        llm_call=llm_call if cfg.flags.llm_judge else None,
        clip_bytes=clip_bytes if cfg.flags.llm_judge else None,
    )
    plog.info(
        "Verifier kept %d, rejected %d in %.1fs",
        len(verified["cues"]), len(verified["rejected_cues"]), time.perf_counter() - t2,
        step_id="verifier",
    )
    ui_log.data_block("verifier", "output", "Verifier log entries", verified.get("verifier_log"))
    ui_log.step_end(
        "verifier",
        f"kept={len(verified['cues'])} rejected={len(verified['rejected_cues'])}",
        outputs={"cues": verified["cues"], "rejected_cues": verified["rejected_cues"]},
    )

    # ----- write artifacts --------------------------------------------------
    ui_log.step_begin("write_artifacts", "Write annotation artifacts")
    cue_artifact = {
        "model": cfg.models.proposer,
        "prompt_name": "dataset_slot_fill_proposer_v2",
        "prompt_sha256": _hash_text("dataset_slot_fill_proposer_v2"),
        "generation_timestamp": _utc_now(),
        "gemini_received_audio": True,
        "source_clip": str(clip_path),
        "clip_sha256": clip_sha256,
        "annotation_type": "pseudo_ground_truth",
        "source": "gemini+dsp",
        "processing_level": cfg.level,
        "normalizations": [],
        "cues": verified["cues"],
        "rejected_cues": [
            {k: v for k, v in r.items() if k != "verifier_actions"}
            for r in verified["rejected_cues"]
        ],
        "verifier_log": verified["verifier_log"],
        "llm_call_count": call_counter["n"],
    }
    cues_path = out_dir / "cue_annotations.json"
    _atomic_json(cues_path, cue_artifact)
    rej_path = out_dir / "rejected_cues.json"
    _atomic_json(rej_path, {"cues": cue_artifact["rejected_cues"]})

    # Write verifier log as JSONL for easy grepping.
    vlog_path = out_dir / "verifier_log.jsonl"
    if vlog_path.exists():
        vlog_path.unlink()
    for entry in verified["verifier_log"]:
        _append_jsonl(vlog_path, entry)

    ui_log.file_written("write_artifacts", cues_path, detail={"cue_count": len(cue_artifact["cues"])})
    ui_log.file_written("write_artifacts", rej_path)
    ui_log.file_written("write_artifacts", vlog_path)
    ui_log.file_written("write_artifacts", llm_log_path)
    ui_log.file_written("write_artifacts", plog.path)
    ui_log.file_written("write_artifacts", ui_log.path)
    ui_log.step_end(
        "write_artifacts",
        f"{len(verified['cues'])} cues written",
        outputs={"cue_annotations": str(cues_path), "ui_log": str(ui_log.path)},
    )

    plog.info(
        "Done level=%s: %d cues, %d rejected, %d LLM calls, %.1fs total",
        cfg.level, len(verified["cues"]), len(verified["rejected_cues"]),
        call_counter["n"], time.perf_counter() - t0,
        step_id="annotate",
    )
    ui_log.step_end(
        "annotate",
        f"{len(verified['cues'])} cues, {call_counter['n']} LLM calls",
        outputs={"cues": len(verified["cues"]), "rejected": len(verified["rejected_cues"])},
    )
    return {
        "explanation": explanation_doc,
        "cues": verified["cues"],
        "rejected_cues": cue_artifact["rejected_cues"],
        "verifier_log": verified["verifier_log"],
        "cue_artifact": cue_artifact,
        "llm_call_count": call_counter["n"],
        "out_dir": str(out_dir),
    }


def _try_reuse(out_dir: Path, clip_sha: str, cfg: PipelineConfig) -> dict[str, Any] | None:
    expl = out_dir / "gemini_explanation.json"
    cues = out_dir / "cue_annotations.json"
    if not expl.is_file() or not cues.is_file():
        return None
    try:
        explanation = json.loads(expl.read_text(encoding="utf-8"))
        cue_artifact = json.loads(cues.read_text(encoding="utf-8"))
        if explanation.get("clip_sha256") != clip_sha or cue_artifact.get("clip_sha256") != clip_sha:
            return None
        if cue_artifact.get("processing_level") != cfg.level:
            return None
        if cue_artifact.get("source") not in {"gemini+dsp", "gemini"}:
            return None
        rejected = cue_artifact.get("rejected_cues") or []
        vlog = cue_artifact.get("verifier_log") or []
        return {
            "explanation": explanation,
            "cues": cue_artifact.get("cues") or [],
            "rejected_cues": rejected,
            "verifier_log": vlog,
            "cue_artifact": cue_artifact,
            "llm_call_count": cue_artifact.get("llm_call_count", 0),
            "out_dir": str(out_dir),
            "reused": True,
        }
    except Exception:
        return None


def _logged_llm_call(
    *,
    stage: str,
    model: str,
    prompt: str,
    video_bytes: bytes | None,
    audio_bytes: bytes | None,
    mime_type: str,
    video_fps: int | None,
    llm_log_path: Path,
    llm_dir: Path,
    call_counter: dict,
    query_fn: Callable | None,
    plog: _ClipLogger,
    ui_log: UILogWriter | None = None,
) -> Any:
    """Invoke Gemini (or an injected stub) and persist the call record."""

    def _stage_step_id(name: str) -> str:
        if name == "explanation":
            return "explanation"
        if name.startswith("judge"):
            return "verifier"
        return "proposer"

    step_id = _stage_step_id(stage)
    call_counter["n"] += 1
    n = call_counter["n"]
    prompt_sha = _hash_text(prompt)
    t0 = time.perf_counter()
    error = None
    result = None
    raw_text = None

    try:
        if query_fn is not None:
            result = query_fn(
                model, prompt,
                video_bytes=video_bytes,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                video_fps=video_fps,
            )
        else:
            result = _default_gemini(
                model, prompt,
                video_bytes=video_bytes,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                video_fps=video_fps,
            )
        if isinstance(result, tuple) and len(result) == 2:
            result, raw_text = result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        plog.info("LLM %s FAILED: %s", stage, error, step_id=step_id)
        raise
    finally:
        latency = time.perf_counter() - t0
        raw_path = llm_dir / f"{stage}_{n:02d}.json"
        try:
            _atomic_json(raw_path, {
                "stage": stage,
                "model": model,
                "prompt_sha256": prompt_sha,
                "result": result,
                "raw_text": raw_text,
                "error": error,
            })
        except Exception:
            pass
        _append_jsonl(llm_log_path, {
            "n": n,
            "stage": stage,
            "model": model,
            "prompt_sha256": prompt_sha,
            "prompt_chars": len(prompt),
            "video_bytes": len(video_bytes) if video_bytes else 0,
            "audio_bytes": len(audio_bytes) if audio_bytes else 0,
            "mime_type": mime_type,
            "video_fps": video_fps,
            "latency_s": round(latency, 3),
            "error": error,
            "raw_response": str(raw_path.name),
            "timestamp": _utc_now(),
        })
        plog.info("LLM %s #%d model=%s %.1fs", stage, n, model, latency, step_id=step_id)
        if ui_log is not None:
            ui_log.llm_call(
                step_id,
                stage=stage,
                model=model,
                prompt=prompt,
                result=result,
                raw_text=raw_text,
                latency_s=round(latency, 3),
                video_bytes=len(video_bytes) if video_bytes else 0,
                audio_bytes=len(audio_bytes) if audio_bytes else 0,
                mime_type=mime_type,
                raw_file=raw_path.name,
                error=error,
            )
            ui_log.file_written(step_id, raw_path)

    if result is None:
        raise RuntimeError(f"Gemini {stage} returned no parsed JSON")
    return result


def _default_gemini(
    model: str,
    prompt: str,
    *,
    video_bytes: bytes | None,
    audio_bytes: bytes | None,
    mime_type: str,
    video_fps: int | None,
) -> Any:
    """Call into Utils.llm.query_gemini_multimodal (repo root on sys.path)."""
    # Ensure repo root is importable.
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from Utils.llm import query_gemini_multimodal

    payload = video_bytes if video_bytes is not None else audio_bytes
    if payload is None:
        raise ValueError("llm_call requires video_bytes or audio_bytes")
    effective_mime = mime_type
    if audio_bytes is not None and video_bytes is None:
        effective_mime = mime_type if mime_type.startswith("audio/") else "audio/wav"
    return query_gemini_multimodal(
        model, prompt,
        video_bytes=payload,
        mime_type=effective_mime,
        video_fps=video_fps,
    )
