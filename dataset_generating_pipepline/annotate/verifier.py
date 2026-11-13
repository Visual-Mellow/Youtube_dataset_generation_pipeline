"""Layer 2 — rule-based verifier + optional HIGH judge.

Every action is logged (never silent). Rejected cues move to ``rejected_cues``
rather than disappearing. Timing authority stays with DSP (snap); the judge
may only keep / drop / relabel.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from core.schemas import PipelineConfig
from annotate.snap import snap_all
from annotate.prompts import JUDGE_PROMPT
from perception.anchors import format_anchors_for_prompt

LOG = logging.getLogger("visualmellow.annotate.verifier")

_STOP = {"a", "an", "the", "of", "on", "in", "with", "and", "continuous", "close", "mid", "distant", "not"}


def _stem(t: str) -> str:
    for suf in ("ing", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _tokens(text: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP}


def _text_sim(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def verify_cues(
    cues: list[dict[str, Any]],
    *,
    anchors_doc: dict[str, Any],
    explanation: dict[str, Any],
    duration_ms: int,
    cfg: PipelineConfig,
    llm_call=None,
    clip_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Run rules (+ optional judge). Return final artifacts dict."""
    anchors = list(anchors_doc.get("anchors") or [])
    log: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    # Snapshot pre-verifier values on every cue.
    working = []
    for i, raw in enumerate(cues):
        c = dict(raw)
        c.setdefault("cue_id", f"cue_{i + 1:03d}")
        c["proposed"] = {
            "start_time_ms": int(c.get("start_time_ms") or 0),
            "duration_ms": int(c.get("duration_ms") or 0),
            "end_time_ms": int(c.get("end_time_ms") or 0),
            "audio_class": c.get("audio_class"),
            "audio_type": c.get("audio_type"),
        }
        c["verifier_actions"] = []
        c["processing_level"] = cfg.level
        working.append(c)

    # --- rule: set_audibility ------------------------------------------------
    working, acts = _rule_audibility(working, anchors, cfg)
    _attach_actions(working, acts, log)

    # --- rule: split multi-onset blobs --------------------------------------
    working, acts = _rule_split(working, anchors, cfg)
    _attach_actions(working, acts, log)

    # --- rule: merge near-duplicates ----------------------------------------
    working, acts, dropped = _rule_merge(working, cfg)
    for d in dropped:
        rejected.append(d)
    _attach_actions(working, acts, log)

    # --- rule: snap ---------------------------------------------------------
    working, acts = snap_all(working, anchors, cfg, duration_ms)
    _attach_actions(working, acts, log)

    # --- rule: soft-reject not_audible SFX / non-ASR dialogue ---------------
    kept = []
    for c in working:
        a_type = str(c.get("audio_type") or "").upper()
        src = str(c.get("source") or "")
        should_reject = (
            c.get("audibility") == "not_audible"
            and src != "asr"
            and a_type in {"SFX", "DIALOGUE", "NARRATOR"}
        )
        if should_reject:
            action = {
                "cue_id": c.get("cue_id"),
                "action": "reject",
                "actor": "rule:not_audible",
                "before": {"audibility": c.get("audibility"), "audio_type": a_type},
                "after": {"status": "rejected"},
                "evidence": {"anchor_id": c.get("anchor_id")},
                "reason": "no matching onset/speech_start within audibility tolerance",
            }
            c.setdefault("verifier_actions", []).append(len(log))
            log.append(action)
            rejected.append(c)
        else:
            kept.append(c)
    working = kept

    # --- optional CLAP soft similarity (never hard-rejects) -----------------
    working, acts = _rule_clap_soft(working, clip_bytes)
    _attach_actions(working, acts, log)

    # --- HIGH judge ---------------------------------------------------------
    if cfg.flags.llm_judge and llm_call is not None:
        working, rejected, acts = _run_judge(
            working, rejected, anchors, explanation, duration_ms, cfg, llm_call, clip_bytes,
        )
        _attach_actions(working, acts, log)

    # Final clamp + provenance.
    final = []
    for i, c in enumerate(working):
        start = max(0, int(c.get("start_time_ms") or 0))
        start = min(start, max(duration_ms - 1, 0))
        dur = max(50, int(c.get("duration_ms") or 50))
        dur = min(dur, max(duration_ms - start, 50))
        end = start + dur
        c["start_time_ms"] = start
        c["duration_ms"] = dur
        c["end_time_ms"] = end
        c["cue_id"] = c.get("cue_id") or f"cue_{i + 1:03d}"
        c["audio_prompt"] = c.get("audio_prompt") or c.get("audio_class")
        c["source"] = "gemini+dsp"
        c["annotation_type"] = "pseudo_ground_truth"
        c["temporal_annotation"] = {
            "start_time_ms": start,
            "duration_ms": dur,
            "end_time_ms": end,
            "source": "gemini+dsp",
            "annotation_type": "pseudo_ground_truth",
            "snap_delta_ms": c.get("snap_delta_ms"),
            "anchor_id": c.get("anchor_id"),
        }
        # Collect tagger from nearest onset anchor.
        if not c.get("tagger_topk") and c.get("anchor_id"):
            for a in anchors:
                if a.get("id") == c["anchor_id"]:
                    c["tagger_topk"] = a.get("tagger_topk") or []
                    c["on_music_beat"] = a.get("on_music_beat", False)
                    break
        final.append(c)

    # Deduplicate cue_ids.
    seen = set()
    for c in final:
        cid = c["cue_id"]
        if cid in seen:
            c["cue_id"] = f"{cid}_{c['start_time_ms']}"
        seen.add(c["cue_id"])

    LOG.info(
        "Verifier (%s): kept %d, rejected %d, log entries %d",
        cfg.level, len(final), len(rejected), len(log),
    )
    return {
        "cues": final,
        "rejected_cues": rejected,
        "verifier_log": log,
    }


def _attach_actions(
    cues: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    log: list[dict[str, Any]],
) -> None:
    by_id = {c.get("cue_id"): c for c in cues}
    for action in actions:
        idx = len(log)
        log.append(action)
        cid = action.get("cue_id")
        if cid and cid in by_id:
            by_id[cid].setdefault("verifier_actions", []).append(idx)


def _rule_clap_soft(
    cues: list[dict[str, Any]],
    clip_bytes: bytes | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Optional CLAP text↔audio similarity; soft signal only (never rejects)."""
    actions: list[dict[str, Any]] = []
    if not cues:
        return cues, actions
    try:
        import importlib

        importlib.import_module("laion_clap")
    except Exception:
        LOG.info("CLAP soft-check skipped (laion_clap not installed)")
        actions.append({
            "action": "clap_skip",
            "actor": "rule:clap_soft",
            "reason": "laion_clap unavailable",
            "evidence": {},
        })
        return cues, actions

    # Soft path reserved: when CLAP is installed, callers can enrich later.
    # We intentionally do not hard-fail or download large weights here.
    LOG.info("CLAP available — soft scores not wired for batch yet; logging presence only")
    actions.append({
        "action": "clap_available",
        "actor": "rule:clap_soft",
        "reason": "laion_clap importable; scores deferred",
        "evidence": {"n_cues": len(cues), "has_clip_bytes": bool(clip_bytes)},
    })
    return cues, actions


def _rule_audibility(
    cues: list[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mark each SFX/DIALOGUE cue audible / partially_audible / not_audible."""
    actions = []
    onset_times = [
        (int(a["t_ms"]), a)
        for a in anchors
        if a.get("kind") in {"audio_onset", "speech_start", "sed_event"}
    ]
    tol = cfg.snap.audibility_tol_ms
    out = []
    for c in cues:
        a_type = str(c.get("audio_type") or "").upper()
        start = int(c.get("start_time_ms") or 0)
        audibility = "audible"
        evidence: dict[str, Any] = {}
        if a_type in {"SFX", "DIALOGUE", "NARRATOR"}:
            near = None
            best_d = None
            for t, a in onset_times:
                # For SFX prefer effects/mixture; dialogue prefers speech_start.
                if a_type == "SFX" and a.get("kind") == "speech_start":
                    continue
                if a_type in {"DIALOGUE", "NARRATOR"} and a.get("kind") != "speech_start" and a.get("stem") != "dialogue":
                    continue
                d = abs(t - start)
                if best_d is None or d < best_d:
                    best_d, near = d, a
            if near is None or best_d is None:
                audibility = "not_audible"
            elif best_d <= tol:
                audibility = "audible"
                evidence = {"anchor_id": near.get("id"), "delta_ms": best_d}
                if not c.get("anchor_id"):
                    c = dict(c)
                    c["anchor_id"] = near.get("id")
            elif best_d <= tol * 3:
                audibility = "partially_audible"
                evidence = {"anchor_id": near.get("id"), "delta_ms": best_d}
            else:
                audibility = "not_audible"
                evidence = {"nearest_delta_ms": best_d}
        c = dict(c)
        c["audibility"] = audibility
        action = {
            "cue_id": c.get("cue_id"),
            "action": "set_audibility",
            "actor": "rule:onset_proximity",
            "before": {},
            "after": {"audibility": audibility},
            "evidence": evidence,
            "reason": f"audibility={audibility}",
        }
        actions.append(action)
        out.append(c)
    return out, actions


def _rule_split(
    cues: list[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split long SFX EVENT cues that cover ≥2 distinct effects onsets."""
    actions = []
    out: list[dict[str, Any]] = []
    gap = cfg.snap.min_impact_gap_ms
    effects = [
        int(a["t_ms"])
        for a in anchors
        if a.get("kind") == "audio_onset" and a.get("stem") in {"effects", "mixture"}
    ]
    for c in cues:
        a_type = str(c.get("audio_type") or "").upper()
        event = str(c.get("event_type") or "").upper()
        start = int(c.get("start_time_ms") or 0)
        end = int(c.get("end_time_ms") or start + int(c.get("duration_ms") or 0))
        if a_type != "SFX" or event in {"CONTINUOUS", "GLOBAL", "SEGMENT"}:
            out.append(c)
            continue
        span = end - start
        if span < 1500:
            out.append(c)
            continue
        hits = [t for t in effects if start - 50 <= t <= end + 50]
        # Collapse hits closer than gap.
        collapsed: list[int] = []
        for t in sorted(hits):
            if not collapsed or t - collapsed[-1] >= gap:
                collapsed.append(t)
        if len(collapsed) < 2:
            out.append(c)
            continue
        # Promote parent to CONTINUOUS with impact_times; emit children.
        parent = dict(c)
        parent["event_type"] = "CONTINUOUS"
        parent["impact_times_ms"] = collapsed
        parent["audio_class"] = c.get("audio_class") or ""
        if not str(parent["audio_class"]).upper().startswith("CONTINUOUS"):
            parent["audio_class"] = f"CONTINUOUS: {parent['audio_class']}"
            parent["audio_prompt"] = parent["audio_class"]
        children = []
        for j, t in enumerate(collapsed):
            child = dict(c)
            child["cue_id"] = f"{c.get('cue_id', 'cue')}_h{j + 1}"
            child["start_time_ms"] = t
            child["duration_ms"] = min(400, max(100, gap))
            child["end_time_ms"] = child["start_time_ms"] + child["duration_ms"]
            child["event_type"] = "INSTANT"
            child["proposed"] = c.get("proposed")
            child["verifier_actions"] = []
            children.append(child)
        out.append(parent)
        out.extend(children)
        actions.append({
            "cue_id": c.get("cue_id"),
            "action": "split",
            "actor": "rule:multi_onset",
            "before": {"start_time_ms": start, "duration_ms": span, "event_type": event},
            "after": {
                "children": [ch["cue_id"] for ch in children],
                "impact_times_ms": collapsed,
                "parent_event_type": "CONTINUOUS",
            },
            "evidence": {"effects_onsets_in_span": len(collapsed)},
            "reason": f"{len(collapsed)} distinct effects onsets inside a single {event} cue",
        })
    return out, actions


def _rule_merge(
    cues: list[dict[str, Any]],
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop near-duplicate cues (same type, ≤100 ms, text sim ≥0.6)."""
    actions = []
    rejected = []
    kept: list[dict[str, Any]] = []
    for c in sorted(cues, key=lambda x: int(x.get("start_time_ms") or 0)):
        dup_of = None
        for k in kept:
            if str(k.get("audio_type") or "").upper() != str(c.get("audio_type") or "").upper():
                continue
            if abs(int(k.get("start_time_ms") or 0) - int(c.get("start_time_ms") or 0)) > 100:
                continue
            if _text_sim(str(k.get("audio_class") or ""), str(c.get("audio_class") or "")) >= 0.6:
                dup_of = k
                break
        if dup_of is not None:
            actions.append({
                "cue_id": c.get("cue_id"),
                "action": "merge",
                "actor": "rule:near_duplicate",
                "before": {"audio_class": c.get("audio_class"), "start_time_ms": c.get("start_time_ms")},
                "after": {"merged_into": dup_of.get("cue_id")},
                "evidence": {},
                "reason": f"duplicate of {dup_of.get('cue_id')}",
            })
            c = dict(c)
            c["reject_reason"] = f"merged into {dup_of.get('cue_id')}"
            rejected.append(c)
        else:
            kept.append(c)
    return kept, actions, rejected


def _run_judge(
    cues: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
    explanation: dict[str, Any],
    duration_ms: int,
    cfg: PipelineConfig,
    llm_call,
    clip_bytes: bytes | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """One pro-model judge call; semantics only."""
    payload = []
    for c in cues:
        payload.append({
            "id": c.get("cue_id"),
            "audio_type": c.get("audio_type"),
            "audio_class": c.get("audio_class"),
            "start_time_ms": c.get("start_time_ms"),
            "duration_ms": c.get("duration_ms"),
            "audibility": c.get("audibility"),
            "anchor_id": c.get("anchor_id"),
            "source": c.get("source"),
            "audio_only_class": c.get("audio_only_class"),
            "av_audio_only_disagree": c.get("av_audio_only_disagree"),
            "transcript": c.get("transcript"),
        })
    prompt = JUDGE_PROMPT.format(
        video_duration_ms=duration_ms,
        visual_story=json.dumps(explanation, ensure_ascii=False),
        anchors=format_anchors_for_prompt(list(anchors), 0, duration_ms, limit=80),
        cues=json.dumps(payload, ensure_ascii=False),
    )
    result = llm_call(
        "judge", cfg.models.judge, prompt,
        video_bytes=clip_bytes, mime_type="video/mp4",
        video_fps=cfg.perception.gemini_video_fps,
    )
    verdicts: dict[str, dict] = {}
    items = []
    if isinstance(result, dict):
        items = result.get("verdicts") or result.get("audio_cues") or []
    elif isinstance(result, list):
        items = result
    for v in items:
        if isinstance(v, dict) and v.get("id") is not None:
            verdicts[str(v["id"])] = v

    if not verdicts:
        LOG.warning("Judge returned no verdicts; keeping all %d cues", len(cues))
        return cues, rejected, []

    actions = []
    kept = []
    for c in cues:
        v = verdicts.get(str(c.get("cue_id")))
        if v is None:
            kept.append(c)
            continue
        verdict = str(v.get("verdict") or "keep").strip().lower()
        reason = str(v.get("reason") or "").strip()
        if verdict == "drop":
            actions.append({
                "cue_id": c.get("cue_id"),
                "action": "reject",
                "actor": "judge",
                "before": {"audio_class": c.get("audio_class")},
                "after": {"status": "rejected"},
                "evidence": {"grounded": v.get("grounded")},
                "reason": reason or "judge drop",
            })
            rc = dict(c)
            rc["reject_reason"] = reason or "judge drop"
            rejected.append(rc)
            continue
        if verdict == "relabel" and v.get("audio_class"):
            new_label = str(v["audio_class"]).strip()
            actions.append({
                "cue_id": c.get("cue_id"),
                "action": "relabel",
                "actor": "judge",
                "before": {"audio_class": c.get("audio_class")},
                "after": {"audio_class": new_label},
                "evidence": {"grounded": v.get("grounded")},
                "reason": reason or "judge relabel",
            })
            c = dict(c)
            c["audio_class"] = new_label
            c["audio_prompt"] = new_label
        else:
            actions.append({
                "cue_id": c.get("cue_id"),
                "action": "keep",
                "actor": "judge",
                "before": {},
                "after": {"verdict": "keep"},
                "evidence": {"grounded": v.get("grounded")},
                "reason": reason or "judge keep",
            })
        kept.append(c)
    LOG.info("Judge: kept %d / %d (dropped %d)", len(kept), len(cues), len(cues) - len(kept))
    return kept, rejected, actions
