"""Onset / offset / event metrics for Layer-0 anchors vs gold."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _onset(e: dict[str, Any]) -> float:
    if e.get("onset_ms") is not None:
        return float(e["onset_ms"])
    if e.get("t_ms_f") is not None:
        return float(e["t_ms_f"])
    return float(e.get("t_ms") or e.get("start_time_ms") or 0)


def _offset(e: dict[str, Any]) -> float | None:
    for key in ("offset_ms", "t_end_ms_f", "end_ms", "end_time_ms"):
        if e.get(key) is not None:
            return float(e[key])
    return None


def hungarian_match(
    pred_ms: Sequence[float],
    gold_ms: Sequence[float],
    collar_ms: float = 50.0,
) -> list[tuple[int, int, float]]:
    """One-to-one matches within collar. Returns (pred_i, gold_j, abs_err)."""
    if not pred_ms or not gold_ms:
        return []
    from scipy.optimize import linear_sum_assignment

    p = np.asarray(pred_ms, dtype=np.float64)
    g = np.asarray(gold_ms, dtype=np.float64)
    cost = np.abs(p[:, None] - g[None, :])
    # Forbid matches beyond collar by large cost.
    big = 1e9
    cost = np.where(cost <= collar_ms, cost, big)
    ri, ci = linear_sum_assignment(cost)
    out: list[tuple[int, int, float]] = []
    for i, j in zip(ri, ci):
        err = float(cost[i, j])
        if err >= big / 2:
            continue
        out.append((int(i), int(j), err))
    return out


def onset_errors(
    pred_events: Sequence[dict[str, Any]],
    gold_events: Sequence[dict[str, Any]],
    collar_ms: float = 200.0,
) -> dict[str, Any]:
    pred = [_onset(e) for e in pred_events]
    gold = [_onset(e) for e in gold_events]
    matches = hungarian_match(pred, gold, collar_ms=collar_ms)
    errs = [m[2] for m in matches]
    n_gold = len(gold)
    n_pred = len(pred)

    def pct_within(tol: float) -> float:
        if not errs:
            return 0.0
        return 100.0 * sum(1 for e in errs if e <= tol) / len(errs)

    return {
        "n_gold": n_gold,
        "n_pred": n_pred,
        "n_matched": len(matches),
        "median_err_ms": float(np.median(errs)) if errs else None,
        "p90_err_ms": float(np.percentile(errs, 90)) if errs else None,
        "mean_err_ms": float(np.mean(errs)) if errs else None,
        "pct_within_20": pct_within(20),
        "pct_within_50": pct_within(50),
        "pct_within_100": pct_within(100),
        "pct_within_200": pct_within(200),
        "errors_ms": errs,
    }


def precision_recall_at_collar(
    pred_events: Sequence[dict[str, Any]],
    gold_events: Sequence[dict[str, Any]],
    collar_ms: float = 50.0,
) -> dict[str, Any]:
    pred = [_onset(e) for e in pred_events]
    gold = [_onset(e) for e in gold_events]
    matches = hungarian_match(pred, gold, collar_ms=collar_ms)
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gold) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "collar_ms": collar_ms}


def offset_errors(
    pred_events: Sequence[dict[str, Any]],
    gold_events: Sequence[dict[str, Any]],
    collar_ms: float = 200.0,
) -> dict[str, Any]:
    """Offset error for onset-matched pairs that both have offsets."""
    pred_on = [_onset(e) for e in pred_events]
    gold_on = [_onset(e) for e in gold_events]
    matches = hungarian_match(pred_on, gold_on, collar_ms=collar_ms)
    errs: list[float] = []
    for pi, gi, _ in matches:
        po = _offset(pred_events[pi])
        go = _offset(gold_events[gi])
        if po is None or go is None:
            continue
        errs.append(abs(po - go))
    return {
        "n_pairs": len(errs),
        "median_err_ms": float(np.median(errs)) if errs else None,
        "p90_err_ms": float(np.percentile(errs, 90)) if errs else None,
        "errors_ms": errs,
    }


def event_f1_sed_eval(
    pred_events: Sequence[dict[str, Any]],
    gold_events: Sequence[dict[str, Any]],
    collar_s: float = 0.2,
    class_aware: bool = False,
) -> dict[str, Any]:
    """Optional sed_eval event-based F1. Falls back to collar F1 if unavailable."""
    try:
        import sed_eval
        import dcase_util
    except Exception:
        pr = precision_recall_at_collar(pred_events, gold_events, collar_ms=collar_s * 1000)
        return {"available": False, "fallback_f1": pr["f1"], "class_aware": class_aware, **pr}

    # Minimal sed_eval path: build event lists as seconds.
    ref = dcase_util.containers.MetaDataContainer([
        {
            "event_label": (e.get("label") or e.get("kind") or "event") if class_aware else "event",
            "event_onset": _onset(e) / 1000.0,
            "event_offset": (_offset(e) or (_onset(e) + 50)) / 1000.0,
        }
        for e in gold_events
    ])
    est = dcase_util.containers.MetaDataContainer([
        {
            "event_label": (e.get("label") or e.get("kind") or e.get("audio_class") or "event") if class_aware else "event",
            "event_onset": _onset(e) / 1000.0,
            "event_offset": (_offset(e) or (_onset(e) + 50)) / 1000.0,
        }
        for e in pred_events
    ])
    labels = sorted({x["event_label"] for x in ref} | {x["event_label"] for x in est}) or ["event"]
    evaluator = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=labels,
        t_collar=collar_s,
        percentage_of_length=0.0,
    )
    evaluator.evaluate(reference_event_list=ref, estimated_event_list=est)
    overall = evaluator.results_overall_metrics()
    return {
        "available": True,
        "class_aware": class_aware,
        "f_measure": overall.get("f_measure", {}).get("f_measure"),
        "precision": overall.get("f_measure", {}).get("precision"),
        "recall": overall.get("f_measure", {}).get("recall"),
        "raw": overall,
    }


def summarize_clip(
    pred_events: Sequence[dict[str, Any]],
    gold_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "onset": onset_errors(pred_events, gold_events),
        "pr50": precision_recall_at_collar(pred_events, gold_events, 50.0),
        "offset": offset_errors(pred_events, gold_events),
        "event_f1_agnostic": event_f1_sed_eval(pred_events, gold_events, class_aware=False),
        "event_f1_class": event_f1_sed_eval(pred_events, gold_events, class_aware=True),
    }


def aggregate(clip_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool onset errors / PR across clips."""
    all_errs: list[float] = []
    tp = fp = fn = 0
    for s in clip_summaries:
        all_errs.extend(s.get("onset", {}).get("errors_ms") or [])
        pr = s.get("pr50") or {}
        tp += int(pr.get("tp") or 0)
        fp += int(pr.get("fp") or 0)
        fn += int(pr.get("fn") or 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    def pct(tol: float) -> float:
        if not all_errs:
            return 0.0
        return 100.0 * sum(1 for e in all_errs if e <= tol) / len(all_errs)

    return {
        "n_clips": len(clip_summaries),
        "n_matched_onsets": len(all_errs),
        "median_onset_err_ms": float(np.median(all_errs)) if all_errs else None,
        "p90_onset_err_ms": float(np.percentile(all_errs, 90)) if all_errs else None,
        "pct_within_20": pct(20),
        "pct_within_50": pct(50),
        "pct_within_100": pct(100),
        "pct_within_200": pct(200),
        "precision_50": prec,
        "recall_50": rec,
        "f1_50": f1,
    }
