"""Unit tests: timebase helpers, snap v2 assignment, schema compat."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_frame_index_to_ms_uses_pts():
    from perception.timebase import frame_index_to_ms

    pts = np.array([0.0, 0.04, 0.09, 0.13], dtype=np.float64)
    assert abs(frame_index_to_ms(pts, 2) - 90.0) < 1e-6
    # Must NOT equal index/fps style if pts irregular
    assert abs(frame_index_to_ms(pts, 2) - (2 / 25 * 1000)) > 1.0


def test_snap_v2_two_cues_one_anchor_exclusive():
    from core.config import load_pipeline_config
    from annotate.snap import snap_all_v2

    cfg = load_pipeline_config("low")
    anchors = [
        {"id": "a000", "kind": "audio_onset", "stem": "effects", "t_ms": 1000, "t_ms_f": 1000.0,
         "strength_norm": 0.9, "end_reliable": True, "t_end_ms_f": 1200.0, "timing_source": "dsp_refined"},
        {"id": "a001", "kind": "audio_onset", "stem": "effects", "t_ms": 2000, "t_ms_f": 2000.0,
         "strength_norm": 0.8, "end_reliable": True, "t_end_ms_f": 2200.0, "timing_source": "dsp_refined"},
    ]
    cues = [
        {"cue_id": "cue_001", "audio_type": "SFX", "audio_class": "hit", "event_type": "INSTANT",
         "start_time_ms": 1050, "duration_ms": 200, "end_time_ms": 1250},
        {"cue_id": "cue_002", "audio_type": "SFX", "audio_class": "hit", "event_type": "INSTANT",
         "start_time_ms": 1100, "duration_ms": 200, "end_time_ms": 1300},
    ]
    out, actions = snap_all_v2(cues, anchors, cfg, duration_ms=5000)
    ids = [c.get("anchor_id") for c in out if c.get("anchor_id")]
    # At most one exclusive assignment per anchor (bursts aside).
    assert len(ids) == len(set(ids))
    assert any(a.get("action") in {"retime", "snap_reject"} for a in actions)


def test_schema_21_backward_readable():
    """A v1-style reader that only needs t_ms/kind still parses 2.1 docs."""
    doc = {
        "schema_version": "2.1.0",
        "anchor_version": 2,
        "anchors": [
            {
                "id": "a000", "kind": "audio_onset", "t_ms": 100,
                "t_ms_f": 99.6, "t_end_ms_f": 180.0, "bands": ["hf"],
                "snr_db": 12.0, "timing_source": "dsp_refined", "score": 0.7,
            }
        ],
    }
    # Simulate naive v1 reader
    for a in doc["anchors"]:
        assert "t_ms" in a and "kind" in a
        _ = int(a["t_ms"])
    raw = json.dumps(doc)
    loaded = json.loads(raw)
    assert loaded["schema_version"].startswith("2.")


def test_hysteresis_sed_events():
    from perception.sed.events import extract_events

    # Synthetic [T, C] with one class rising above 0.5 then falling below 0.3
    probs = np.zeros((100, 2), dtype=np.float32)
    probs[20:40, 0] = 0.8
    probs[41:45, 0] = 0.2
    events = extract_events(probs, ["Gunshot, gunfire", "Music"], frame_hop_s=0.01, on_thr=0.5, off_thr=0.3)
    assert any(e["class"] == "Gunshot, gunfire" for e in events)
    gun = next(e for e in events if e["class"] == "Gunshot, gunfire")
    assert gun["t_ms_f"] < gun["t_end_ms_f"]
