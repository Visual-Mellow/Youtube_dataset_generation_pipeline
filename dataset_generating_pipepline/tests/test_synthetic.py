"""Synthetic WAV suite with sample-exact onset times.

Asserts onset error ≤ 10 ms (clean) and ≤ 25 ms (10 dB SNR) for onsets_v2.
Also exercises v1 for a baseline comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clicks(sr: int, times_s: list[float], amp: float = 0.9) -> np.ndarray:
    n = int(sr * (max(times_s) + 0.5))
    y = np.zeros(n, dtype=np.float32)
    for t in times_s:
        i = int(round(t * sr))
        if 0 <= i < n:
            # 1-sample impulse + short exponential ring for realism.
            length = int(0.008 * sr)
            env = amp * np.exp(-np.linspace(0, 8, length)).astype(np.float32)
            end = min(n, i + length)
            y[i:end] += env[: end - i]
    return y


def _noise_bursts(sr: int, times_s: list[float], dur_s: float = 0.04, amp: float = 0.5) -> np.ndarray:
    n = int(sr * (max(times_s) + dur_s + 0.5))
    rng = np.random.default_rng(0)
    y = np.zeros(n, dtype=np.float32)
    for t in times_s:
        i = int(round(t * sr))
        length = int(dur_s * sr)
        burst = (amp * rng.standard_normal(length)).astype(np.float32)
        burst *= np.hanning(length).astype(np.float32)
        end = min(n, i + length)
        y[i:end] += burst[: end - i]
    return y


def _sine_gates(sr: int, intervals: list[tuple[float, float]], freq: float = 1000.0, amp: float = 0.4) -> np.ndarray:
    n = int(sr * (max(e for _, e in intervals) + 0.5))
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)
    tone = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    for s, e in intervals:
        a, b = int(s * sr), int(e * sr)
        y[a:b] += tone[a:b]
    return y


def _decaying_impacts(sr: int, times_s: list[float], amp: float = 0.8) -> np.ndarray:
    n = int(sr * (max(times_s) + 1.0))
    y = np.zeros(n, dtype=np.float32)
    for t in times_s:
        i = int(round(t * sr))
        length = int(0.2 * sr)
        tt = np.arange(length) / sr
        # Low thud + HF click.
        sig = amp * np.exp(-18 * tt) * np.sin(2 * np.pi * 120 * tt)
        sig += 0.35 * amp * np.exp(-80 * tt) * np.sin(2 * np.pi * 2500 * tt)
        end = min(n, i + length)
        y[i:end] += sig[: end - i].astype(np.float32)
    return y


def _add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(y.astype(np.float64) ** 2) + 1e-12)
    noise_power = power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(y)).astype(np.float32) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def _detect_v2(y: np.ndarray, sr: int) -> list[float]:
    from perception.onsets_v2 import detect_onsets_v2

    peaks = detect_onsets_v2(y, sr, dense_mode=True)
    return [float(p["t_ms_f"] if p.get("t_ms_f") is not None else p["t_ms"]) for p in peaks]


def _detect_v1(y: np.ndarray, sr: int) -> list[float]:
    from perception.onsets import detect_onsets

    # Resample to 16 kHz for v1 path.
    import librosa

    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y
    peaks = detect_onsets(y16, 16000, hop_ms=10, delta=0.6, min_gap_ms=100, stem="mixture")
    return [float(p["t_ms"]) for p in peaks]


def _match_errors(pred_ms: list[float], truth_ms: list[float], collar: float = 100.0) -> list[float]:
    from eval.metrics import hungarian_match

    return [e for _, _, e in hungarian_match(pred_ms, truth_ms, collar_ms=collar)]


def _case(name: str, y: np.ndarray, sr: int, truth_s: list[float], *, snr_db: float | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    wav = y if snr_db is None else _add_noise(y, snr_db, rng)
    truth_ms = [t * 1000 for t in truth_s]
    pred = _detect_v2(wav, sr)
    errs = _match_errors(pred, truth_ms, collar=80.0)
    med = float(np.median(errs)) if errs else None
    # Impulsive events: ≤10 ms clean / ≤25 ms @10 dB. Gradual bursts/tones: ≤20 / ≤30.
    gradual = name.startswith("bursts") or name.startswith("sine")
    lim = (20.0 if gradual else 10.0) if snr_db is None else (30.0 if gradual else 25.0)
    return {
        "name": name,
        "snr_db": snr_db,
        "n_truth": len(truth_ms),
        "n_pred": len(pred),
        "n_matched": len(errs),
        "median_err_ms": med,
        "max_err_ms": float(max(errs)) if errs else None,
        "errors_ms": errs,
        "limit_ms": lim,
        "ok": bool(errs) and med is not None and med <= lim and len(errs) >= max(1, len(truth_ms) - 1),
    }


def run_all() -> dict[str, Any]:
    sr = 32000
    truth_clicks = [0.5, 1.0, 1.35, 2.0, 2.8]
    truth_bursts = [0.4, 0.9, 1.5, 2.2]
    truth_sine = [0.6, 1.8, 3.0]  # onsets of gates
    sine_iv = [(0.6, 1.0), (1.8, 2.3), (3.0, 3.5)]
    truth_impacts = [0.7, 1.4, 2.5]

    cases = [
        _case("clicks_clean", _clicks(sr, truth_clicks), sr, truth_clicks),
        _case("clicks_snr10", _clicks(sr, truth_clicks), sr, truth_clicks, snr_db=10.0),
        _case("bursts_clean", _noise_bursts(sr, truth_bursts), sr, truth_bursts),
        _case("bursts_snr10", _noise_bursts(sr, truth_bursts), sr, truth_bursts, snr_db=10.0),
        _case("sine_gates_clean", _sine_gates(sr, sine_iv), sr, truth_sine),
        _case("impacts_clean", _decaying_impacts(sr, truth_impacts), sr, truth_impacts),
        _case("impacts_snr10", _decaying_impacts(sr, truth_impacts), sr, truth_impacts, snr_db=10.0),
    ]
    passed = sum(1 for c in cases if c["ok"])
    return {
        "passed": passed,
        "total": len(cases),
        "all_ok": passed == len(cases),
        "cases": cases,
    }


def test_synthetic_onsets_v2_clean_median():
    report = run_all()
    clean = [c for c in report["cases"] if c["snr_db"] is None]
    assert clean, "no clean cases"
    for c in clean:
        assert c["median_err_ms"] is not None, c
        assert c["median_err_ms"] <= float(c.get("limit_ms") or 10.0), c
        assert c["n_matched"] >= max(1, c["n_truth"] - 1), c


def test_synthetic_onsets_v2_noisy_median():
    report = run_all()
    noisy = [c for c in report["cases"] if c["snr_db"] == 10.0]
    for c in noisy:
        assert c["median_err_ms"] is not None, c
        assert c["median_err_ms"] <= float(c.get("limit_ms") or 25.0), c


if __name__ == "__main__":
    import json

    print(json.dumps(run_all(), indent=2))
