"""Multi-band SuperFlux onsets + waveform refinement + offsets (Layer-0 v2).

Designed for ~32 kHz DSP audio. Peak-picks the *start of the rise* via Hilbert
envelope backtracking, not the spectral-flux maximum. Keep ``onsets.py`` for
``anchor_version=1`` A/B.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

LOG = logging.getLogger("visualmellow.perception.onsets_v2")

DEFAULT_BANDS_HZ = ((0, 500), (500, 2000), (2000, 8000), (8000, 16000))
BAND_NAMES = ("lf", "lowmid", "mid", "hf")


def _stft_mag(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = 1024,
    hop: int = 160,
    win_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs_hz, magnitude spectrogram [F, T])."""
    import librosa

    window = np.hanning(win_length).astype(np.float32)
    # Center pad so frames align predictably.
    S = np.abs(
        librosa.stft(
            audio.astype(np.float32),
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            window=window,
            center=True,
        )
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, S.astype(np.float64)


def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs < hi)


def superflux_band(
    S_band: np.ndarray,
    *,
    lag: int = 2,
    max_filter: int = 3,
) -> np.ndarray:
    """SuperFlux-style positive spectral difference on a sub-band spectrogram.

    ``S_band`` shape [F, T] (linear magnitude). We log-compress, apply a
    frequency max-filter on the lagged frame, then sum positive diffs.
    """
    if S_band.size == 0 or S_band.shape[1] < lag + 2:
        return np.zeros(S_band.shape[1], dtype=np.float64)
    logS = np.log1p(np.maximum(S_band, 0.0) * 1000.0)
    # Max filter across frequency on previous frame (size max_filter).
    from numpy.lib.stride_tricks import sliding_window_view

    F, T = logS.shape
    if F >= max_filter:
        windows = sliding_window_view(logS, max_filter, axis=0)  # [F-mf+1, T, mf]
        filtered = windows.max(axis=-1)
        # Pad back to F
        pad = F - filtered.shape[0]
        left = pad // 2
        filtered = np.pad(filtered, ((left, pad - left), (0, 0)), mode="edge")
    else:
        filtered = logS

    env = np.zeros(T, dtype=np.float64)
    for t in range(lag, T):
        diff = logS[:, t] - filtered[:, t - lag]
        env[t] = float(np.sum(np.maximum(diff, 0.0)))
    return env


def _adaptive_peaks(
    env: np.ndarray,
    times_s: np.ndarray,
    *,
    mad_k: float = 3.5,
    median_win_frames: int = 120,
    min_gap_s: float = 0.035,
) -> list[dict[str, Any]]:
    """Peak-pick with median + MAD adaptive threshold."""
    if len(env) < 5:
        return []
    peaks: list[dict[str, Any]] = []
    last_t = -1e9
    half = max(3, median_win_frames // 2)
    for i in range(1, len(env) - 1):
        if env[i] < env[i - 1] or env[i] < env[i + 1]:
            continue
        lo = max(0, i - half)
        hi = min(len(env), i + half + 1)
        local = env[lo:hi]
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med))) + 1e-9
        thr = med + mad_k * mad
        if env[i] < thr:
            continue
        t = float(times_s[i])
        if t - last_t < min_gap_s:
            # Keep stronger.
            if peaks and env[i] > peaks[-1]["strength"]:
                peaks[-1] = {"t_s": t, "strength": float(env[i]), "frame": i}
                last_t = t
            continue
        peaks.append({"t_s": t, "strength": float(env[i]), "frame": i})
        last_t = t
    return peaks


def detect_candidates(
    audio: np.ndarray,
    sample_rate: int = 32000,
    *,
    n_fft: int = 1024,
    hop: int = 160,
    win_length: int = 512,
    bands_hz: Sequence[Sequence[float]] = DEFAULT_BANDS_HZ,
    mad_k: float = 3.5,
    median_win_ms: float = 600,
    min_gap_ms: float = 35,
    dense_min_gap_ms: float = 20,
    dense_mode: bool = False,
    cross_band_merge_ms: float = 25,
) -> list[dict[str, Any]]:
    """Multi-band SuperFlux candidates with cross-band merge."""
    freqs, S = _stft_mag(audio, sample_rate, n_fft=n_fft, hop=hop, win_length=win_length)
    times_s = np.arange(S.shape[1]) * hop / sample_rate
    median_win_frames = max(5, int(round(median_win_ms / 1000.0 * sample_rate / hop)))
    gap = (dense_min_gap_ms if dense_mode else min_gap_ms) / 1000.0

    per_band: list[list[dict[str, Any]]] = []
    for bi, (lo, hi) in enumerate(bands_hz):
        mask = _band_mask(freqs, float(lo), float(hi))
        if not np.any(mask):
            per_band.append([])
            continue
        env = superflux_band(S[mask])
        # Prefer denser gap on HF band.
        band_gap = gap if bi < 3 else min(gap, dense_min_gap_ms / 1000.0)
        peaks = _adaptive_peaks(
            env, times_s, mad_k=mad_k, median_win_frames=median_win_frames, min_gap_s=band_gap,
        )
        name = BAND_NAMES[bi] if bi < len(BAND_NAMES) else f"b{bi}"
        # Normalize strength per band.
        strengths = [p["strength"] for p in peaks] or [1.0]
        smax = max(strengths) or 1.0
        for p in peaks:
            p["band"] = name
            p["strength_norm"] = float(p["strength"] / smax)
        per_band.append(peaks)

    # Flatten and merge across bands.
    flat: list[dict[str, Any]] = []
    for peaks in per_band:
        flat.extend(peaks)
    flat.sort(key=lambda p: p["t_s"])

    merged: list[dict[str, Any]] = []
    merge_s = cross_band_merge_ms / 1000.0
    for p in flat:
        if merged and abs(p["t_s"] - merged[-1]["t_s"]) <= merge_s:
            cur = merged[-1]
            cur["bands"] = sorted(set(cur.get("bands") or []) | {p["band"]})
            if p["strength_norm"] > cur.get("strength_norm", 0):
                cur["strength"] = p["strength"]
                cur["strength_norm"] = p["strength_norm"]
                cur["t_s"] = p["t_s"]  # take strongest band's time as coarse
            continue
        merged.append({
            "t_s": p["t_s"],
            "strength": p["strength"],
            "strength_norm": p["strength_norm"],
            "bands": [p["band"]],
        })
    return merged


def _bandpass(audio: np.ndarray, sr: int, bands: list[str], bands_hz: Sequence[Sequence[float]]) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    # Union of selected band ranges.
    name_to_hz = {BAND_NAMES[i]: tuple(bands_hz[i]) for i in range(min(len(BAND_NAMES), len(bands_hz)))}
    ranges = [name_to_hz[b] for b in bands if b in name_to_hz]
    if not ranges:
        return audio.astype(np.float32)
    lo = min(r[0] for r in ranges)
    hi = max(r[1] for r in ranges)
    nyq = sr / 2.0
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    if hi_n <= lo_n:
        return audio.astype(np.float32)
    try:
        sos = butter(2, [lo_n, hi_n], btype="band", output="sos")
        return sosfiltfilt(sos, audio.astype(np.float64)).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def refine_onset(
    audio: np.ndarray,
    sample_rate: int,
    t_s: float,
    bands: list[str],
    *,
    bands_hz: Sequence[Sequence[float]] = DEFAULT_BANDS_HZ,
    pre_ms: float = 120,
    post_ms: float = 60,
    env_smooth_ms: float = 2,
    floor_db: float = 6,
    min_snr_db: float = 6,
) -> dict[str, Any]:
    """Hilbert-envelope backtrack to rise start; return refined timing fields."""
    from scipy.signal import hilbert

    n = len(audio)
    i0 = max(0, int(round((t_s - pre_ms / 1000.0) * sample_rate)))
    i1 = min(n, int(round((t_s + post_ms / 1000.0) * sample_rate)))
    if i1 - i0 < max(8, int(0.01 * sample_rate)):
        t_ms_f = t_s * 1000.0
        return {
            "t_ms_f": t_ms_f,
            "t_ms_coarse": t_s * 1000.0,
            "rise_time_ms": None,
            "peak_db": None,
            "snr_db": 0.0,
            "low_confidence": True,
        }

    seg = _bandpass(audio[i0:i1], sample_rate, bands, bands_hz)
    analytic = hilbert(seg.astype(np.float64))
    env = np.abs(analytic)
    smooth_n = max(1, int(round(env_smooth_ms / 1000.0 * sample_rate)))
    if smooth_n > 1:
        kernel = np.ones(smooth_n) / smooth_n
        env = np.convolve(env, kernel, mode="same")
    env_db = 20.0 * np.log10(env + 1e-10)

    # Noise floor from [t-300ms, t-100ms] relative to coarse, clipped to clip.
    floor_a = max(0, int(round((t_s - 0.300) * sample_rate)))
    floor_b = max(floor_a + 1, int(round((t_s - 0.100) * sample_rate)))
    floor_a = max(0, min(floor_a, n - 1))
    floor_b = max(floor_a + 1, min(floor_b, n))
    floor_seg = audio[floor_a:floor_b]
    if floor_seg.size > 16:
        fa = np.abs(hilbert(floor_seg.astype(np.float64)))
        floor = float(np.percentile(fa, 10))
        floor_db_val = 20.0 * np.log10(floor + 1e-10)
    else:
        floor_db_val = float(np.min(env_db))

    peak_i = int(np.argmax(env_db))
    peak_db = float(env_db[peak_i])
    # Guard against -inf floor on digital silence (would thrash rise-start to window edge).
    floor_db_val = max(floor_db_val, peak_db - 60.0)
    snr_db = peak_db - floor_db_val
    thr = floor_db_val + floor_db

    # Do not backtrack more than max_backtrack_ms before the coarse candidate / peak.
    max_backtrack = int(round(0.040 * sample_rate))  # 40 ms
    search_lo = max(0, peak_i - max_backtrack)

    # Last sample before peak where env < thr (within backtrack budget).
    rise_start = search_lo
    for i in range(peak_i, search_lo - 1, -1):
        if env_db[i] < thr:
            rise_start = i
            break
    # If we hit the search edge without crossing thr, use 10% rise instead of window start.
    # 10%-90% rise line intercept with floor.
    lo_lvl = floor_db_val + 0.1 * (peak_db - floor_db_val)
    hi_lvl = floor_db_val + 0.9 * (peak_db - floor_db_val)
    i10 = rise_start
    i90 = peak_i
    for i in range(rise_start, peak_i + 1):
        if env_db[i] >= lo_lvl:
            i10 = i
            break
    for i in range(peak_i, rise_start - 1, -1):
        if env_db[i] <= hi_lvl:
            i90 = max(i, i10 + 1)
            break
    if i90 > i10:
        xs = np.arange(i10, i90 + 1, dtype=np.float64)
        ys = env_db[i10 : i90 + 1]
        if len(xs) >= 2 and np.std(ys) > 1e-6:
            coef = np.polyfit(xs, ys, 1)
            if abs(coef[0]) > 1e-9:
                x_int = (floor_db_val - coef[1]) / coef[0]
                x_int = float(np.clip(x_int, search_lo, peak_i))
            else:
                x_int = float(i10)
        else:
            x_int = float(i10)
    else:
        x_int = float(max(rise_start, search_lo))

    refined_sample = i0 + x_int
    t_ms_f = refined_sample / sample_rate * 1000.0
    # Never move more than max_backtrack before the coarse SuperFlux time.
    coarse_ms = t_s * 1000.0
    if t_ms_f < coarse_ms - 40.0:
        t_ms_f = coarse_ms - min(40.0, max(0.0, coarse_ms - t_ms_f))
        # Prefer staying near peak onset for impulsive events.
        peak_ms = (i0 + peak_i) / sample_rate * 1000.0
        if abs(peak_ms - coarse_ms) <= 25.0:
            t_ms_f = min(t_ms_f, peak_ms)
            # For near-impulse: use peak time (click energy is at the sample).
            if (peak_i - x_int) <= max(2, int(0.002 * sample_rate)):
                t_ms_f = peak_ms
    # Blend: if refinement diverged > 30 ms from coarse, trust coarse (STFT peak).
    if abs(t_ms_f - coarse_ms) > 30.0:
        t_ms_f = coarse_ms

    rise_time_ms = max(0.0, (i0 + peak_i) / sample_rate * 1000.0 - t_ms_f)
    return {
        "t_ms_f": float(t_ms_f),
        "t_ms_coarse": float(coarse_ms),
        "rise_time_ms": float(rise_time_ms),
        "peak_db": peak_db,
        "snr_db": float(snr_db),
        "low_confidence": bool(snr_db < min_snr_db),
        "_peak_sample": int(i0 + peak_i),
        "_floor_db": floor_db_val,
    }


def estimate_offset(
    audio: np.ndarray,
    sample_rate: int,
    refined: dict[str, Any],
    *,
    next_onset_ms: float | None = None,
    decay_db_below_peak: float = 30,
    floor_db: float = 6,
    max_event_ms: float = 3000,
) -> dict[str, Any]:
    """Decay-based offset after peak."""
    from scipy.signal import hilbert

    t_ms = float(refined["t_ms_f"])
    peak_sample = int(refined.get("_peak_sample") or round(t_ms / 1000.0 * sample_rate))
    max_n = int(round(max_event_ms / 1000.0 * sample_rate))
    if next_onset_ms is not None:
        max_n = min(max_n, max(1, int(round((next_onset_ms - t_ms) / 1000.0 * sample_rate))))
    i1 = min(len(audio), peak_sample + max_n)
    if i1 - peak_sample < 4:
        return {
            "t_end_ms_f": t_ms + 30.0,
            "decay_ms": 30.0,
            "end_reliable": False,
        }

    seg = audio[peak_sample:i1].astype(np.float64)
    env = np.abs(hilbert(seg))
    smooth_n = max(1, int(round(0.002 * sample_rate)))
    env = np.convolve(env, np.ones(smooth_n) / smooth_n, mode="same")
    env_db = 20.0 * np.log10(env + 1e-10)
    peak_db = float(refined.get("peak_db") or env_db[0])
    floor = float(refined.get("_floor_db") or (peak_db - 40))
    thr = max(floor + floor_db, peak_db - decay_db_below_peak)

    end_i = len(env_db) - 1
    interrupted = False
    for i in range(1, len(env_db)):
        if env_db[i] < thr:
            end_i = i
            break
    else:
        interrupted = True

    if next_onset_ms is not None and (t_ms + end_i / sample_rate * 1000) >= next_onset_ms - 5:
        interrupted = True

    t_end = peak_sample / sample_rate * 1000.0 + end_i / sample_rate * 1000.0
    decay_ms = t_end - t_ms
    return {
        "t_end_ms_f": float(t_end),
        "decay_ms": float(decay_ms),
        "end_reliable": not interrupted and not refined.get("low_confidence"),
    }


def detect_onsets_v2(
    audio: np.ndarray,
    sample_rate: int = 32000,
    *,
    stem: str = "mixture",
    dense_mode: bool = False,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Full v2 pipeline: candidates → refine → offsets."""
    cfg = dict(cfg or {})
    refine_cfg = dict(cfg.get("refine") or {})
    offset_cfg = dict(cfg.get("offset") or {})
    bands_hz = cfg.get("bands_hz") or DEFAULT_BANDS_HZ

    cands = detect_candidates(
        audio,
        sample_rate,
        n_fft=int(cfg.get("n_fft", 1024)),
        hop=int(cfg.get("hop", 160)),
        win_length=int(cfg.get("win_length", 512)),
        bands_hz=bands_hz,
        mad_k=float(cfg.get("mad_k", 3.5)),
        median_win_ms=float(cfg.get("median_win_ms", 600)),
        min_gap_ms=float(cfg.get("min_gap_ms", 35)),
        dense_min_gap_ms=float(cfg.get("dense_min_gap_ms", 20)),
        dense_mode=dense_mode,
        cross_band_merge_ms=float(cfg.get("cross_band_merge_ms", 25)),
    )

    refined: list[dict[str, Any]] = []
    for c in cands:
        r = refine_onset(
            audio,
            sample_rate,
            c["t_s"],
            c.get("bands") or ["mid"],
            bands_hz=bands_hz,
            pre_ms=float(refine_cfg.get("pre_ms", 120)),
            post_ms=float(refine_cfg.get("post_ms", 60)),
            env_smooth_ms=float(refine_cfg.get("env_smooth_ms", 2)),
            floor_db=float(refine_cfg.get("floor_db", 6)),
            min_snr_db=float(refine_cfg.get("min_snr_db", 6)),
        )
        r["bands"] = c.get("bands") or []
        r["strength"] = float(c.get("strength") or 0)
        r["strength_norm"] = float(c.get("strength_norm") or 0)
        r["stem"] = stem
        r["kind"] = "audio_onset"
        r["timing_source"] = "dsp_refined"
        refined.append(r)

    refined.sort(key=lambda x: x["t_ms_f"])
    max_map = offset_cfg.get("max_event_ms") or {}
    default_max = float(max_map.get("default", 3000) if isinstance(max_map, dict) else 3000)
    impact_max = float(max_map.get("impact", 1500) if isinstance(max_map, dict) else 1500)

    out: list[dict[str, Any]] = []
    for i, r in enumerate(refined):
        next_ms = refined[i + 1]["t_ms_f"] if i + 1 < len(refined) else None
        is_hf = "hf" in (r.get("bands") or []) or "mid" in (r.get("bands") or [])
        max_ms = impact_max if is_hf else default_max
        off = estimate_offset(
            audio,
            sample_rate,
            r,
            next_onset_ms=next_ms,
            decay_db_below_peak=float(offset_cfg.get("decay_db_below_peak", 30)),
            floor_db=float(refine_cfg.get("floor_db", 6)),
            max_event_ms=max_ms,
        )
        item = {
            "t_ms": int(round(r["t_ms_f"])),
            "t_ms_f": round(float(r["t_ms_f"]), 1),
            "t_ms_coarse": round(float(r["t_ms_coarse"]), 1),
            "t_end_ms_f": round(float(off["t_end_ms_f"]), 1),
            "end_reliable": bool(off["end_reliable"]),
            "decay_ms": round(float(off["decay_ms"]), 1),
            "rise_time_ms": None if r.get("rise_time_ms") is None else round(float(r["rise_time_ms"]), 1),
            "peak_db": r.get("peak_db"),
            "snr_db": None if r.get("snr_db") is None else round(float(r["snr_db"]), 2),
            "low_confidence": bool(r.get("low_confidence")),
            "uncertainty_ms": 40.0 if r.get("low_confidence") else max(4.0, float(r.get("rise_time_ms") or 8.0)),
            "bands": r.get("bands") or [],
            "strength": r.get("strength"),
            "strength_norm": r.get("strength_norm"),
            "stem": stem,
            "kind": "audio_onset",
            "timing_source": "dsp_refined",
        }
        out.append(item)

    LOG.info("Onsets v2 (%s): %d candidates → %d refined @ %d Hz", stem, len(cands), len(out), sample_rate)
    return out
