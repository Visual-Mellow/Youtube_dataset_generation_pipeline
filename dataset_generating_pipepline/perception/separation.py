"""Demucs source separation (HIGH level only).

Uses ``htdemucs`` to produce vocals / drums / bass / other stems. We map:
  - dialogue  ≈ vocals
  - music     ≈ drums + bass + other  (music-oriented; imperfect for SFX)
  - effects   ≈ mixture − vocals − drums   (proxy: residual after removing
                speech and percussive music energy)

Onsets re-run on the effects stem become the primary SFX clock. If Demucs
is not installed or fails, we return ``None`` and the caller falls back to
beat-grid filtering with a logged warning.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("visualmellow.perception.separation")


def separate_stems(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> dict[str, np.ndarray] | None:
    """Return ``{dialogue, music, effects, mixture}`` float32 arrays, or None.

    Demucs natively works at 44.1 kHz; we resample up for separation and
    back down so the rest of the pipeline stays at ``sample_rate``.
    """
    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        LOG.warning("Demucs not available (%s); skipping separation", exc)
        return None

    try:
        model = get_model("htdemucs")
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Upsample to Demucs sample rate if needed.
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, T)
        if sample_rate != model.samplerate:
            wav = torchaudio.functional.resample(wav, sample_rate, model.samplerate)
        # Demucs expects (batch, channels, time); duplicate mono → stereo.
        wav = wav.unsqueeze(0).repeat(1, 2, 1).to(device)

        with torch.inference_mode():
            sources = apply_model(model, wav, device=device)  # (1, S, C, T)

        # sources order for htdemucs: drums, bass, other, vocals
        names = list(model.sources)
        stems_44k = {}
        for i, name in enumerate(names):
            # Average stereo → mono.
            mono = sources[0, i].mean(0).cpu()
            if sample_rate != model.samplerate:
                mono = torchaudio.functional.resample(mono.unsqueeze(0), model.samplerate, sample_rate)[0]
            stems_44k[name] = mono.numpy().astype(np.float32)

        mixture = audio.astype(np.float32)
        # Align lengths (resampling can drift by a few samples).
        n = min(len(mixture), *(len(v) for v in stems_44k.values()))
        mixture = mixture[:n]
        for k in list(stems_44k):
            stems_44k[k] = stems_44k[k][:n]

        dialogue = stems_44k.get("vocals", np.zeros(n, dtype=np.float32))
        drums = stems_44k.get("drums", np.zeros(n, dtype=np.float32))
        bass = stems_44k.get("bass", np.zeros(n, dtype=np.float32))
        other = stems_44k.get("other", np.zeros(n, dtype=np.float32))
        music = drums + bass + other
        # Effects proxy: residual after removing vocals and drums (keeps
        # non-percussive SFX that Demucs dumped into "other", and anything
        # Demucs missed). Clamp lightly to avoid numerical blow-ups.
        effects = mixture - dialogue - drums
        effects = np.clip(effects, -1.0, 1.0).astype(np.float32)

        LOG.info(
            "Demucs separation OK: dialogue=%.1fs music=%.1fs effects=%.1fs",
            len(dialogue) / sample_rate,
            len(music) / sample_rate,
            len(effects) / sample_rate,
        )
        return {
            "mixture": mixture,
            "dialogue": dialogue,
            "music": music,
            "effects": effects,
            "drums": drums,
            "bass": bass,
            "other": other,
            "vocals": dialogue,
        }
    except Exception as exc:
        LOG.warning("Demucs separation failed (%s); falling back to mixture-only", exc)
        return None


def save_stems(stems: dict[str, np.ndarray], out_dir: Path, sample_rate: int = 16000) -> dict[str, str]:
    """Write stem WAVs for debugging; return {name: relative path}."""
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name in ("dialogue", "music", "effects", "mixture"):
        if name not in stems:
            continue
        path = out_dir / f"{name}.wav"
        sf.write(str(path), stems[name], sample_rate)
        paths[name] = str(path)
    return paths
