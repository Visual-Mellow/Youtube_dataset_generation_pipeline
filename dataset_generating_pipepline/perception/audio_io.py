"""Decode video audio to mono float32 PCM via ffmpeg.

We always re-sample to the perception sample rate (default 16 kHz) so every
downstream stage (onsets, beats, tagger, ASR) sees a consistent clock.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

LOG = logging.getLogger("visualmellow.perception.audio_io")


def decode_mono(path: Path | str, sample_rate: int = 16000) -> np.ndarray:
    """Return mono float32 samples at ``sample_rate`` from a media file.

    Uses ffmpeg's ``f32le`` raw PCM pipe so we never write an intermediate WAV.
    Raises ``RuntimeError`` if ffmpeg fails or returns no samples.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio source not found: {path}")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio decode failed for {path}: {(proc.stderr or b'')[-500:].decode('utf-8', 'replace')}"
        )
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg returned empty audio for {path}")
    LOG.debug("Decoded %s -> %d samples (%.2fs @ %d Hz)", path.name, audio.size, audio.size / sample_rate, sample_rate)
    return audio


def decode_dsp(path: Path | str, sample_rate: int = 32000) -> np.ndarray:
    """Decode mono float32 at DSP rate (default 32 kHz) for onset refinement.

    High-frequency transients are clipped by 16 kHz Nyquist; keep model audio
    at 16 kHz via :func:`decode_mono` and use this stream only for DSP.
    """
    return decode_mono(path, sample_rate=sample_rate)


def extract_audio_bytes(
    path: Path | str,
    *,
    start_ms: int = 0,
    duration_ms: int | None = None,
    sample_rate: int = 16000,
    mime: str = "audio/wav",
) -> bytes:
    """Extract a WAV (or other container) byte blob for Gemini audio-only passes.

    ``start_ms`` / ``duration_ms`` select a sub-window; omit duration for the
    rest of the clip. Returns raw file bytes ready for ``query_gemini_multimodal``.
    """
    path = Path(path)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_ms > 0:
        cmd += ["-ss", f"{start_ms / 1000:.3f}"]
    cmd += ["-i", str(path)]
    if duration_ms is not None:
        cmd += ["-t", f"{duration_ms / 1000:.3f}"]
    # Stereo→mono, fixed rate, 16-bit PCM WAV is universally accepted by Gemini.
    fmt = "wav" if "wav" in mime else "mp3"
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", fmt, "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg audio extract failed: {(proc.stderr or b'')[-500:].decode('utf-8', 'replace')}"
        )
    return proc.stdout


def extract_video_bytes(
    path: Path | str,
    *,
    start_ms: int = 0,
    duration_ms: int | None = None,
) -> bytes:
    """Cut a sub-clip to an in-memory MP4 for Gemini window proposers."""
    path = Path(path)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_ms > 0:
        cmd += ["-ss", f"{start_ms / 1000:.3f}"]
    cmd += ["-i", str(path)]
    if duration_ms is not None:
        cmd += ["-t", f"{duration_ms / 1000:.3f}"]
    cmd += [
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        "-f", "mp4", "pipe:1",
    ]
    # MP4 in a pipe needs fragmented/empty_moov; use a temp-file fallback via stdout
    # is unreliable for MP4. Write to a NamedTemporaryFile instead.
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        file_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if start_ms > 0:
            file_cmd += ["-ss", f"{start_ms / 1000:.3f}"]
        file_cmd += ["-i", str(path)]
        if duration_ms is not None:
            file_cmd += ["-t", f"{duration_ms / 1000:.3f}"]
        file_cmd += [
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart",
            tmp_path,
        ]
        proc = subprocess.run(file_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg video extract failed: {(proc.stderr or b'')[-500:].decode('utf-8', 'replace')}"
            )
        return Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
