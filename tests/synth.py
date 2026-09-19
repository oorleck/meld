"""Synthetic 'concert' material for tests."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from meld.media import run_ffmpeg

NOTES = [110, 146.8, 196, 220, 261.6, 329.6, 440, 523.3]


def synth_music(seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = np.zeros(n, np.float32)
    for k in range(int(seconds * 4)):
        # jittered onsets, detune and random phase: separate performances must not be phase-coherent
        start = int((k * 0.25 + rng.uniform(0, 0.12)) * sr)
        m = min(int(0.3 * sr), n - start)
        if m <= 0:
            break
        t = np.arange(m) / sr
        f = rng.choice(NOTES) * (1 + 0.01 * rng.standard_normal())
        phase = rng.uniform(0, 2 * np.pi, 5)
        x[start:start + m] += 0.3 * np.exp(-t * 8) * sum(
            np.sin(2 * np.pi * f * h * t + phase[h - 1]) / h for h in range(1, 6)
        )
        if k % 2 == 0:
            x[start:start + m] += 0.5 * rng.standard_normal(m) * np.exp(-t * 40)
    return x


def phone(music: np.ndarray, sr: int, t0: float, t1: float, seed: int, gain: float = 1.0, noise: float = 0.05):
    """What a phone in the crowd might record of music[t0:t1]: room smear, noise, clipping."""
    rng = np.random.default_rng(seed)
    seg = music[int(t0 * sr): int(t1 * sr)].astype(np.float64)
    n_ir = int(0.06 * sr)
    ir = rng.standard_normal(n_ir) * np.exp(-np.arange(n_ir) / (0.01 * sr))
    ir[0] = 3 * np.abs(ir).max()
    seg = fftconvolve(seg, ir)[: len(seg)]
    seg = seg / seg.std() * 0.15 * gain
    seg = seg + noise * rng.standard_normal(len(seg))
    return np.clip(seg, -0.98, 0.98).astype(np.float32)


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def make_clip(path: Path, audio: np.ndarray, sr: int, video_src: str, vf: str | None = None) -> None:
    wav = path.with_suffix(".tmp.wav")
    write_wav(wav, audio, sr)
    dur = len(audio) / sr
    args = ["-f", "lavfi", "-i", f"{video_src}:duration={dur}", "-i", wav]
    if vf:
        args += ["-vf", vf]
    run_ffmpeg([*args, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path])
    wav.unlink()
