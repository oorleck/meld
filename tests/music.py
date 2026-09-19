"""Synthetic 'concert' music with a drum kit, for testing beat tracking and cutting to the music.

Unlike synth.py's random notes, this has what beat tracking looks at: a kick, a snare and hi-hats playing a groove at a
tempo that can change, sections that are quiet, medium or hard, and the true beat times to compare the results with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 22050


@dataclass
class Section:
    bpm: float
    bars: int
    style: str = "medium"  # soft (a ballad: sparse, quiet), medium, hard (dense, loud), or none (no rhythm: a drone)


@dataclass
class Music:
    audio: np.ndarray
    beats: np.ndarray  # the true beat times
    downbeats: np.ndarray  # the true first beat of each bar
    sections: list[tuple[float, float, Section]] = field(default_factory=list)  # (start, end, section)


def _kick(sr: int) -> np.ndarray:
    t = np.arange(int(0.18 * sr)) / sr
    phase = 2 * np.pi * np.cumsum(45 + 90 * np.exp(-t * 28)) / sr
    return np.sin(phase) * np.exp(-t * 22)


def _snare(sr: int, rng) -> np.ndarray:
    t = np.arange(int(0.16 * sr)) / sr
    return (0.7 * rng.standard_normal(len(t)) * np.exp(-t * 30) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 35))


def _hat(sr: int, rng) -> np.ndarray:
    t = np.arange(int(0.05 * sr)) / sr
    noise = rng.standard_normal(len(t))
    return np.diff(noise, prepend=0.0) * np.exp(-t * 110)  # a rough high-pass


def _bass_note(sr: int, freq: float) -> np.ndarray:
    t = np.arange(int(0.6 * sr)) / sr
    return (np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * 2 * freq * t)) * np.exp(-t * 4.0)


def _crash(sr: int, rng) -> np.ndarray:
    t = np.arange(int(1.2 * sr)) / sr
    return np.diff(rng.standard_normal(len(t)), prepend=0.0) * np.exp(-t * 3.0)


def _add(out: np.ndarray, hit: np.ndarray, at: float, gain: float, sr: int, rng, jitter: float) -> None:
    i = int((at + rng.normal(0, jitter)) * sr)
    if 0 <= i < len(out):
        j = min(len(out), i + len(hit))
        out[i:j] += gain * hit[: j - i]


def render(sections: list[Section], seed: int = 0, sr: int = SR, jitter: float = 0.006) -> Music:
    """Play the sections one after another. `jitter` is how far, in seconds, a live drummer is from the grid."""
    rng = np.random.default_rng(seed)
    kick, snare, hat = _kick(sr), _snare(sr, rng), _hat(sr, rng)
    total = sum(s.bars * 4 * 60.0 / s.bpm for s in sections)
    out = np.zeros(int(total * sr) + sr, np.float32)
    beats, downbeats, spans = [], [], []
    t = 0.0
    for sec in sections:
        beat = 60.0 / sec.bpm
        start = t
        for bar in range(sec.bars):
            downbeats.append(t)
            root = 55.0 * 2 ** (((bar // 2) % 4) * 5 / 12)  # a bass line that changes every two bars
            if sec.style != "none":
                # what real music does on beat one: the bass plays a note, and at the start of a phrase there is a crash
                _add(out, _bass_note(sr, root), t, {"soft": 0.35, "medium": 0.5, "hard": 0.6}[sec.style], sr, rng, jitter)
                if sec.style != "soft" and bar % 4 == 0:
                    _add(out, _crash(sr, rng), t, 0.5, sr, rng, jitter)
            for k in range(4):
                at = t + k * beat
                beats.append(at)
                if sec.style == "none":
                    continue
                loud = {"soft": 0.35, "medium": 0.7, "hard": 1.0}[sec.style]
                if sec.style == "soft":
                    if k == 0:
                        _add(out, kick, at, loud, sr, rng, jitter)
                    if k == 2:
                        _add(out, snare, at, loud * 0.4, sr, rng, jitter)
                    _add(out, hat, at, loud * 0.25, sr, rng, jitter)
                elif sec.style == "medium":
                    if k in (0, 2):
                        _add(out, kick, at, loud, sr, rng, jitter)
                    if k in (1, 3):
                        _add(out, snare, at, loud * 0.8, sr, rng, jitter)
                    for sub in (0, 0.5):
                        _add(out, hat, at + sub * beat, loud * 0.3, sr, rng, jitter)
                else:  # hard
                    _add(out, kick, at, loud * (1.0 if k == 0 else 0.8), sr, rng, jitter)
                    _add(out, kick, at + 0.5 * beat, loud * 0.6, sr, rng, jitter)
                    if k in (1, 3):
                        _add(out, snare, at, loud, sr, rng, jitter)
                    for sub in (0, 0.25, 0.5, 0.75):
                        _add(out, hat, at + sub * beat, loud * 0.35, sr, rng, jitter)
            t += 4 * beat
        # a bass line and a pad under the whole section, so the sound is not only drums
        i0, i1 = int(start * sr), min(len(out), int(t * sr))
        tt = np.arange(i1 - i0) / sr
        amp = {"soft": 0.12, "medium": 0.2, "hard": 0.32, "none": 0.3}[sec.style]
        bass_root = 55.0 * (1 + 0.0 * tt)
        tone = np.sin(2 * np.pi * bass_root * tt) + 0.5 * np.sin(2 * np.pi * 2 * bass_root * tt)
        pad = np.sin(2 * np.pi * 220 * tt) + np.sin(2 * np.pi * 277 * tt) + np.sin(2 * np.pi * 330 * tt)
        if sec.style == "none":
            out[i0:i1] += amp * 0.4 * pad.astype(np.float32) + 0.05 * rng.standard_normal(i1 - i0).astype(np.float32)
        else:
            out[i0:i1] += amp * 0.5 * tone.astype(np.float32) + amp * 0.15 * pad.astype(np.float32)
        spans.append((start, t, sec))
    out += 0.01 * rng.standard_normal(len(out)).astype(np.float32)  # room noise
    peak = float(np.abs(out).max())
    return Music((out / peak * 0.9).astype(np.float32), np.array(beats), np.array(downbeats), spans)


def beats_on_truth(found: np.ndarray, truth: np.ndarray, tol: float = 0.04) -> float:
    """The share of found beats that are on a true beat (within `tol` seconds): are the beats where real ones are? Finding
    every other beat, which is half the tempo, is fine by this measure."""
    if len(found) == 0 or len(truth) == 0:
        return 0.0
    return float((np.abs(found[:, None] - truth[None, :]).min(axis=1) <= tol).mean())


def beats_found_near(found: np.ndarray, truth: np.ndarray, tol: float = 0.04, after: float = 0.0) -> float:
    """The share of true beats (from `after` on) that have a found beat within `tol` seconds: is every beat found?"""
    truth = truth[truth >= after]
    if len(truth) == 0 or len(found) == 0:
        return 0.0
    nearest = np.abs(found[None, :] - truth[:, None]).min(axis=1)
    return float((nearest <= tol).mean())
