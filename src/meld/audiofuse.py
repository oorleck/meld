"""Fuse the aligned clips into one audio track.

Phone mics sit at different distances from the stage, so summing them causes comb filtering.
Instead every 0.5 s block of every clip gets a quality score (clipping, level, how much its
spectrum agrees with the other clips) and the output crossfades towards the best sources.
"""
from __future__ import annotations

import warnings
import wave

import numpy as np
from scipy.ndimage import uniform_filter1d

from .media import load_audio
from .project import Project
from .timeline import OUT_FPS, Timeline

SR = 48000
AR = 16000
BLOCK = 0.5
N_BANDS = 20
EDGE_FADE = int(0.05 * SR)
CHUNK = 20 * SR
PARTIAL_FLOOR = 0.02  # score for blocks a clip only partly covers, so it still plays if alone

_L16 = int(BLOCK * AR)
_L48 = int(BLOCK * SR)
_EDGES = np.unique(np.round(np.geomspace(100, 7500, N_BANDS + 1) / (AR / _L16)).astype(int))
_WIN = np.hanning(_L16).astype(np.float32)


def _block_features(x16, x48, offset: float):
    """Features for every 0.5 s block fully covered by the clip: (first_block, rms_db, spectral_shape, clip_fraction)."""
    b0 = int(np.ceil(offset / BLOCK - 1e-9))
    s16 = int(round((b0 * BLOCK - offset) * AR))
    s48 = int(round((b0 * BLOCK - offset) * SR))
    n = min((len(x16) - s16) // _L16, (len(x48) - s48) // _L48)
    if n <= 0:
        return b0, None
    rms = np.empty(n, np.float32)
    shape = np.empty((n, len(_EDGES) - 1), np.float32)
    clip = np.empty(n, np.float32)
    for i in range(0, n, 256):
        j = min(n, i + 256)
        seg = np.asarray(x16[s16 + i * _L16: s16 + j * _L16]).reshape(j - i, _L16)
        rms[i:j] = 10 * np.log10((seg ** 2).mean(axis=1) + 1e-12)
        power = np.abs(np.fft.rfft(seg * _WIN, axis=1)) ** 2
        bands = np.add.reduceat(power[:, _EDGES[0]:_EDGES[-1]], _EDGES[:-1] - _EDGES[0], axis=1)
        db = 10 * np.log10(bands + 1e-12)
        shape[i:j] = db - db.mean(axis=1, keepdims=True)
        seg48 = np.asarray(x48[s48 + i * _L48: s48 + j * _L48]).reshape(j - i, _L48)
        clip[i:j] = (np.abs(seg48) > 0.98).mean(axis=1)
    return b0, (rms, shape, clip)


def compute_scores(entries, x16s, x48s, nblocks: int):
    C = len(entries)
    rms = np.full((C, nblocks), np.nan, np.float32)
    clipf = np.zeros((C, nblocks), np.float32)
    shape = np.full((C, nblocks, len(_EDGES) - 1), np.nan, np.float32)
    cover = np.zeros((C, nblocks), bool)
    for c, e in enumerate(entries):
        cover[c, int(np.floor(e.offset / BLOCK + 1e-9)): int(np.ceil(e.end / BLOCK - 1e-9))] = True
        b0, feats = _block_features(x16s[c], x48s[c], e.offset)
        if feats is not None:
            r, s, k = feats
            rms[c, b0:b0 + len(r)] = r
            shape[c, b0:b0 + len(r)] = s
            clipf[c, b0:b0 + len(r)] = k

    valid = ~np.isnan(rms)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        median_shape = np.nanmedian(shape, axis=0)
        dist = np.sqrt(np.nanmean((shape - median_shape[None]) ** 2, axis=2))
        consensus = np.where(valid.sum(axis=0)[None] >= 3, np.exp(-(dist / 6.0) ** 2), 1.0)
        level = 1 / (1 + np.exp(-(np.nan_to_num(rms, nan=-100.0) + 50) / 3))
        score = np.where(valid, level * np.exp(-clipf * 40) * np.nan_to_num(consensus, nan=1.0), 0.0)
    score = np.where(cover & (score < PARTIAL_FLOOR), PARTIAL_FLOOR, score)
    score = uniform_filter1d(score, 5, axis=1, mode="nearest")  # ~2.5 s smoothing avoids mic flip-flopping
    return np.where(cover, score, 0.0), rms


def _loudness_gains(rms) -> np.ndarray:
    med = np.array([np.median(r[~np.isnan(r) & (r > -50)]) if np.any(~np.isnan(r) & (r > -50)) else np.nan for r in rms])
    target = np.nanmedian(med) if not np.all(np.isnan(med)) else 0.0
    gain_db = np.clip(np.nan_to_num(target - med, nan=0.0), -10, 10)
    return (10 ** (gain_db / 20)).astype(np.float32)


def fuse_audio(project: Project, power: float = 4.0, log=print):
    tl = Timeline.load(project.timeline_path)
    entries = tl.clips
    if not entries:
        raise SystemExit("Timeline is empty; run sync first.")
    x16s = [load_audio(project.clip_path(e), project.cache_dir, AR) for e in entries]
    x48s = [load_audio(project.clip_path(e), project.cache_dir, SR) for e in entries]
    offs = [int(round(e.offset * SR)) for e in entries]

    nblocks = int(np.ceil(tl.end / BLOCK)) + 1
    log("Scoring audio quality ...")
    score, rms = compute_scores(entries, x16s, x48s, nblocks)
    gains = _loudness_gains(rms)
    weights = score ** power
    centers = (np.arange(nblocks) + 0.5) * BLOCK

    segs = tl.frame_segments()
    total = sum(nf for _, nf in segs) * (SR // OUT_FPS)
    mix_path = project.cache_dir / "mix48.f32"
    out = np.memmap(mix_path, dtype="<f4", mode="w+", shape=(total,))

    log("Mixing ...")
    pos = 0
    for t0, nf in segs:
        n = nf * (SR // OUT_FPS)
        g_start = int(round(t0 * SR))
        for s in range(0, n, CHUNK):
            e_ = min(n, s + CHUNK)
            g0 = g_start + s
            g1 = g_start + e_
            times = np.arange(g0, g1) / SR
            num = np.zeros(e_ - s, np.float32)
            den = np.zeros(e_ - s, np.float32)
            for c, x in enumerate(x48s):
                lo = max(g0, offs[c])
                hi = min(g1, offs[c] + len(x))
                if hi <= lo:
                    continue
                idx = np.arange(lo - offs[c], hi - offs[c])
                edge = np.minimum(np.minimum(idx, len(x) - 1 - idx) / EDGE_FADE, 1.0).astype(np.float32)
                w = np.interp(times[lo - g0: hi - g0], centers, weights[c]).astype(np.float32) * edge
                num[lo - g0: hi - g0] += w * gains[c] * np.asarray(x[lo - offs[c]: hi - offs[c]])
                den[lo - g0: hi - g0] += w
            out[pos + s: pos + e_] = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
        pos += n

    peak = float(np.abs(out).max()) if total else 0.0
    scale = 0.89 / peak if peak > 0 else 1.0
    wav_path = project.out_dir / "fused_audio.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        for i in range(0, total, CHUNK):
            w.writeframes((np.clip(out[i:i + CHUNK] * scale, -1, 1) * 32767).astype("<i2").tobytes())
    del out

    top = np.argmax(weights, axis=0)
    live = weights.sum(axis=0) > 0
    log(f"Fused audio: {total / SR:.1f}s -> {wav_path}")
    for c, e in enumerate(entries):
        share = float(np.mean(top[live] == c)) if live.any() else 0.0
        log(f"  {share:5.1%} of the time led by {e.file}")
    return wav_path
