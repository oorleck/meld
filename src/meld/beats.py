"""Find the beats, the bars and how intense the music is, to cut the video to it.

The method is the standard one for beat tracking (Ellis 2007, "Beat Tracking by Dynamic Programming"): an onset
strength curve (how much new sound starts, per moment), a tempo from its autocorrelation, and dynamic programming that
picks the beats: on strong onsets, and about one tempo period apart. Two things make it fit a concert:

- The tempo is followed through time, not taken once. A concert is several songs and a song drifts, so a single
  tempo would be wrong most of the time. The tempo is found window by window and smoothed so that it can change
  between songs but does not flip between a tempo and its double.
- Only what cutting needs is worked out: the beats, which of them start a bar (beat one is the strongest place to cut),
  and how intense the music is around each (loudness relative to the rest of the concert, and whether it is building).

Everything here works on audio alone (numpy and scipy), on one continuous stretch at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

SR = 22050  # the audio is analysed at this rate ...
HOP = 441  # ... every 20 ms
FPS = SR / HOP  # analysis frames per second: 50
N_FFT = 1024
BPM_MIN, BPM_MAX = 60.0, 200.0
BAR = 4  # beats to a bar; most popular music. A different real meter still gives beats to cut on.


@dataclass
class BeatMap:
    """The beats of one continuous stretch of music. Times are seconds from its start."""

    beats: np.ndarray  # when each beat is
    period: np.ndarray  # seconds to the next beat, at each beat (about; smoothed over three beats)
    phase: np.ndarray  # place in the bar of each beat: 0 and 2 are strong beats (one and three), 1 and 3 weak ones
    energy: np.ndarray  # 0..1 at each beat: how loud the music is here compared with the rest of the concert
    rise: np.ndarray  # 0..1 at each beat: how much it is building up (louder than a few seconds ago)
    confidence: float  # 0..1: how sure it is that there is a beat here at all (near 0 for talk, applause, noise)

    @property
    def bpm(self) -> np.ndarray:
        return 60.0 / np.maximum(self.period, 1e-3)


# ---------------------------------------------------------------- onset strength and loudness


def _mel_filters(n_mels: int = 40, fmin: float = 30.0, fmax: float = 8000.0) -> tuple[np.ndarray, np.ndarray]:
    """A mel filter bank (n_mels x bins), and the centre frequency of each band."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    bank = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - freqs) / max(hi - mid, 1e-9)
        bank[i] = np.maximum(0.0, np.minimum(up, down))
    return bank / np.maximum(bank.sum(axis=1, keepdims=True), 1e-9), edges[1:-1]


_BANK, _CENTRES = _mel_filters()
_LOW = _CENTRES < 250.0  # the bands where a kick drum lives
_MID = (_CENTRES > 1200.0) & (_CENTRES < 5000.0)  # where a snare cracks, and a kick does not


def analyse(x, chunk_frames: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(onset strength, that of the low bands only, that of the mid bands only, loudness in dB) per 20 ms frame of the
    audio `x` (mono at SR).

    The onset strength is the perceptual spectral flux: per mel band the rise in level (in dB) since the frame before,
    only rises counted, averaged over the bands. Done in chunks, so a long concert needs little memory."""
    n = len(x)
    frames = max(1, int(np.ceil(n / HOP)))
    window = np.hanning(N_FFT).astype(np.float32)
    onset = np.zeros(frames, np.float32)
    low = np.zeros(frames, np.float32)
    mid = np.zeros(frames, np.float32)
    loud = np.zeros(frames, np.float32)
    prev = None
    half = N_FFT // 2
    for f0 in range(0, frames, chunk_frames):
        f1 = min(frames, f0 + chunk_frames)
        lo, hi = f0 * HOP - half, (f1 - 1) * HOP + half + 1
        seg = np.asarray(x[max(lo, 0): min(hi, n)], np.float32)
        seg = np.pad(seg, (max(0, -lo), max(0, hi - n)))
        wins = np.lib.stride_tricks.sliding_window_view(seg, N_FFT)[::HOP][: f1 - f0] * window
        power = np.abs(np.fft.rfft(wins, axis=1)) ** 2
        loud[f0:f1] = 10.0 * np.log10(np.mean(wins ** 2, axis=1) * 2.0 + 1e-10)
        mel = 10.0 * np.log10(power @ _BANK.T + 1e-10)
        mel = np.maximum(mel, -80.0)  # nothing below this is heard; the same floor in every chunk, so no false onsets
        first = mel if prev is None else np.vstack([prev, mel])
        rise = np.maximum(0.0, np.diff(first, axis=0)) if len(first) > 1 else np.zeros((0, mel.shape[1]))
        if prev is None:
            rise = np.vstack([np.zeros((1, mel.shape[1])), rise])
        onset[f0:f1] = rise.mean(axis=1)
        low[f0:f1] = rise[:, _LOW].mean(axis=1)
        mid[f0:f1] = rise[:, _MID].mean(axis=1)
        prev = mel[-1:]
    return onset, low, mid, loud


def _spread(v: np.ndarray) -> np.ndarray:
    """v / its usual size, so that quiet and loud recordings are treated alike."""
    scale = float(np.std(v))
    return v / scale if scale > 1e-9 else v


# ---------------------------------------------------------------- tempo, followed through time


def _tempo_grid() -> np.ndarray:
    return np.exp(np.linspace(np.log(BPM_MIN), np.log(BPM_MAX), 96))


PRIOR_BPM, PRIOR_OCTAVES = 120.0, 1.4  # the tempo most music is tapped at, and how far people stray from it
_COMB = (1.0,) * 8  # a beat repeating 1, 2 ... 8 periods later (two bars) counts towards a tempo, each as much


def local_tempo(onset: np.ndarray, window_s: float = 12.0, step_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(the tempo in BPM, how clearly there is one, 0..1) at each frame. Per window the autocorrelation of the onset
    strength gives a score for each tempo: how strongly the sound repeats after 1, 2 ... 8 beats (all eight, so that a
    false tempo of double, two thirds or four thirds the real one, which matches only some of them, does not win), weighted towards the
    tempi people actually play at. Then the best track through the windows is found, in which a tempo may stay, drift a
    little, or jump (a new song) at a cost. How clear it is says whether this is rhythm at all (talk and noise are not)."""
    n = len(onset)
    grid = _tempo_grid()
    win, step = int(window_s * FPS), max(1, int(step_s * FPS))
    starts = list(range(0, max(1, n - win + step), step)) or [0]
    lags = 60.0 * FPS / grid  # frames per beat, for each tempo
    prior = np.exp(-0.5 * (np.log2(grid / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)  # log-Gaussian round the tempo people tap
    scores = np.zeros((len(starts), len(grid)))
    for w, s in enumerate(starts):
        seg = onset[s: s + win]
        if len(seg) < 6 * FPS:
            scores[w] = scores[w - 1] if w else 1.0
            continue
        seg = seg - uniform_filter1d(seg, int(FPS), mode="nearest")  # only the changes, not a slow loudness drift
        size = 1 << (2 * len(seg) - 1).bit_length()
        spec = np.fft.rfft(seg, size)
        ac = np.fft.irfft(spec * np.conj(spec), size)[: len(seg)]
        ac = ac / (ac[0] + 1e-9)

        def at(lag):
            lag = np.clip(lag, 1, len(ac) - 2)
            i = lag.astype(int)
            return ac[i] * (1 - (lag - i)) + ac[i + 1] * (lag - i)

        comb = sum(wt * at(k * lags) for k, wt in enumerate(_COMB, 1)) / sum(_COMB)
        scores[w] = np.maximum(comb, 0.0) * prior
    peak = scores.max(axis=1)  # the best a window could do, before it is made relative
    scores = scores / (peak[:, None] + 1e-9)
    emit = 6.0 * scores  # what a window says: a clear peak is worth 6, a flat window nothing
    d = np.log2(grid[None, :] / grid[:, None])
    jump = 0.5 * np.minimum((d / 0.05) ** 2, 8.0)  # drifting a few percent is nearly free; a jump costs at most 4
    best = np.zeros_like(scores)
    came = np.zeros(scores.shape, int)
    best[0] = emit[0]
    for w in range(1, len(starts)):
        cand = best[w - 1][:, None] - jump
        came[w] = cand.argmax(axis=0)
        best[w] = cand.max(axis=0) + emit[w]
    path = np.zeros(len(starts), int)
    path[-1] = best[-1].argmax()
    for w in range(len(starts) - 1, 0, -1):
        path[w - 1] = came[w, path[w]]
    centres = np.array(starts) + win / 2
    clarity = np.clip((peak - 0.10) / 0.25, 0.0, 1.0)  # about 0.05 for a drone, 0.4 to 0.8 for music (measured)
    frames = np.arange(n)
    return np.interp(frames, centres, grid[path]), np.interp(frames, centres, clarity)


# ---------------------------------------------------------------- the beats


def track_beats(onset: np.ndarray, tempo: np.ndarray, tightness: float = 100.0) -> np.ndarray:
    """Beat times in frames (fractional). Dynamic programming, as Ellis describes: a beat at frame t scores the onset
    strength there plus the best score of an earlier beat, less a penalty for that gap not being one tempo period
    (measured on a log scale, so half and double the period are as far off as each other)."""
    n = len(onset)
    o = _spread(onset - onset.mean())
    period = 60.0 * FPS / np.clip(tempo, BPM_MIN, BPM_MAX)
    cum = np.zeros(n)
    back = np.full(n, -1)
    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t in range(n):
        p = period[t]
        key = int(round(p * 4))  # a table for each quarter-frame of period
        if key not in tables:
            pp = key / 4.0
            lag = np.arange(max(1, int(np.ceil(pp / 2))), int(2 * pp) + 1)
            tables[key] = (lag, -tightness * np.log(lag / pp) ** 2)
        lag, pen = tables[key]
        prev = t - lag
        ok = prev >= 0
        if not ok.any():
            cum[t] = o[t]
            continue
        sc = cum[prev[ok]] + pen[ok]
        j = int(sc.argmax())
        cum[t] = o[t] + sc[j]
        back[t] = prev[ok][j]
    # the last beat: the best score in the final period; then follow the pointers back
    tail = int(max(0, n - 1 - int(period[-1])))
    t = tail + int(cum[tail:].argmax())
    chain = [t]
    while back[chain[-1]] >= 0:
        chain.append(int(back[chain[-1]]))
    beats = np.array(chain[::-1], float)
    # settle each beat between frames: the peak of a parabola through the onset strength round it
    for i, b in enumerate(beats.astype(int)):
        if 0 < b < n - 1:
            a, c, d = onset[b - 1], onset[b], onset[b + 1]
            denom = a - 2 * c + d
            if denom < -1e-9:
                beats[i] = b + float(np.clip(0.5 * (a - d) / denom, -0.5, 0.5))
    return beats


def drop_offbeats(beats: np.ndarray, low: np.ndarray, ratio: float = 4.0, reach: int = 8) -> np.ndarray:
    """Drop the beats that are really the eighth notes between the beats. The tell is the bass: the beats on the beat
    have a kick or a bass note starting, those between have only hi-hats, so alternate beats differ in the low bands by
    a large factor. On a right beat grid they do not (a kick and a snare are about as strong as each other there). It is
    judged locally, over the beats round each one, because a stretch of a concert has several songs and only some may
    have this trouble: a beat goes if, among the `reach` beats each side, those of its kind have under 1/`ratio` of the
    bass of the others and are, nearly all of them, much the weaker. If this is ever wrong the result is half the tempo,
    whose beats are still in the right places."""
    m = len(beats)
    if m < 16:
        return beats
    idx = np.clip(np.round(beats).astype(int), 2, len(low) - 3)
    strength = np.array([low[i - 2: i + 3].max() for i in idx])  # a kick is short: the peak round the beat
    parity = np.arange(m) % 2
    weak = np.zeros(m, bool)
    for i in range(m):
        lo, hi = max(0, i - reach), min(m, i + reach + 1)
        near = strength[lo:hi]
        same, other = near[parity[lo:hi] == parity[i]], near[parity[lo:hi] != parity[i]]
        if len(same) < 4 or len(other) < 4:
            continue
        if same.mean() * ratio <= other.mean() and (same < 0.5 * other.mean()).mean() >= 0.75:
            weak[i] = True
    return beats[~weak]


def _peak_near(v: np.ndarray, idx: np.ndarray, reach: int = 2) -> np.ndarray:
    """The largest value of `v` within `reach` frames of each of the frames `idx`: an onset is short and can peak a frame
    or two either side of where the beat was put."""
    return np.max([v[np.clip(idx + d, 0, len(v) - 1)] for d in range(-reach, reach + 1)], axis=0)


def bar_phase(beats: np.ndarray, accent: np.ndarray, reach: int = 24) -> np.ndarray:
    """Which beat starts each bar: for every beat, its place (0 to 3) in a bar of four. The strong beats (one and three)
    are where the kick and the bass land, and the accent (bass onset less mid-band onset, which is what tells a kick
    from a snare: without that the loud snare on two and four is mistaken for beat one in rock) is greater on them.
    Deciding among four places from a few bars is noisy, so it is done in two steps: first which half of the beats is the
    strong one (odd or even ones: twice the beats to go on, and the difference is large), then which of the two strong
    beats is called beat one (they are alike to the ear, so this matters little, and it is only a tie-break). Each step
    keeps its answer until the other is clearly stronger, and looks at the `reach` beats each side of a beat, so that it
    can change between songs. Phase 0 and 2 are strong beats; 1 and 3 weak ones."""
    m = len(beats)
    if m == 0:
        return np.zeros(0, int)
    residue = np.arange(m) % BAR
    lo = np.clip(np.arange(m) - reach, 0, m)
    hi = np.clip(np.arange(m) + reach + 1, 0, m)

    def around(x: np.ndarray) -> np.ndarray:  # the sum of x over the beats within `reach` of each beat, at any length
        c = np.concatenate([[0.0], np.cumsum(x)])
        return c[hi] - c[lo]

    strength = np.zeros((BAR, m))
    for r in range(BAR):
        here = residue == r
        strength[r] = around(np.where(here, accent, 0.0)) / np.maximum(around(here.astype(float)), 1.0)
    # the accent is signed (bass less mid): shift so that ratios mean something
    strength = strength - min(0.0, float(strength.min())) + 1e-6
    half = np.stack([(strength[0] + strength[2]) / 2, (strength[1] + strength[3]) / 2])  # residues 0,2 against 1,3

    down = np.zeros(m, int)
    parity, one = 0, 0
    for i in range(m):
        if half[1 - parity, i] > 1.15 * half[parity, i]:
            parity = 1 - parity
        a, b = parity, parity + 2
        first = a if strength[a, i] >= strength[b, i] else b
        cur = one if one % 2 == parity else a
        if first != cur and strength[first, i] > 1.15 * strength[cur, i]:
            cur = first
        one = cur
        down[i] = cur
    return (residue - down) % BAR


# ---------------------------------------------------------------- putting it together


def _percentile_rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return order / max(len(v) - 1, 1)


def beat_maps(x, stretches: list[tuple[float, float]], log=print) -> list[BeatMap | None]:
    """A BeatMap for each stretch (start, length in seconds) of the audio `x` (mono at SR); None where there is too
    little to go on. Intensity is judged against the whole of `x`, since a quiet song is quiet compared with the others."""
    onset, low, mid, loud = analyse(x)
    smooth = uniform_filter1d(loud, int(2 * FPS), mode="nearest")  # about two seconds of level
    intensity = _percentile_rank(smooth)
    build = np.clip(smooth - np.roll(uniform_filter1d(loud, int(2 * FPS), mode="nearest"), int(6 * FPS)), 0, None)
    build[: int(6 * FPS)] = 0
    build = np.clip(build / 6.0, 0.0, 1.0)  # 6 dB louder than 6 seconds ago is a full build-up
    out: list[BeatMap | None] = []
    for start, length in stretches:
        a, b = int(round(start * FPS)), int(round((start + length) * FPS))
        if b - a < int(6 * FPS):
            out.append(None)
            continue
        o = onset[a:b]
        tempo, clarity = local_tempo(o)
        beats = drop_offbeats(track_beats(o, tempo), low[a:b])
        if len(beats) < 4:
            out.append(None)
            continue
        idx = np.clip(np.round(beats).astype(int), 0, len(o) - 1)
        gaps = np.diff(beats) / FPS  # the time to the next beat, as they really are (some may have been dropped)
        period = median_filter(np.append(gaps, gaps[-1]), 3, mode="nearest")
        conf = float(np.median(clarity[idx]))
        accent = _peak_near(_spread(low[a:b]), idx) - _peak_near(_spread(mid[a:b]), idx)  # a kick, not a snare
        out.append(BeatMap(
            beats=beats / FPS, period=period, phase=bar_phase(beats, accent),
            energy=median_filter(intensity[a:b][idx], 5, mode="nearest"), rise=build[a:b][idx], confidence=conf,
        ))
        log(f"  {start:7.1f}s +{length:6.1f}s: {len(beats)} beats, {np.median(60.0 / period):.0f} BPM, "
            f"beat confidence {conf:.2f}")
    return out
