"""Put every clip on one shared timeline by cross-correlating the audio (GCC-PHAT)."""
from __future__ import annotations

import json
import subprocess
import time

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from .media import load_audio, probe
from .project import Project
from .timeline import ClipEntry, Timeline

AR = 16000  # analysis sample rate


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def estimate_remaining(
    pending: list[tuple[int, int]], placed_samples: list[int], most_per_clip: int, seconds_per_sample: float
) -> tuple[float, float]:
    """Seconds of matching left: (expected, worst case).

    `pending` has (length in samples, comparisons so far) for every clip still unplaced; `placed_samples` the lengths
    of the placed ones. A clip is compared against every placed clip it has not met yet, longest first, up to
    `most_per_clip` in all, whether or not it ends up matching. So a clip placed when k clips are already placed costs
    about min(most_per_clip, k) comparisons, and one that never matches costs min(most_per_clip, final number of
    placed clips): the cost grows with the placed set, which is why an average over the clips placed so far is far
    too low early on. `seconds_per_sample` is the measured cost of a comparison per sample of the two clips together.

    Worst case: every pending clip uses its whole allowance. Expected: the share of the clips seen so far that matched
    (one in two before there is any evidence) says how many clips will be placed in all, and so how many comparisons
    each of the rest needs.
    """
    if not pending:
        return 0.0, 0.0
    top = sorted(placed_samples, reverse=True)[:most_per_clip]  # what every pending clip is compared against
    mean_placed = sum(top) / len(top)
    placed = len(placed_samples)
    clips = placed - 1 + len(pending)  # all clips but the one everything starts from
    seen = placed - 1 + sum(1 for _, c in pending if c)  # clips that have had at least one chance to match
    share = placed / (seen + 2)
    final_placed = min(clips + 1, max(float(placed), 1 + share * clips))
    to_place = final_placed - placed  # how many of the pending clips will match, on average
    p_member = to_place / len(pending)
    as_member = min(most_per_clip, placed + to_place / 2)  # placed on average halfway through the ones to come
    as_stranger = min(most_per_clip, final_placed)  # never matches: compared against all that end up placed

    worst = expected = 0.0
    for samples, done in pending:
        per_comparison = (samples + mean_placed) * seconds_per_sample
        worst += max(0, most_per_clip - done) * per_comparison
        left = p_member * max(0.0, as_member - done) + (1 - p_member) * max(0.0, as_stranger - done)
        expected += left * per_comparison
    return min(expected, worst), worst


def correlate(a, b, fs: int, band=(150.0, 5000.0), min_overlap: float = 5.0) -> tuple[float, float]:
    """GCC-PHAT cross-correlation of two mono signals.

    Returns (lag_seconds, z): b starts `lag` seconds after a, and z is how many standard
    deviations the correlation peak stands above the rest (true matches score z > 25, unrelated audio rarely exceeds 8;
    pick a threshold in between).
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    la, lb = len(a), len(b)
    min_ov = int(min_overlap * fs)
    if min(la, lb) < min_ov:
        return 0.0, 0.0

    nfft = next_fast_len(la + lb - 1, real=True)
    # workers=1 on purpose: scipy's multi-threaded FFT sometimes returned corrupt output on these large arrays
    # (about 1 run in 200: a bogus peak at a random lag), and was slower than single-threaded anyway.
    A = rfft(a - a.mean(), nfft, workers=1)
    B = rfft(b - b.mean(), nfft, workers=1)
    R = A * np.conj(B)
    mag = np.abs(R)
    # PHAT whitening, floored so near-empty bins are not blown up into noise
    R /= np.maximum(mag, 0.01 * mag.mean() + 1e-30)
    R[: int(band[0] * nfft / fs)] = 0
    R[int(band[1] * nfft / fs) + 1:] = 0
    cc = irfft(R, nfft, workers=1)

    # Only lags where the clips overlap by at least min_overlap are candidates.
    lags = np.arange(min_ov - lb, la - min_ov + 1)
    vals = cc[lags % nfft]
    i = int(np.argmax(vals))
    z = (float(vals[i]) - float(vals.mean())) / (float(vals.std()) + 1e-30)
    return float(lags[i]) / fs, z


def sync_project(
    project: Project, min_z: float = 10.0, min_overlap: float = 5.0, max_compare: int = 25, log=print,
) -> Timeline:
    files = project.clip_files()
    if not files:
        raise SystemExit(f"No media files in {project.clips_dir}")

    rejected: list[dict] = []
    audio: dict[str, np.ndarray] = {}
    infos = {}
    for f in files:
        try:
            info = probe(f)
            if not info.has_audio:
                rejected.append({"file": f.name, "reason": "no audio track"})
                continue
            x = np.asarray(load_audio(f, project.cache_dir, AR))
        except (subprocess.CalledProcessError, RuntimeError) as e:
            rejected.append({"file": f.name, "reason": f"could not decode: {e}"})
            continue
        if len(x) / AR < min_overlap:
            rejected.append({"file": f.name, "reason": f"shorter than {min_overlap}s"})
            continue
        infos[f.name] = info
        audio[f.name] = x
    log(f"Decoded {len(audio)} clip(s) with audio.")

    # Grow a set of aligned clips outward from the longest one. Each unplaced clip is correlated
    # against the placed ones and joins if its best match is confident enough.
    order = sorted(audio, key=lambda n: -len(audio[n]))
    placed: dict[str, tuple[float, float | None, str | None]] = {}
    if order:
        placed[order[0]] = (0.0, None, None)
    pending = order[1:]
    tried: dict[tuple[str, str], tuple[float, float]] = {}
    best_z = {n: 0.0 for n in pending}
    compared = {n: 0 for n in pending}

    started = time.perf_counter()
    spent = 0.0  # seconds inside correlate()
    samples = 0  # length of the two clips, added up over the comparisons finished: what their cost depends on
    done = 0  # comparisons finished
    most_per_clip = min(max_compare, len(order) - 1)

    def time_left() -> str:
        if not done:
            return ""
        expected, worst = estimate_remaining(
            [(len(audio[n]), compared[n]) for n in pending], [len(audio[n]) for n in placed],
            most_per_clip, spent / samples,
        )
        return f" | ~{format_duration(expected)} left (at most {format_duration(worst)})"

    progress = True
    while progress and pending:
        progress = False
        for u in list(pending):
            best = None
            # Longest aligned clips first: they are the most likely to overlap. Every comparison costs a
            # large FFT, so a clip that matches nothing is only tried against `max_compare` of them.
            for p in sorted(placed, key=lambda n: -len(audio[n])):
                if (u, p) not in tried:
                    if compared[u] >= max_compare:
                        continue
                    compared[u] += 1
                    t = time.perf_counter()
                    tried[(u, p)] = correlate(audio[p], audio[u], AR, min_overlap=min_overlap)
                    spent += time.perf_counter() - t
                    samples += len(audio[p]) + len(audio[u])
                    done += 1
                    log(
                        f"  {u} vs {p}: z={tried[(u, p)][1]:.1f} | {len(placed)}/{len(order)} placed, "
                        f"{done} comparisons, {format_duration(time.perf_counter() - started)} elapsed{time_left()}"
                    )
                lag, z = tried[(u, p)]
                best_z[u] = max(best_z[u], z)
                if z >= min_z and (best is None or z > best[2]):
                    best = (p, lag, z)
            if best:
                p, lag, z = best
                placed[u] = (placed[p][0] + lag, z, p)
                pending.remove(u)
                progress = True
                log(f"  placed {u} at {placed[u][0]:+.3f}s (z={z:.1f}, via {p})")
    log(f"Matching took {format_duration(time.perf_counter() - started)} ({done} comparisons).")

    for u in pending:
        rejected.append({"file": u, "reason": f"no confident audio match (best z={best_z[u]:.1f}, need {min_z})"})

    shift = min((v[0] for v in placed.values()), default=0.0)
    clips = [
        ClipEntry(
            file=n, offset=off - shift, duration=len(audio[n]) / AR,
            has_video=infos[n].has_video, width=infos[n].width, height=infos[n].height,
            z=z, anchor=anchor,
        )
        for n, (off, z, anchor) in sorted(placed.items(), key=lambda kv: kv[1][0])
    ]
    tl = Timeline(clips, rejected)
    tl.save(project.timeline_path)

    titles = {}
    if project.sources_path.exists():
        sources = json.loads(project.sources_path.read_text("utf-8"))
        titles = {f.name: sources.get(f.stem, {}).get("title") or "" for f in files}

    log(f"\nAligned {len(clips)} clip(s), rejected {len(rejected)}. Timeline: {project.timeline_path}")
    for c in clips:
        log(f"  {c.offset:9.3f}s  +{c.duration:7.1f}s  {'video' if c.has_video else 'audio'}  {c.file}  {titles.get(c.file, '')}")
    for r in rejected:
        log(f"  rejected {r['file']}: {r['reason']}  {titles.get(r['file'], '')}")
    return tl
