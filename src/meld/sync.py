"""Put every clip on one shared timeline by cross-correlating the audio (GCC-PHAT)."""
from __future__ import annotations

import itertools
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from .matchview import MatchState
from .media import audio_cache_path, load_audio, probe
from .pool import Comparer, default_workers
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
    deviations the correlation peak stands above the rest (true matches score z > 25 when the recordings are good;
    unrelated concert audio usually stays under 8 but now and then reaches 10 or more, more so for long clips, so a
    score in between needs a second look: see `support`).
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    la, lb = len(a), len(b)
    min_ov = int(min_overlap * fs)
    if min(la, lb) < min_ov:
        return 0.0, 0.0

    cc, nfft = _phat_cc(a, b, fs, band)
    # Only lags where the clips overlap by at least min_overlap are candidates.
    lags = np.arange(min_ov - lb, la - min_ov + 1)
    vals = cc[lags % nfft]
    i = int(np.argmax(vals))
    z = (float(vals[i]) - float(vals.mean())) / (float(vals.std()) + 1e-30)
    return float(lags[i]) / fs, z


def _phat_cc(a: np.ndarray, b: np.ndarray, fs: int, band=(150.0, 5000.0)) -> tuple[np.ndarray, int]:
    """The whitened cross-correlation of two clips (as float32 arrays), circular in `nfft` points, and `nfft`."""
    nfft = next_fast_len(len(a) + len(b) - 1, real=True)
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
    return irfft(R, nfft, workers=1), nfft


def support(a, b, lag: float, fs: int, parts: int = 3, tol: float = 0.1, min_part: float = 3.0) -> list[float] | None:
    """Does the match at `lag` (b starts `lag` seconds after a) hold up all through the overlap?

    A real overlap has the same sound all along, so cutting it in `parts` and correlating each part on its own finds
    the same lag in every one: the score here is, per part, the correlation within `tol` seconds of `lag` in standard
    deviations of that part's own correlation. A chance peak is not there in the parts. Measured on real concert audio:
    unrelated clips reach z=10 and more in `correlate` (one pair in 20 to 40, more the longer the clips), but score 4 to
    5 here in a part and at most about 6 in every one; real overlaps, even with noise a lot louder than the sound
    (-15 dB), score 11 and up in every part. Returns None when the overlap is too short to cut in two parts of at least
    `min_part` seconds (nothing to check with), else one score per part."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    shift = int(round(lag * fs))
    a0, b0 = max(0, shift), max(0, -shift)  # a[a0 + i] and b[b0 + i] are the same moment
    n = min(len(a) - a0, len(b) - b0)
    parts = min(parts, int(n / (min_part * fs)))
    if parts < 2:
        return None
    scores = []
    for k in range(parts):
        i0, i1 = k * n // parts, (k + 1) * n // parts
        part_a, part_b = a[a0 + i0: a0 + i1], b[b0 + i0: b0 + i1]
        cc, nfft = _phat_cc(part_a, part_b, fs)
        reach = len(part_a) // 2  # the lags where the parts still overlap by half
        lags = np.arange(-reach, reach + 1)
        vals = cc[lags % nfft]
        near = vals[np.abs(lags) <= int(tol * fs)].max()
        scores.append(float((near - vals.mean()) / (vals.std() + 1e-30)))
    return scores


CONFIDENT_Z = 25.0  # a match this strong needs no second opinion (real matches, undisturbed, score above this)
MIN_SUPPORT = 7.0  # a weaker one must score this in `support` ...
SUPPORT_PARTS = 2  # ... in this many parts of its overlap (of three). Real overlaps score at least 11 in each, in noise
MIN_OFFERED = 3  # a group needs at least this many clips to be worth offering as a choice
LOOSE_PASSES = 2  # how many times the clips that matched nothing get another look
AHEAD_CLIPS, AHEAD_PAIRS = 3, 1  # idle workers get the likely first comparisons of this many next clips (this many each)
OWN_FIRST = 2  # ... but only after this many of the current clip's own: most clips match at once, and its later ones are guesses
POOL_MIN_CLIPS = 8  # worker processes take a moment to start: not worth it for fewer clips than this ...
POOL_MIN_SECONDS = 20  # ... or for clips shorter than this on average (when the number of workers is left to us)

# ---------------------------------------------------------------- groups that grow side by side


class _Groups:
    """Aligned groups of clips. Each group has its own time; a clip's offset is where in it the clip starts."""

    def __init__(self) -> None:
        self.members: dict[int, dict[str, float]] = {}
        self.home: dict[str, int] = {}
        self.link: dict[str, tuple[str | None, float | None]] = {}  # clip -> (clip it was lined up with, z)
        self._next = 0

    def seed(self, name: str) -> int:
        gid, self._next = self._next, self._next + 1
        self.members[gid] = {name: 0.0}
        self.home[name] = gid
        self.link[name] = (None, None)
        return gid

    def add(self, gid: int, name: str, offset: float, anchor: str, z: float) -> None:
        self.members[gid][name] = offset
        self.home[name] = gid
        self.link[name] = (anchor, z)

    def merge(self, into: int, other: int, shift: float) -> None:
        for name, offset in self.members.pop(other).items():
            self.members[into][name] = offset + shift
            self.home[name] = into

    def loose(self, name: str) -> bool:
        return len(self.members[self.home[name]]) == 1


def estimate_remaining_groups(
    budget: int, seconds_per_sample: float, processed: list[int], pending: list[int], phase1_done: int,
    loose: list[int], tried_pairs: int, second_look: float | None = None,
) -> tuple[float, float]:
    """Seconds of matching left when clips are sorted into groups side by side: (expected, worst case).

    The work has two parts. First every clip, longest first, is compared with up to `budget` clips already sorted; the
    i-th clip has i to choose from, and how much of that it uses (a clip that joins a group stops early) is learned
    from the clips done so far (`phase1_done` comparisons for `processed`; lengths are in samples). `pending` are the
    ones still to do. Then the clips that matched nothing (`loose`) get a second look against everything they have not
    met. A comparison is shared by the two clips, so this part costs far less than a budget per clip: the first clip
    compared with the others leaves the next one fewer to do, and so on. Once that part has begun the caller knows
    exactly what is left of it and passes it as `second_look`. `tried_pairs` is how many pairs are already compared:
    there are only so many pairs, which caps the worst case. `seconds_per_sample` is the measured cost of a comparison
    per sample of the two clips together.
    """
    done_n, todo_n = len(processed), len(pending)
    total = done_n + todo_n
    if total < 2:
        return 0.0, 0.0
    biggest = max(processed + pending)
    pairs_left = max(0, total * (total - 1) // 2 - tried_pairs)
    ceiling = pairs_left * 2 * biggest * seconds_per_sample  # every pair left, each as dear as the dearest
    if second_look is not None:  # the first part is over
        return min(second_look, ceiling), min(second_look * LOOSE_PASSES, ceiling)

    top = sorted(processed, reverse=True)[:budget] or sorted(pending, reverse=True)[:budget]
    partner = sum(top) / len(top)  # what a clip is compared with tends to be the longer clips
    everyone = (sum(processed) + sum(pending)) / total

    slots = sum(min(budget, j) for j in range(1, done_n))  # comparisons the clips done so far could have made
    used_share = min(1.0, (phase1_done + 2.5) / (slots + 5))  # about half before there is any evidence
    first = first_worst = 0.0
    for r, samples in enumerate(pending):
        options = min(budget, done_n + r)
        cost = (samples + partner) * seconds_per_sample
        first += used_share * options * cost
        first_worst += options * cost

    loose_now = len(loose)
    share = (loose_now + 1) / (done_n + 2)  # of the clips still to do, about this many will match nothing
    n_loose = int(round(loose_now + share * todo_n))
    mean_loose = sum(loose) / loose_now if loose_now else everyone
    outside = max(0.0, total - n_loose - budget)  # partners the first part did not get to
    second = sum(min(budget, n_loose - 1 - i + outside) for i in range(n_loose))
    second *= (mean_loose + everyone) * seconds_per_sample
    second_worst = (loose_now + todo_n) * min(budget, total - 1) * (everyone + partner) * seconds_per_sample * LOOSE_PASSES
    worst = min(first_worst + second_worst, ceiling)
    return min(first + second, worst), worst


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _align_groups(
    audio: dict, min_z: float, min_overlap: float, max_compare: int, log, comparer: Comparer, on_state=None,
) -> tuple[_Groups, dict]:
    """Sort clips into groups that line up with each other, all growing at the same time.

    Longest clip first, each clip is compared with the clips already sorted (the bigger a group, the more of the
    clip's `max_compare` comparisons it gets, longest members first; a match as strong as CONFIDENT_Z ends the search
    in that group) and:
    - joins the group it matches best;
    - seeds a group of its own if it matches nothing;
    - if it matches two groups, it is the bridge between them: they become one.
    Clips that matched nothing then get another look, against everything they have not met (a group of two only forms
    when its clips meet). Returns the groups and the best z each clip reached.

    The comparisons are what takes the time. `comparer` may run several at once in worker processes: while one is
    looked at, the next ones the search would make (in the order it would make them) are already being computed. They
    are still used in that order, under the same rules, so the result is the same as one at a time; a comparison
    started that turns out not to be needed is thrown away.
    """
    order = sorted(audio, key=lambda n: -len(audio[n]))
    total = len(order)
    budget = max(1, min(max_compare, total - 1))
    groups = _Groups()
    tried: dict[tuple[str, str], tuple[float, float]] = {}  # (a, b) sorted -> (lag, z): b starts `lag` s after a
    checked: dict[tuple[str, str], bool] = {}  # (a, b) sorted -> whether its score is to be believed (see confirmed)
    compared = {n: 0 for n in order}  # comparisons each clip has been part of
    best_z = {n: 0.0 for n in order}

    started = time.perf_counter()
    spent = samples = 0.0  # seconds inside correlate(), and the length of the two clips, summed over comparisons
    done = phase1_done = 0
    processed: list[str] = []
    phase = 1
    lengths = {n: len(audio[n]) / AR for n in order}  # seconds, for `on_state`
    current: str | None = None  # the clip being sorted now
    pending_from = 0  # in the first sweep, order[pending_from:] has not been looked at yet
    retry: list[str] = []  # phase 2: the clips to get another look
    retried = looking = 0
    second_units = 0.0  # phase 2: what this pass is expected to cost, in samples of the two clips compared
    pass_start = (0.0, 0.0, 0)  # `spent`, `samples` and `done` when the pass began

    def time_left() -> str:
        if not samples:  # nothing to go on until the second comparison
            return ""
        lengths = [len(audio[n]) for n in processed]
        if phase == 1:
            todo = [len(audio[n]) for n in order[len(processed):]]
            expected, worst = estimate_remaining_groups(
                budget, spent / samples, lengths, todo, phase1_done,
                [len(audio[n]) for n in processed if groups.loose(n)], len(tried),
            )
        else:
            # what a comparison costs in this pass (several run at once, and the clips here are the loose ones) is
            # better known from the pass itself than from the first part
            pass_spent, pass_samples, pass_done = spent - pass_start[0], samples - pass_start[1], done - pass_start[2]
            rate = pass_spent / pass_samples if pass_done >= 4 and pass_samples else spent / samples
            expected, worst = estimate_remaining_groups(
                budget, spent / samples, lengths, [], phase1_done, [], len(tried),
                second_look=max(0.0, second_units - pass_samples) * rate,
            )
        return f" | ~{format_duration(expected)} left (at most {format_duration(worst)})"

    def progress() -> str:
        return f"{len(processed)}/{total} placed" if phase == 1 else f"{retried}/{looking} placed (a second look)"

    def emit(compare: tuple[str, str, float | None] | None = None, doubted: bool = False) -> None:
        """Tell whoever is watching (the window) how things stand: a snapshot, so they need not follow every step."""
        if on_state is not None:
            on_state(MatchState(
                durations=lengths, groups=tuple(dict(m) for m in groups.members.values()),
                pending=tuple(order[pending_from:]) if phase == 1 else (), current=current, compare=compare,
                phase=phase, compared=done, min_z=min_z, doubted=doubted,
            ))

    def compare(u: str, p: str) -> None:
        nonlocal spent, samples, done, phase1_done
        a, b = _pair(u, p)
        emit((u, p, None))  # this pair is being worked out (or waited for, if a worker is already on it)
        t = time.perf_counter()
        tried[(a, b)] = comparer.result(a, b)  # waits for it if it is being computed in the background
        if done:  # the first comparison also pays for starting the workers and for cold caches: not part of the rate
            spent += time.perf_counter() - t
            samples += len(audio[a]) + len(audio[b])
        done += 1
        phase1_done += phase == 1
        compared[u] += 1
        compared[p] += 1
        z = tried[(a, b)][1]
        doubted = not confirmed(u, p)
        log(
            f"  {u} vs {p}: z={z:.1f}{' (does not hold through the overlap: not a match)' if doubted and z >= min_z else ''}"
            f" | {progress()}, {done} comparisons, {format_duration(time.perf_counter() - started)} elapsed{time_left()}"
        )
        emit((u, p, z), doubted and z >= min_z)

    def after(u: str, p: str) -> tuple[float, float]:
        """(lag, z) of the pair, with lag = how many seconds later u starts than p."""
        a, b = _pair(u, p)
        lag, z = tried[(a, b)]
        return (lag if b == u else -lag), z

    def confirmed(u: str, p: str) -> bool:
        """Is the score of this pair to be believed? One that is high enough (CONFIDENT_Z) always is. A middling one
        (from min_z up) is a match only if it holds through the overlap (see `support`): among the hundreds of pairs
        in a run there are always a few unrelated ones that reach it, and each would put a song, or a whole night, on
        top of another. A pair too short an overlap to check is believed, as before."""
        a, b = _pair(u, p)
        if a == b or (a, b) not in tried:
            return True
        if (a, b) not in checked:
            lag, z = tried[(a, b)]
            if z < min_z or z >= CONFIDENT_Z:
                checked[(a, b)] = True
            else:
                scores = support(audio[a], audio[b], lag, AR)
                checked[(a, b)] = scores is None or sum(v >= MIN_SUPPORT for v in scores) >= min(SUPPORT_PARTS, len(scores))
        return checked[(a, b)]

    def plan(u: str) -> list[tuple[int, int, list[str]]]:
        """In what order u is compared with the other groups, how many comparisons each may get, and its members."""
        here = groups.home.get(u)
        big = sorted((g for g in groups.members if g != here and len(groups.members[g]) > 1),
                     key=lambda g: -len(groups.members[g]))
        single = sorted((g for g in groups.members if g != here and len(groups.members[g]) == 1),
                        key=lambda g: -len(audio[next(iter(groups.members[g]))]))
        in_big = sum(len(groups.members[g]) for g in big) or 1
        return [
            (
                gid,
                1 if len(groups.members[gid]) == 1 else max(2, math.ceil(budget * len(groups.members[gid]) / in_big)),
                sorted(groups.members[gid], key=lambda n: -len(audio[n])),
            )
            for gid in big + single
        ]

    def second_look_units(names: list[str]) -> float:
        """What looking at these clips again will cost if nothing more matches. A comparison is shared by its two
        clips, so a pair is only paid for once."""
        counted: set[tuple[str, str]] = set()
        cost = 0
        for u in names:
            used = 0
            for _, quota, members in plan(u):
                new = 0
                for p in members:
                    key = _pair(u, p)
                    if key in tried or key in counted:
                        continue
                    if new >= quota or used >= budget:
                        break
                    counted.add(key)
                    new += 1
                    used += 1
                    cost += len(audio[u]) + len(audio[p])
                if used >= budget:
                    break
        return float(cost)

    def attach(u: str, following: tuple[str, ...] = ()) -> dict[int, tuple[str, float, float]]:
        """Compare u with the other groups. Returns, for each group it matches, (member, lag, z) of its best match."""
        found: dict[int, tuple[str, float, float]] = {}
        used = 0
        steps = plan(u)
        singles = {gid for gid, _, members in steps if len(members) == 1}
        finished: set[int] = set()  # groups this search is done with

        def upcoming(skipped: frozenset, used_so_far: int):
            """The pairs the rest of this search would compare, in order, if nothing stopped it early."""
            n = used_so_far
            for gid, quota, members in steps:
                if gid in skipped:
                    continue
                new = 0
                for p in members:
                    if _pair(u, p) in tried:
                        continue
                    if new >= quota or n >= budget:
                        break
                    yield _pair(u, p)
                    new += 1
                    n += 1
                if n >= budget:
                    return

        def ahead():
            """Then what the next few clips would compare first: worked out early, for workers that would otherwise
            be idle while this clip finishes, and only used if the search really asks for it."""
            for v in following:
                n = 0
                for _, quota, members in plan(v):
                    new = 0
                    for p in members:
                        if _pair(v, p) in tried:
                            continue
                        if new >= quota or n >= AHEAD_PAIRS:
                            break
                        yield _pair(v, p)
                        new += 1
                        n += 1
                    if n >= AHEAD_PAIRS:
                        break

        def feeder(skipped: frozenset, used_so_far: int):
            own = upcoming(skipped, used_so_far)
            return itertools.chain(itertools.islice(own, OWN_FIRST), ahead(), own)

        feed = feeder(frozenset(), 0)
        comparer.top_up(feed)
        for gid, quota, members in steps:
            if used >= budget:
                break
            if len(members) == 1 and phase == 1 and found:
                continue  # a clip that has found its group leaves the loose ends to the second look
            new, best = 0, None
            for p in members:
                if _pair(u, p) not in tried:
                    if new >= quota or used >= budget:
                        break
                    compare(u, p)
                    comparer.top_up(feed)
                    new += 1
                    used += 1
                lag, z = after(u, p)
                best_z[u] = max(best_z[u], z)
                if z >= min_z and confirmed(u, p) and (best is None or z > best[2]):
                    best = (p, lag, z)
                if best and best[2] >= CONFIDENT_Z:
                    break
            finished.add(gid)
            first_find = best is not None and not found
            if best:
                found[gid] = best
            # The comparisons started ahead assumed the search would go on as far as it can. It stopped early in this
            # group, or (in the first part) has just found its group and will skip the loose ends: start again from here.
            if best and (best[2] >= CONFIDENT_Z or (phase == 1 and first_find)):
                comparer.cancel()
                feed = feeder(frozenset(finished | (singles if phase == 1 else set())), used)
                comparer.top_up(feed)
        return found

    def place(u: str, found: dict[int, tuple[str, float, float]]) -> None:
        main = max(found, key=lambda g: found[g][2])
        p, lag, z = found[main]
        offset = groups.members[main][p] + lag
        if u in groups.home:  # a loose clip: its one-clip group goes away
            del groups.members[groups.home.pop(u)]
        groups.add(main, u, offset, p, z)
        log(f"  placed {u} at {offset:+.3f}s (z={z:.1f}, via {p})")
        for gid, (q, lag_q, _) in found.items():
            if gid != main:  # u lines up with this group too: it is the bridge, and the two groups become one
                shift = (offset - lag_q) - groups.members[gid][q]
                groups.merge(main, gid, shift)
                log(f"  {u} links two groups (via {q}): they are one now")

    emit()
    for i, u in enumerate(order):
        current, pending_from = u, i + 1
        emit()
        found = attach(u, tuple(order[i + 1: i + 1 + AHEAD_CLIPS])) if groups.members else {}
        if found:
            place(u, found)
        else:
            groups.seed(u)
        processed.append(u)
        emit()

    phase = 2
    for _ in range(LOOSE_PASSES):
        retry = sorted((n for n in order if groups.loose(n)), key=lambda n: -len(audio[n]))
        if not retry or len(groups.members) < 2:
            break
        retried, looking, merged = 0, len(retry), False
        second_units, pass_start = second_look_units(retry), (spent, samples, done)
        for i, u in enumerate(retry):
            if groups.loose(u):  # not already taken into a group by a clip that looked before it
                current = u
                emit()
                found = attach(u, tuple(retry[i + 1: i + 1 + AHEAD_CLIPS]))
                if found:
                    place(u, found)
                    merged = True
                emit()
            retried += 1
        if not merged:
            break
    current = None
    emit()
    log(f"Matching took {format_duration(time.perf_counter() - started)} ({done} comparisons).")
    return groups, best_z


# ---------------------------------------------------------------- the original: one group grown from the longest clip


def _align_single(audio: dict, min_z: float, min_overlap: float, max_compare: int, log) -> tuple[dict, list, dict]:
    """Grow one set of aligned clips outward from the longest clip. Each unplaced clip is correlated against the
    placed ones and joins if its best match is confident enough. Returns (placed, pending, best z per pending clip)."""
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
    return placed, pending, best_z


def sync_project(
    project: Project, min_z: float = 10.0, min_overlap: float = 5.0, max_compare: int = 25, log=print,
    clusters: bool = True, cluster: int = 1, choose=None, workers: int | None = None, on_state=None,
) -> Timeline:
    """Line all the clips up. With `clusters` (the default) clips are sorted into groups that line up with each other,
    all at once, so it does not matter which clip is the longest or which concert it is from; the biggest group is
    used (`cluster`=2 takes the second biggest, and so on). `choose(groups)`, if given, is asked which one when there
    are several worth choosing between (each a Timeline); it returns an index into that list, or None for the biggest.
    Without `clusters`, one group is grown from the longest clip and everything that does not join it is rejected.
    `workers` is how many comparisons are run at once, in separate processes: None leaves it to the number of cores and
    the size of the job, 1 is one at a time. The result does not depend on it. `on_state(MatchState)`, if given, is
    called as the matching goes on (from the thread that runs it) to show it as it happens; see matchview.py."""
    files = project.clip_files()
    if not files:
        raise SystemExit(f"No media files in {project.clips_dir}")

    rejected: list[dict] = []
    audio: dict[str, np.ndarray] = {}
    paths: dict[str, str] = {}  # where each clip's decoded audio is kept: the worker processes read it from there
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
        paths[f.name] = str(audio_cache_path(f, project.cache_dir, AR))
    log(f"Decoded {len(audio)} clip(s) with audio.")

    def no_match(name: str, z: float) -> dict:
        return {"file": name, "reason": f"no confident audio match (best z={z:.1f}, need {min_z})"}

    def entries(offsets: dict, link: dict) -> list[ClipEntry]:
        """The clips of one group, with times counted from its first clip."""
        shift = min(offsets.values(), default=0.0)
        return [
            ClipEntry(
                file=n, offset=off - shift, duration=len(audio[n]) / AR,
                has_video=infos[n].has_video, width=infos[n].width, height=infos[n].height,
                z=link[n][1], anchor=link[n][0],
            )
            for n, off in sorted(offsets.items(), key=lambda kv: kv[1])
        ]

    others: list[list[ClipEntry]] = []
    if clusters:
        if workers is not None:
            n_workers = workers
        else:
            average = sum(len(x) for x in audio.values()) / max(len(audio), 1)
            n_workers = default_workers() if len(audio) >= POOL_MIN_CLIPS and average >= POOL_MIN_SECONDS * AR else 1
        comparer = Comparer(
            paths, lambda a, b: correlate(audio[a], audio[b], AR, min_overlap=min_overlap), n_workers, AR, min_overlap,
            on_fallback=lambda why: log(f"  ({why}: carrying on one comparison at a time)"),
        )
        if comparer.parallel:
            log(f"Comparing up to {comparer.workers} clips at a time.")
        last: list[MatchState] = []

        def watch(state: MatchState) -> None:
            last[:] = [state]
            on_state(state)

        try:
            groups, best_z = _align_groups(
                audio, min_z, min_overlap, max_compare, log, comparer, watch if on_state is not None else None,
            )
        finally:
            comparer.close()
        ranked = sorted(
            groups.members.values(),
            key=lambda m: (-len(m), -(max(o + len(audio[n]) / AR for n, o in m.items()) - min(m.values()))),
        )
        made = [entries(m, groups.link) for m in ranked]
        pick = 0
        worth = [i for i, g in enumerate(made) if len(g) >= MIN_OFFERED]
        if choose is not None and len(worth) >= 2:
            answer = choose([Timeline(made[i]) for i in worth])
            pick = worth[answer or 0]
        elif cluster > 1:
            if cluster > len(made):
                raise SystemExit(f"Only {len(made)} group(s) of clips line up; there is no group {cluster}.")
            pick = cluster - 1
        clips = made[pick] if made else []
        if on_state is not None and last:  # over: which group is used
            on_state(replace(
                last[0], current=None, compare=None, pending=(), finished=True, chosen=frozenset(c.file for c in clips),
            ))
        for i, group in enumerate(made):
            if i == pick:
                continue
            if len(group) >= 2:
                others.append(group)
            else:
                rejected.append(no_match(group[0].file, best_z[group[0].file]))
        if len(made) > 1:
            log("Groups of clips that line up with each other (the fused one is marked):")
            for i, group in enumerate(made):
                span = Timeline(group).end / 60
                log(f"  {'->' if i == pick else '  '} {len(group)} clip(s), {span:.0f} min")
    else:
        placed, pending, best_z = _align_single(audio, min_z, min_overlap, max_compare, log)
        rejected += [no_match(u, best_z[u]) for u in pending]
        clips = entries({n: v[0] for n, v in placed.items()}, {n: (v[2], v[1]) for n, v in placed.items()})

    tl = Timeline(clips, rejected, others)
    tl.save(project.timeline_path)

    titles = {}
    if project.sources_path.exists():
        sources = json.loads(project.sources_path.read_text("utf-8"))
        titles = {f.name: sources.get(f.stem, {}).get("title") or "" for f in files}

    log(f"\nAligned {len(clips)} clip(s), rejected {len(rejected)}"
        + (f", {sum(len(g) for g in others)} more in {len(others)} other group(s)" if others else "")
        + f". Timeline: {project.timeline_path}")
    for c in clips:
        log(f"  {c.offset:9.3f}s  +{c.duration:7.1f}s  {'video' if c.has_video else 'audio'}  {c.file}  {titles.get(c.file, '')}")
    for n, group in enumerate(others, 2):
        for c in group:
            log(f"  group {n}: {c.file}  {titles.get(c.file, '')}")
    for r in rejected:
        log(f"  rejected {r['file']}: {r['reason']}  {titles.get(r['file'], '')}")
    return tl


def line_up_downloads(
    preview: Project, project: Project, entries: list[ClipEntry], min_z: float = 10.0, log=print,
) -> Timeline:
    """Carry a timeline made from audio previews over to the videos downloaded afterwards.

    `entries` are the chosen clips as they were lined up in `preview`. The sound of each downloaded video is the
    sound of its preview, but the files are not cut at exactly the same place, so each video is correlated with its own
    preview (one comparison per clip, not a whole new sync) and moves by the small shift found. A video that is missing,
    or whose sound does not match its preview, is left out. The result is saved as the project's timeline."""
    clips: list[ClipEntry] = []
    rejected: list[dict] = []
    for e in entries:
        video = project.find_clip(Path(e.file).stem)
        if video is None:
            rejected.append({"file": e.file, "reason": "the video was not downloaded"})
            continue
        try:
            info = probe(video)
            x = np.asarray(load_audio(video, project.cache_dir, AR))
            before = np.asarray(load_audio(preview.clip_path(e), preview.cache_dir, AR))
        except (subprocess.CalledProcessError, RuntimeError, OSError) as err:
            rejected.append({"file": video.name, "reason": f"could not decode: {err}"})
            continue
        lag, z = correlate(before, x, AR)  # the video starts `lag` seconds after its preview
        log(f"  {video.name}: lined up with its preview (z={z:.1f}, moved {lag * 1000:+.0f} ms)")
        if z < min_z:
            rejected.append({"file": video.name, "reason": f"the video does not match its preview (z={z:.1f})"})
            continue
        clips.append(ClipEntry(
            file=video.name, offset=e.offset + lag, duration=len(x) / AR, has_video=info.has_video,
            width=info.width, height=info.height, z=z, anchor=e.file,
        ))
    shift = min((c.offset for c in clips), default=0.0)
    clips = sorted((replace(c, offset=c.offset - shift) for c in clips), key=lambda c: c.offset)
    tl = Timeline(clips, rejected)
    tl.save(project.timeline_path)
    return tl
