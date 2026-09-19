import numpy as np
from synth import phone, synth_music

from meld.sync import MIN_SUPPORT, SUPPORT_SHARE, correlate, holds, support

SR = 16000


def test_recovers_positive_lag():
    music = synth_music(90, SR)
    a = phone(music, SR, 0, 60, seed=1)
    b = phone(music, SR, 20.3, 80, seed=2, gain=3.0)  # clipped, louder, starts 20.3 s later
    lag, z = correlate(a, b, SR)
    assert abs(lag - 20.3) < 0.002
    assert z > 10


def test_recovers_negative_lag():
    music = synth_music(90, SR)
    a = phone(music, SR, 30, 80, seed=3)
    b = phone(music, SR, 5.55, 45, seed=4)
    lag, z = correlate(a, b, SR)
    assert abs(lag - (5.55 - 30)) < 0.002
    assert z > 10


def test_short_overlap_still_found():
    music = synth_music(90, SR)
    a = phone(music, SR, 0, 40, seed=5)
    b = phone(music, SR, 32, 90, seed=6)  # 8 s overlap
    lag, z = correlate(a, b, SR)
    assert abs(lag - 32) < 0.002
    assert z > 8


def test_unrelated_audio_is_not_a_match():
    a = phone(synth_music(60, SR, seed=1), SR, 0, 60, seed=1)
    b = phone(synth_music(60, SR, seed=2), SR, 0, 60, seed=2)
    _, z = correlate(a, b, SR)
    assert z < 8


def test_a_real_overlap_holds_in_every_window_and_a_chance_peak_does_not():
    music = synth_music(90, SR)
    a = phone(music, SR, 0, 60, seed=1)
    b = phone(music, SR, 20.3, 80, seed=2, gain=3.0)
    lag, _ = correlate(a, b, SR)
    real = support(a, b, lag, SR)
    assert len(real) == 4 and min(real) > 3 * MIN_SUPPORT and holds(real)  # the same lag in every window of the overlap

    other = phone(synth_music(60, SR, seed=2), SR, 0, 60, seed=2)
    chance = support(a, other, 20.3, SR)  # some lag that is not one: the sound is not the same in any part of it
    assert max(chance) < MIN_SUPPORT and not holds(chance)


def test_a_match_that_holds_in_only_part_of_the_overlap_does_not_hold():
    # two recordings that are alike for part of the time (the same instruments, room and crowd, some of the same sound)
    # but not the same performance: what pairs of different songs of one concert are. The overlap cut in thirds let 75 to 82%
    # of such pairs through on real audio; a peak in most of the windows is what a real match has.
    first, second = synth_music(120, SR, seed=21), synth_music(120, SR, seed=22)
    shared = phone(first, SR, 0, 60, seed=1)  # 60 s of the same sound ...
    a = np.concatenate([shared, phone(first, SR, 60, 120, seed=2)])
    b = np.concatenate([phone(first, SR, 0, 60, seed=3), phone(second, SR, 60, 120, seed=4)])  # ... then another song
    scores = support(a, b, 0.0, SR)
    assert sum(v >= MIN_SUPPORT for v in scores) >= 0.4 * len(scores)  # it is there where the sound is the same ...
    assert not holds(scores)  # ... but that is not most of the overlap
    whole = support(a, np.concatenate([phone(first, SR, 0, 60, seed=3), phone(first, SR, 60, 120, seed=5)]), 0.0, SR)
    assert holds(whole) and SUPPORT_SHARE == 0.6


def test_support_at_the_wrong_lag_is_low_and_nothing_is_said_of_an_overlap_too_short_to_cut():
    music = synth_music(90, SR)
    a = phone(music, SR, 0, 60, seed=1)
    b = phone(music, SR, 20.3, 80, seed=2)
    assert not holds(support(a, b, 20.3 + 3.0, SR))  # 3 s out: no longer the same moment
    eight = support(phone(music, SR, 0, 40, seed=5), phone(music, SR, 32, 90, seed=6), 32.0, SR)
    assert len(eight) == 2 and min(eight) > MIN_SUPPORT and holds(eight)  # an 8 s overlap is still cut in two, and holds in both
    six_a, six_b = phone(music, SR, 0, 40, seed=5), phone(music, SR, 34.5, 90, seed=6)  # 5.5 s: too short to cut
    assert support(six_a, six_b, 34.5, SR) is None


def test_format_duration():
    from meld.sync import format_duration

    assert format_duration(4.4) == "4s"
    assert format_duration(59.6) == "1m 00s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3 * 3600 + 7 * 60 + 12) == "3h 07m"


def test_estimate_remaining_worst_case_and_edges():
    from meld.sync import estimate_remaining

    assert estimate_remaining([], [100], 5, 1.0) == (0.0, 0.0)  # nothing pending: nothing left

    # worst case: every pending clip uses the rest of its allowance (5 comparisons, minus those already made), each
    # comparison costing (the clip + the mean of the placed clips) samples at 2 s per sample
    expected, worst = estimate_remaining([(100, 1), (200, 0)], [300, 100], 5, 2.0)
    assert worst == (4 * (100 + 200) + 5 * (200 + 200)) * 2.0
    assert 0 < expected <= worst


def test_estimate_remaining_is_never_above_the_worst_case_or_below_zero():
    import numpy as np

    from meld.sync import estimate_remaining

    rng = np.random.default_rng(0)
    for _ in range(300):
        cap = int(rng.integers(1, 30))
        placed = [int(x) for x in rng.integers(1000, 10_000, int(rng.integers(1, 40)))]
        pending = [(int(rng.integers(1000, 10_000)), int(rng.integers(0, cap + 1))) for _ in range(int(rng.integers(1, 60)))]
        expected, worst = estimate_remaining(pending, placed, cap, 1e-6)
        assert 0 <= expected <= worst + 1e-9


def test_estimate_is_not_tiny_before_anything_is_known():
    from meld.sync import estimate_remaining

    # One clip placed and forty waiting, each allowed 25 comparisons. Every clip is compared against all the clips
    # placed by then, so a lot of work is coming. An estimate built from the first few placements (which needed one
    # or two comparisons each) said "about nothing" here.
    expected, worst = estimate_remaining([(1_000_000, 0)] * 40, [1_000_000], 25, 1e-6)
    assert expected > 0.3 * worst


def test_estimate_falls_as_comparisons_are_made_and_grows_with_clip_length():
    from meld.sync import estimate_remaining

    placed = [1_000_000] * 10
    fresh = estimate_remaining([(1_000_000, 0)] * 20, placed, 25, 1e-6)[0]
    worked = estimate_remaining([(1_000_000, 10)] * 20, placed, 25, 1e-6)[0]
    longer = estimate_remaining([(3_000_000, 0)] * 20, placed, 25, 1e-6)[0]
    assert worked < fresh < longer


def _run_matching_on_a_fake_clock(tmp_path, monkeypatch, members, strangers, cap, seed):
    """Run the real matching loop on synthetic clips, with a fake correlate and a fake clock that advances by an amount
    proportional to the clips' lengths (as a real FFT does). Returns [(fraction of the run done, seconds actually left,
    seconds the log line promised as "~expected", seconds it gave as "at most")] for every progress line."""
    import re

    import numpy as np
    from synth import write_wav

    from meld import sync as sync_mod
    from meld.project import Project

    clock = [0.0]

    class FakeTime:
        @staticmethod
        def perf_counter():
            return clock[0]

    def fake_correlate(a, b, fs, band=None, min_overlap=5.0):
        clock[0] += 1.5e-5 * (len(a) + len(b))
        both = float(np.abs(a).max()) < 0.4 and float(np.abs(b).max()) < 0.4  # loud clips are from another concert
        return 0.0, (40.0 if both else 1.0)

    monkeypatch.setattr(sync_mod, "time", FakeTime)
    monkeypatch.setattr(sync_mod, "correlate", fake_correlate)
    project = Project(tmp_path)
    rng = np.random.default_rng(seed)
    for i in range(members + strangers):
        seconds = 15.0 if i == 0 else float(rng.uniform(6, 14))  # clip 0: the longest, and a match, so it can grow
        amplitude = 0.2 if i < members else 0.6
        tone = (amplitude * np.sin(2 * np.pi * 440 * np.arange(int(seconds * 16000)) / 16000)).astype(np.float32)
        write_wav(project.clips_dir / f"{'m' if i < members else 'x'}{i:03d}.wav", tone, 16000)

    lines = []
    sync_mod.sync_project(project, max_compare=cap, log=lambda m="": lines.append((clock[0], str(m))))
    end = clock[0]

    def seconds(text):
        return sum(int(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in re.findall(r"(\d+)([hms])", text))

    rows = []
    for t, text in lines:
        m = re.search(r"\| ~(.+?) left \(at most (.+?)\)", text)
        if m:
            rows.append(((t) / end, end - t, seconds(m[1]), seconds(m[2])))
    return rows


def test_time_left_tracks_what_really_happens(tmp_path, monkeypatch):
    rows = _run_matching_on_a_fake_clock(tmp_path, monkeypatch, members=10, strangers=14, cap=10, seed=1)
    assert len(rows) > 50
    assert all(expected <= worst + 1 for _, _, expected, worst in rows)  # a ceiling is a ceiling
    early = [e / left for done, left, e, _ in rows if 0.05 <= done <= 0.85 and left > 20]
    assert early and all(0.6 <= ratio <= 1.7 for ratio in early), early  # the old estimate was 0.2x here


def test_estimate_learns_that_few_clips_match():
    from meld.sync import estimate_remaining

    # 20 clips waiting, room for 25 comparisons each. With 10 clips placed and the rest not yet tried, half are
    # expected to match, so each still faces ~25; once 19 of the 20 have been tried and only the reference is
    # placed, it is clear nearly nothing matches: there is nothing more to place, so little work is left.
    unsure = estimate_remaining([(1_000_000, 0)] * 20, [1_000_000] * 10, 25, 1e-6)[0]
    strangers = estimate_remaining([(1_000_000, 1)] * 19 + [(1_000_000, 0)], [1_000_000], 25, 1e-6)[0]
    assert strangers < unsure / 3
