"""Beat tracking, judged against synthetic music whose true beats, bars and sections are known."""
import itertools

import numpy as np
import pytest
from music import Section, beats_found_near, beats_on_truth, render

from meld import beats as B


def find(sections, seed=1):
    music = render(sections, seed=seed)
    bm = B.beat_maps(music.audio, [(0.0, len(music.audio) / B.SR)], log=lambda *_: None)[0]
    return music, bm


def within(section_span, bm, margin=8.0):
    """The beats found in a section, leaving the first `margin` seconds for the tempo to settle after a change, and the
    last two, where the tempo is already changing to the next song's."""
    s0, s1, _ = section_span
    return (bm.beats >= s0 + margin) & (bm.beats < s1 - 2.0)


def test_a_steady_tempo_is_found_and_the_beats_land_on_the_beats():
    music, bm = find([Section(120, 20, "medium")])
    sel = bm.beats > 6
    assert np.median(bm.bpm[sel]) == pytest.approx(120, rel=0.03)
    assert beats_found_near(bm.beats, music.beats, after=6.0) >= 0.95  # every beat is found ...
    assert beats_on_truth(bm.beats[sel], music.beats) >= 0.95  # ... and the beats found are real
    assert len(bm.beats) == pytest.approx(len(music.beats), rel=0.05)
    assert bm.confidence > 0.9


def test_the_tempo_is_followed_through_a_change_of_song():
    music, bm = find([Section(100, 12, "medium"), Section(160, 20, "hard")])
    for span in music.sections:
        sel = within(span, bm)
        assert np.median(bm.bpm[sel]) == pytest.approx(span[2].bpm, rel=0.05), span[2]
        assert beats_on_truth(bm.beats[sel], music.beats) >= 0.9, span[2]


def test_a_slow_quiet_song_is_tracked_too():
    music, bm = find([Section(76, 14, "soft")])
    sel = bm.beats > 6
    ratio = np.median(bm.bpm[sel]) / 76
    assert min(abs(ratio - 1.0), abs(ratio - 0.5)) < 0.06  # the tempo, or every other beat of it
    assert beats_on_truth(bm.beats[sel], music.beats) >= 0.85


def _many_songs(seed=5):
    """(tempo, style, music, beat map) for songs of every style at tempi from slow to fast."""
    for bpm, style in itertools.product((76, 90, 100, 112, 120, 128, 140, 152, 160, 172), ("soft", "medium", "hard")):
        music, bm = find([Section(bpm, max(14, int(30 * bpm / 240)), style)], seed=seed)
        yield bpm, style, music, bm


def test_across_many_songs_the_tempo_is_right_or_an_octave_out_and_the_beats_are_on_real_beats():
    songs = list(_many_songs())
    ok_tempo = precision = 0
    for bpm, style, music, bm in songs:
        sel = bm.beats > 8
        ratio = np.median(bm.bpm[sel]) / bpm
        ok_tempo += min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.06
        # a beat is right if it is on a beat or on the half beat between (an eighth note grid has both)
        halves = np.sort(np.concatenate([music.beats, music.beats + 30.0 / bpm]))
        precision += beats_on_truth(bm.beats[sel], halves) >= 0.8
    assert ok_tempo >= 0.9 * len(songs) and precision >= 0.85 * len(songs), (ok_tempo, precision, len(songs))


def _beats_and_low(strong, weak, count=32, gap=25):
    beats = np.arange(count) * float(gap) + 30.0
    low = np.zeros(int(beats[-1]) + 60, np.float32)
    for i, b in enumerate(beats.astype(int)):
        low[b] = strong if i % 2 == 0 else weak
    return beats, low


def test_off_beats_are_dropped_only_when_every_other_beat_has_no_bass():
    beats, low = _beats_and_low(strong=8.0, weak=0.1)
    assert list(B.drop_offbeats(beats, low)) == list(beats[0::2])  # eighth notes: the bass only plays on the beat
    beats, low = _beats_and_low(strong=8.0, weak=0.1)
    beats = beats[1:]  # the same, but the first beat given is an off-beat: the other half is kept
    assert list(B.drop_offbeats(beats, low)) == list(beats[1::2])


def test_a_real_beat_grid_is_left_alone_even_if_kick_and_snare_differ_a_little():
    beats, low = _beats_and_low(strong=8.0, weak=5.5)  # kick against snare: about 1.5 times, not 4
    assert list(B.drop_offbeats(beats, low)) == list(beats)
    beats, low = _beats_and_low(strong=8.0, weak=8.0)
    assert list(B.drop_offbeats(beats, low)) == list(beats)


def test_off_beats_are_not_dropped_on_a_short_run():
    beats, low = _beats_and_low(strong=8.0, weak=0.1, count=12)
    assert list(B.drop_offbeats(beats, low)) == list(beats)  # too few to be sure


def test_the_correction_is_made_song_by_song_within_one_stretch():
    # 48 beats of eighth notes (bass only on alternate beats), then 48 of a right beat grid (bass on every beat)
    beats, low = _beats_and_low(strong=8.0, weak=0.1, count=96)
    for b in beats[48:].astype(int):
        low[b] = 8.0
    kept = B.drop_offbeats(beats, low)
    first = kept[kept < beats[48]]
    second = kept[kept >= beats[48]]
    assert 20 <= len(first) <= 28  # about half of the first song's 48 beats
    assert len(second) >= 44  # the second song keeps (nearly) all of its 48: it was never in doubt


def test_bars_start_on_strong_beats_in_most_songs_of_every_style():
    good = {"soft": 0, "medium": 0, "hard": 0}
    seen = {"soft": 0, "medium": 0, "hard": 0}
    for bpm, style, music, bm in _many_songs(seed=4):
        t = bm.beats[(bm.phase == 0) & (bm.beats > 8)]
        seen[style] += 1
        good[style] += len(t) > 3 and (np.abs(t[:, None] - music.beats[::2][None, :]).min(axis=1) <= 0.05).mean() >= 0.75
    # beats one and three are alike to the ear, so 'a strong beat' is what counts. Rock is the case that matters: the loud
    # snare on two and four must not be taken for the bar start, which going by loudness alone it is.
    assert all(good[k] >= 0.75 * seen[k] for k in seen), (good, seen)


def test_energy_is_high_where_the_music_is_hard_and_low_where_it_is_soft():
    music, bm = find([Section(100, 10, "soft"), Section(128, 14, "hard"), Section(100, 10, "soft")])
    soft1, hard, soft2 = (bm.energy[within(sp, bm, margin=3.0)].mean() for sp in music.sections)
    assert hard > soft1 + 0.3 and hard > soft2 + 0.3
    assert 0.0 <= bm.energy.min() and bm.energy.max() <= 1.0


def test_a_build_up_is_seen_as_rising():
    music = render([Section(120, 30, "medium")], seed=2)
    n = len(music.audio)
    gain = np.ones(n, np.float32)
    ramp = slice(int(14 * B.SR), int(34 * B.SR))
    gain[: ramp.start] = 0.25
    gain[ramp] = np.linspace(0.25, 1.0, ramp.stop - ramp.start)
    bm = B.beat_maps(music.audio * gain, [(0.0, n / B.SR)], log=lambda *_: None)[0]
    building = (bm.beats > 18) & (bm.beats < 32)
    steady = (bm.beats > 8) & (bm.beats < 13)
    assert bm.rise[building].mean() > bm.rise[steady].mean() + 0.2
    assert 0.0 <= bm.rise.min() and bm.rise.max() <= 1.0


def test_something_with_no_rhythm_has_little_confidence_and_music_has_a_lot():
    _, drone = find([Section(100, 20, "none")])
    _, music = find([Section(100, 20, "medium")])
    assert drone.confidence < 0.25 and music.confidence > 0.5


def test_random_clicks_and_talk_are_not_taken_for_music():
    rng = np.random.default_rng(3)
    n = 30 * B.SR
    x = 0.02 * rng.standard_normal(n).astype(np.float32)
    for at in rng.uniform(0, 30, 90):  # a click about three times a second, at random
        i = int(at * B.SR)
        x[i:i + 300] += rng.standard_normal(min(300, n - i)).astype(np.float32) * 0.6 * np.exp(-np.arange(min(300, n - i)) / 60)
    bm = B.beat_maps(x, [(0.0, 30.0)], log=lambda *_: None)[0]
    assert bm is None or bm.confidence < 0.25


def test_a_stretch_that_is_too_short_or_silent_gives_nothing_to_cut_to():
    music = render([Section(120, 20, "medium")], seed=1)
    assert B.beat_maps(music.audio, [(0.0, 4.0)], log=lambda *_: None) == [None]
    silent = B.beat_maps(np.zeros(20 * B.SR, np.float32), [(0.0, 20.0)], log=lambda *_: None)[0]
    assert silent is None or silent.confidence < 0.25


def test_each_stretch_has_its_own_beats_counted_from_its_own_start():
    music = render([Section(120, 30, "medium")], seed=6)
    maps = B.beat_maps(music.audio, [(0.0, 20.0), (30.0, 20.0)], log=lambda *_: None)
    assert all(m is not None for m in maps)
    for m in maps:
        assert m.beats.min() >= 0 and m.beats.max() <= 20.5  # times from the start of the stretch
    second = maps[1].beats + 30.0
    assert beats_found_near(second, music.beats[(music.beats > 34) & (music.beats < 48)]) >= 0.9


@pytest.mark.parametrize("count", [0, 1, 3, 25, 33, 200])
def test_bar_positions_can_be_worked_out_for_any_number_of_beats(count):
    # a stretch of 25 beats used to fail here: the smoothing window was longer than the beats
    rng = np.random.default_rng(count)
    phase = B.bar_phase(np.arange(count) * 25.0, rng.random(count))
    assert len(phase) == count and set(phase.tolist()) <= {0, 1, 2, 3}


def test_a_long_piece_is_analysed_in_pieces_without_changing_the_answer():
    music = render([Section(120, 16, "medium")], seed=8)
    whole = B.analyse(music.audio, chunk_frames=100000)
    pieces = B.analyse(music.audio, chunk_frames=700)  # many chunk boundaries
    for a, b in zip(whole, pieces):
        assert np.allclose(a, b, atol=1e-3)


def test_the_tempo_and_its_clarity_are_given_for_every_frame():
    music = render([Section(120, 12, "medium")], seed=1)
    onset = B.analyse(music.audio)[0]
    tempo, clarity = B.local_tempo(onset)
    assert len(tempo) == len(clarity) == len(onset)
    assert B.BPM_MIN <= tempo.min() and tempo.max() <= B.BPM_MAX
    assert 0.0 <= clarity.min() and clarity.max() <= 1.0
