"""Cutting the video to the music: how long shots are, where the cuts fall, and which angle is shown."""
import numpy as np
import pytest
from music import Section, render
from synth import make_clip, write_wav

from meld import beats as B
from meld import videocut as V
from meld.project import Project
from meld.timeline import OUT_FPS, ClipEntry, Timeline

MIN_D, MAX_D = V.MUSIC_MIN_SHOT, 12.0


def regular_map(seconds, bpm=120.0, energy=0.5, rise=0.0, first=0.3, start_phase=0):
    """A BeatMap of a steady beat. `bpm`, `energy` and `rise` may be numbers or functions of time (seconds)."""
    f = lambda v: v if callable(v) else (lambda t: v)  # noqa: E731
    bpm_f, en_f, rise_f = f(bpm), f(energy), f(rise)
    beats, t = [], first
    while t < seconds:
        beats.append(t)
        t += 60.0 / bpm_f(t)
    beats = np.array(beats)
    return B.BeatMap(
        beats=beats, period=np.array([60.0 / bpm_f(t) for t in beats]), phase=(np.arange(len(beats)) + start_phase) % 4,
        energy=np.array([en_f(t) for t in beats]), rise=np.array([rise_f(t) for t in beats]), confidence=1.0,
    )


def clips(*spans):
    return [ClipEntry(f"c{i}.mp4", a, b - a, True, 640, 360) for i, (a, b) in enumerate(spans)]


def quality(entries, seconds, per_clip=None):
    """S: one row per clip, 0.5 s steps; a clip is worth `per_clip[i]` (default 1) where it is there and 0 where not."""
    steps = int(seconds / V.GRID) + 4
    S = np.zeros((len(entries), steps), np.float32)
    for i, e in enumerate(entries):
        S[i, int(e.offset / V.GRID): int(np.ceil(e.end / V.GRID))] = (per_clip or {}).get(i, 1.0)
    return S


def plan(bm, seconds=60.0, spans=None, per_clip=None, pace=1.0, t0=0.0, min_shot=MIN_D):
    spans = spans or [(t0, t0 + seconds), (t0, t0 + seconds)]  # two cameras that filmed all of it
    entries = clips(*spans)
    S = quality(entries, seconds + t0, per_clip)
    shots = V.plan_music(S, entries, list(range(len(entries))), t0, t0 + seconds, bm, min_shot, MAX_D, pace, V.LEAD, 0.8)
    return [(a * V.GRID, b * V.GRID, c) for a, b, c in shots]


def per_minute(shots, lo, hi):
    n = sum(1 for a, b, c in shots if lo <= a < hi)
    return 60.0 * n / (hi - lo)


# ---- what length of shot the music asks for


def test_shots_get_shorter_as_the_music_gets_more_intense_and_as_it_builds():
    calm, mid, hard = (V.target_shot(e, 0.0, 120.0) for e in (0.0, 0.5, 1.0))
    assert calm == pytest.approx(V.SHOT_CALM) and hard == pytest.approx(V.SHOT_INTENSE)
    assert calm > mid > hard
    assert V.target_shot(0.5, 1.0, 120.0) < V.target_shot(0.5, 0.0, 120.0)  # building up: faster


def test_shots_get_shorter_at_a_faster_tempo_and_longer_at_a_slower_one():
    slow, neutral, fast = (V.target_shot(0.5, 0.0, b) for b in (80.0, 120.0, 170.0))
    assert slow > neutral > fast
    assert V.target_shot(0.5, 0.0, 400.0) == V.target_shot(0.5, 0.0, 200.0)  # a wild tempo estimate is not taken at its word
    assert V.target_shot(0.5, 0.0, 10.0) == V.target_shot(0.5, 0.0, 60.0)


def test_pace_scales_the_length():
    assert V.target_shot(0.5, 0.0, 120.0, pace=2.0) == pytest.approx(V.target_shot(0.5, 0.0, 120.0) / 2)


# ---- the cuts


def test_the_shots_tile_the_stretch_and_every_cut_is_just_before_a_beat():
    bm = regular_map(60.0, energy=0.5)
    shots = plan(bm, 60.0, t0=10.0)  # the stretch starts at 10 s on the timeline
    assert shots[0][0] == pytest.approx(10.0) and shots[-1][1] == pytest.approx(70.0)
    for (a, b, c), (a2, b2, c2) in zip(shots, shots[1:]):
        assert b == pytest.approx(a2)  # no gap, no overlap
    beats = bm.beats + 10.0
    for a, b, c in shots[:-1]:
        assert np.abs(b + V.LEAD - beats).min() < 1e-6  # the cut is the lead before a beat


def test_shots_are_never_shorter_or_longer_than_allowed():
    for energy in (0.0, 0.5, 1.0):
        shots = plan(regular_map(80.0, bpm=140, energy=energy), 80.0)
        d = np.array([b - a for a, b, c in shots])
        assert d[:-1].min() >= MIN_D - 1e-6 and d.max() <= MAX_D + 1e-6, energy


def test_intense_music_is_cut_much_faster_than_calm_music():
    bm = regular_map(90.0, energy=lambda t: 0.1 if t < 45 else 0.95)
    shots = plan(bm, 90.0)
    calm, intense = per_minute(shots, 0, 45), per_minute(shots, 45, 90)
    assert intense >= 2.2 * calm, (calm, intense)


def test_a_faster_tempo_is_cut_faster_at_the_same_intensity():
    bm = regular_map(90.0, bpm=lambda t: 80.0 if t < 45 else 170.0, energy=0.6)
    shots = plan(bm, 90.0)
    slow, fast = per_minute(shots, 0, 45), per_minute(shots, 45, 90)
    assert fast >= 1.4 * slow, (slow, fast)


def test_cutting_speeds_up_while_the_music_builds():
    flat = plan(regular_map(80.0, energy=0.5, rise=0.0), 80.0)
    building = plan(regular_map(80.0, energy=0.5, rise=lambda t: 1.0 if 30 <= t < 60 else 0.0), 80.0)
    assert per_minute(building, 30, 60) >= 1.3 * per_minute(flat, 30, 60)
    assert per_minute(building, 0, 30) == pytest.approx(per_minute(flat, 0, 30), rel=0.2)  # only where it builds


def test_pace_changes_how_often_it_cuts():
    bm = regular_map(90.0, energy=0.5)
    base, quick, slow = (len(plan(bm, 90.0, pace=p)) for p in (1.0, 2.0, 0.5))
    assert quick >= 1.5 * base and slow <= 0.7 * base


def test_cuts_fall_on_strong_beats_even_when_the_first_cut_lands_on_a_weak_one():
    for start_phase in (0, 1, 3):  # the stretch starts on a strong beat, or on a weak one, whichever
        bm = regular_map(90.0, energy=0.5, start_phase=start_phase)
        shots = plan(bm, 90.0)
        phase_at = {round(t - V.LEAD, 3): int(p) for t, p in zip(bm.beats, bm.phase)}
        ends = [phase_at[round(b, 3)] for a, b, c in shots[:-1]]
        strong = sum(p in (0, 2) for p in ends) / len(ends)
        assert strong >= 0.85, (start_phase, strong)


def test_the_same_shot_length_is_not_used_for_ever():
    bm = regular_map(120.0, energy=0.5)
    lengths = [round((b - a) / (60 / 120)) for a, b, c in plan(bm, 120.0)][:-1]
    runs = max(len(list(g)) for _, g in __import__("itertools").groupby(lengths))
    assert len(set(lengths)) >= 2 and runs <= 6


# ---- the pictures


def test_the_angle_changes_at_every_cut_when_there_is_another_to_show():
    shots = plan(regular_map(90.0, energy=0.7), 90.0, spans=[(0, 90), (0, 90), (0, 90)])
    assert all(x[2] != y[2] for x, y in zip(shots, shots[1:]))


def test_a_better_angle_is_shown_more_but_not_twice_running():
    shots = plan(regular_map(120.0, energy=0.7), 120.0, spans=[(0, 120)] * 3, per_clip={0: 4.0, 1: 1.0, 2: 1.0})
    share = [sum(1 for s in shots if s[2] == i) / len(shots) for i in range(3)]
    assert share[0] >= 0.4 and share[0] > max(share[1], share[2])
    assert all(x[2] != y[2] for x, y in zip(shots, shots[1:]))


def test_a_very_poor_angle_is_hardly_used():
    shots = plan(regular_map(120.0, energy=0.7), 120.0, spans=[(0, 120)] * 3, per_clip={0: 1.0, 1: 1.0, 2: 0.05})
    assert sum(1 for s in shots if s[2] == 2) / len(shots) <= 0.1


def test_a_shot_only_uses_a_clip_that_shows_all_of_it():
    spans = [(0.0, 40.0), (20.0, 60.0)]
    entries = clips(*spans)
    shots = plan(regular_map(60.0, energy=0.8), 60.0, spans=spans)
    for a, b, c in shots:
        assert c is not None and entries[c].offset <= a + 0.03 and entries[c].end >= b - 0.03, (a, b, c)
    assert all(c == 0 for a, b, c in shots if b <= 20.0) and all(c == 1 for a, b, c in shots if a >= 40.0)


def test_when_no_clip_shows_a_whole_beat_the_cut_goes_where_one_clip_ends():
    spans = [(0.0, 30.17), (30.17, 60.0)]  # one clip ends and the next begins between two beats: no clip has both sides
    shots = plan(regular_map(60.0, energy=0.5), 60.0, spans=spans)
    assert any(abs(b - 30.17) < 0.05 for a, b, c in shots)  # an off-beat cut, at the join
    entries = clips(*spans)
    assert all(entries[c].offset <= a + 0.05 and entries[c].end >= b - 0.05 for a, b, c in shots)


def test_a_stretch_that_no_clip_covers_is_shown_black_not_skipped():
    shots = plan(regular_map(40.0, energy=0.5), 40.0, spans=[(0.0, 20.0)])
    assert shots[-1][2] is None and shots[0][2] == 0
    assert shots[0][0] == pytest.approx(0.0) and shots[-1][1] == pytest.approx(40.0)


def test_it_is_the_same_every_time_it_is_asked():
    bm = regular_map(90.0, energy=lambda t: 0.3 + 0.5 * np.sin(t / 7) ** 2)
    assert plan(bm, 90.0, spans=[(0, 90)] * 3) == plan(bm, 90.0, spans=[(0, 90)] * 3)


def test_a_very_short_stretch_is_one_shot():
    shots = plan(regular_map(3.0, energy=0.9), 3.0, min_shot=2.5)
    assert shots[0][0] == pytest.approx(0.0) and shots[-1][1] == pytest.approx(3.0)
    assert 1 <= len(shots) <= 2


# ---- finding the beats of a whole video, and keeping them


def _project_with_audio(tmp_path, sections, seconds_of_clip=None):
    project = Project(tmp_path)
    music = render(sections, seed=1)
    path = project.out_dir / "fused_audio.wav"
    write_wav(path, music.audio, B.SR)
    seconds = len(music.audio) / B.SR
    tl = Timeline([ClipEntry("a.mp4", 0.0, seconds, True, 640, 360)])
    return project, tl, path, music


def test_the_beats_of_a_video_are_found_once_and_kept_until_the_audio_changes(tmp_path):
    project, tl, path, music = _project_with_audio(tmp_path, [Section(120, 16, "medium")])
    first, second, third = [], [], []
    maps1 = V.find_beats(project, tl, path, first.append)
    assert any("Finding the beats" in line for line in first) and maps1[0] is not None
    maps2 = V.find_beats(project, tl, path, second.append)
    assert any("what was found last time" in line for line in second)
    assert np.array_equal(maps1[0].beats, maps2[0].beats) and maps1[0].confidence == maps2[0].confidence
    write_wav(path, render([Section(150, 16, "medium")], seed=2).audio, B.SR)  # the audio is not the same any more
    maps3 = V.find_beats(project, tl, path, third.append)
    assert any("Finding the beats" in line for line in third)
    assert np.median(maps3[0].bpm) == pytest.approx(150, rel=0.05)


def test_a_stretch_with_no_beat_gives_none_and_is_cut_the_old_way(tmp_path):
    project, tl, path, _ = _project_with_audio(tmp_path, [Section(100, 12, "none")])
    (bm,) = V.find_beats(project, tl, path, lambda *_: None)
    assert bm is None or bm.confidence < V.MIN_CONFIDENCE


# ---- the whole thing: a real video, cut to real audio


def _cut_project(tmp_path):
    project = Project(tmp_path)
    # not 150 BPM hard: the tempo tracker mistakes dense hard grooves at 145 to 155 BPM for other tempi
    music = render([Section(120, 12, "medium"), Section(128, 10, "hard")], seed=2)
    seconds = len(music.audio) / B.SR
    cut = int(0.7 * seconds * B.SR)
    later = int(0.4 * seconds * B.SR)
    make_clip(project.clips_dir / "a.mp4", music.audio[:cut], B.SR, "testsrc2=size=320x180:rate=30")
    make_clip(project.clips_dir / "b.mp4", music.audio[later:], B.SR, "rgbtestsrc=size=320x180:rate=30")
    Timeline([
        ClipEntry("a.mp4", 0.0, cut / B.SR, True, 320, 180),
        ClipEntry("b.mp4", later / B.SR, (len(music.audio) - later) / B.SR, True, 320, 180),
    ]).save(project.timeline_path)
    return project, music


def _shot_lengths(project):
    from meld.media import probe

    files = [line.split("'")[1] for line in project.cache_dir.joinpath("shots.txt").read_text().splitlines()]
    return [probe(__import__("pathlib").Path(f)).duration for f in files]


def test_the_rendered_video_is_cut_on_the_beats_of_its_music(tmp_path):
    from meld.audiofuse import fuse_audio

    project, music = _cut_project(tmp_path)
    fuse_audio(project, log=lambda *_: None)
    lines = []
    out = V.render_video(project, size=(320, 180), log=lines.append)
    assert out.exists() and any("Cut to the music in 1 stretch" in line for line in lines)
    cuts = np.cumsum(_shot_lengths(project))[:-1]
    assert len(cuts) >= 6
    near = np.abs((cuts + V.LEAD)[:, None] - music.beats[None, :]).min(axis=1)
    assert (near <= 0.08).mean() >= 0.85  # within a frame or two of just before a beat
    lengths = np.diff(np.concatenate([[0.0], cuts]))
    assert lengths.min() >= MIN_D - 0.1


def test_it_can_be_told_not_to_cut_to_the_music(tmp_path):
    from meld.audiofuse import fuse_audio

    project, _ = _cut_project(tmp_path)
    fuse_audio(project, log=lambda *_: None)
    lines = []
    V.render_video(project, size=(320, 180), log=lines.append, beats=False)
    assert not any("Cut to the music" in line for line in lines)
    assert min(_shot_lengths(project)[:-1]) >= 2.9  # the old way: never shorter than 3 s


def test_the_command_line_can_turn_it_off_and_set_the_pace(tmp_path, monkeypatch):
    from meld import cli, videocut

    seen = []
    monkeypatch.setattr(videocut, "render_video", lambda *a, **k: seen.append((a, k)))
    cli.main(["video", "-p", str(tmp_path)])
    cli.main(["video", "-p", str(tmp_path), "--no-beats", "--cut-pace", "2", "--min-shot", "1.5"])
    (a1, k1), (a2, k2) = seen
    assert k1["beats"] is True and k1["pace"] == 1.0 and a1[2] is None  # by default: to the music, shortest shot left to it
    assert k2["beats"] is False and k2["pace"] == 2.0 and a2[2] == 1.5
