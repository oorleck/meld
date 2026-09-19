"""A video that is one picture held while the sound plays (cover art, a slideshow) is not an angle to cut to, however sharp."""
import re

import numpy as np
import pytest
from music import Section, render
from synth import make_clip, write_wav

from meld import videocut as V
from meld.audiofuse import fuse_audio
from meld.media import run_ffmpeg
from meld.project import Project
from meld.timeline import ClipEntry, Timeline

SR = 22050
MOVING = "testsrc2=size=320x180:rate=30"  # a picture that changes all the time (made a little blurred: see below)


def moving_clip(path, audio) -> None:
    make_clip(path, audio, SR, MOVING, vf="boxblur=3")  # a phone: not the sharpest picture there is


def still_clip(path, audio) -> None:
    """One picture, sharper than that of the phone, held while the sound plays: what an upload of cover art is."""
    png = path.with_suffix(".png")
    run_ffmpeg(["-f", "lavfi", "-i", f"{MOVING}:duration=4", "-ss", "3", "-frames:v", "1", png])
    wav = path.with_suffix(".tmp.wav")
    write_wav(wav, audio, SR)
    run_ffmpeg(["-loop", "1", "-i", png, "-i", wav, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", path])
    wav.unlink()
    png.unlink()


# ---- what a score is worth for how much the picture changes

def test_a_still_is_worth_the_floor_a_moving_picture_all_and_in_between_in_step():
    f = V.liveness(np.array([0.0, 0.003, V.STATIC_MOTION, 1.9, V.LIVE_MOTION, 30.0] * 9))
    per = f[:6]  # (the middle of the array, where the median has nothing of the edges in it)
    assert per[0] == per[1] == per[2] == pytest.approx(V.STATIC_FLOOR)
    assert 0 < V.LIVE_MOTION and V.STATIC_FLOOR < 1
    # steady values, one after the other, each held: below the still limit, in between, and above the live limit
    for motion, expected in ((0.0, V.STATIC_FLOOR), (V.STATIC_MOTION, V.STATIC_FLOOR), (30.0, 1.0), (V.LIVE_MOTION, 1.0)):
        assert V.liveness(np.full(40, motion))[20] == pytest.approx(expected)
    mid = V.liveness(np.full(40, (V.STATIC_MOTION + V.LIVE_MOTION) / 2))[20]
    assert V.STATIC_FLOOR < mid < 1.0
    assert len(V.liveness(np.zeros(0))) == 0


def test_a_slideshow_is_a_still_all_through_though_the_picture_jumps_now_and_then():
    motion = np.zeros(60)
    motion[5::10] = 30.0  # a new picture every 5 s: a jump each tenth step, nothing between
    assert np.allclose(V.liveness(motion), V.STATIC_FLOOR)


def test_the_limits_leave_room_for_real_footage_and_for_a_picture_that_is_held():
    # measured: real phone clips changed 22 to 38 (median) and never under 11; a held picture 0.003
    assert V.STATIC_MOTION < 11 and V.LIVE_MOTION < 11 and V.STATIC_MOTION > 0.1


# ---- clips

@pytest.fixture
def clips(tmp_path):
    project = Project(tmp_path)
    audio = np.random.default_rng(0).standard_normal(SR * 20).astype(np.float32) * 0.1
    moving_clip(project.clips_dir / "moving.mp4", audio)
    still_clip(project.clips_dir / "still.mp4", audio)
    entry = lambda name: ClipEntry(name, 0.0, 20.0, True, 320, 180)
    return project, entry("moving.mp4"), entry("still.mp4")


def test_a_held_picture_does_not_change_and_a_moving_one_does(clips):
    project, moving, still = clips
    _, m_moving = V.analyse_clip(project, moving)
    _, m_still = V.analyse_clip(project, still)
    assert np.median(m_still) < 0.2 and np.median(m_moving) > V.LIVE_MOTION
    assert V.still_share(project, still) == 1.0 and V.still_share(project, moving) == 0.0


def test_the_score_of_a_still_is_a_tenth_of_its_quality_and_that_of_a_moving_clip_is_all_of_it(clips):
    project, moving, still = clips
    q_still, _ = V.analyse_clip(project, still)
    q_moving, _ = V.analyse_clip(project, moving)
    assert np.allclose(V.score_clip(project, still), q_still * V.STATIC_FLOOR)
    assert np.allclose(V.score_clip(project, moving), q_moving)


def test_what_was_worked_out_of_a_clip_is_kept_and_used_again(clips, monkeypatch):
    project, moving, _ = clips
    first = V.analyse_clip(project, moving)
    assert (project.cache_dir / "vscore" / "moving.mp4.v2.npz").exists()

    def no_decoding(*a, **k):
        raise AssertionError("the video was decoded again")

    monkeypatch.setattr(V, "iter_gray_frames", no_decoding)
    again = V.analyse_clip(project, moving)
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])


# ---- the choice of angle

def _show(tmp_path, moving_seconds):
    """A concert of 40 s. A phone that is really there for the first `moving_seconds`, and a still picture (sharp: as good
    as any) for all of it."""
    project = Project(tmp_path)
    music = render([Section(120, 20, "medium")], seed=1)
    audio = music.audio
    seconds = len(audio) / SR
    still_clip(project.clips_dir / "still.mp4", audio)
    cut = audio[: int(moving_seconds * SR)]
    moving_clip(project.clips_dir / "moving.mp4", cut)
    Timeline([
        ClipEntry("still.mp4", 0.0, seconds, True, 320, 180),
        ClipEntry("moving.mp4", 0.0, len(cut) / SR, True, 320, 180),
    ]).save(project.timeline_path)
    return project, seconds


def _seconds_shown(lines):
    out = {}
    for line in lines:
        m = re.match(r"\s+([\d.]+)s shown from (\S+)", line)
        if m:
            out[m[2]] = float(m[1])
    return out


def test_a_still_picture_is_not_cut_to_while_a_phone_is_filming_but_is_where_it_is_all_there_is(tmp_path):
    project, seconds = _show(tmp_path, moving_seconds=28.0)
    fuse_audio(project, log=lambda *_: None)
    lines = []
    V.render_video(project, size=(320, 180), log=lines.append)
    shown = _seconds_shown(lines)
    assert any("still picture" in line and "still.mp4" in line for line in lines)  # it says so
    q_still = float(np.mean(V.analyse_clip(project, ClipEntry("still.mp4", 0.0, seconds, True, 320, 180))[0]))
    q_moving = float(np.mean(V.analyse_clip(project, ClipEntry("moving.mp4", 0.0, 28.0, True, 320, 180))[0]))
    assert q_still > q_moving  # the still is the better picture: by quality alone it would win
    assert shown["moving.mp4"] > 27.0  # the phone gets all of the time it is there for (28 s) ...
    assert shown["still.mp4"] == pytest.approx(seconds - 28.0, abs=1.0)  # ... and the still only what is left: the last 13 s,
    # where nothing else is filming and it is what there is


def test_the_picture_of_a_phone_that_is_there_is_chosen_for_the_3d_model_over_a_still_that_is_sharper(clips):
    from meld import recon

    project, moving, still = clips
    tl = Timeline([still, moving])
    q_still = float(np.mean(V.analyse_clip(project, still)[0]))
    q_moving = float(np.mean(V.analyse_clip(project, moving)[0]))
    assert q_still > q_moving  # as sharp and as well exposed: better, if that were all that counted
    assert recon.select_views(project, tl, 10.0, max_views=1) == [1]  # the moving one, not the still
