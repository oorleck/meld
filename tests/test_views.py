"""Which phones can see the same thing (views.py): synthetic scenes, so that what is right is known."""
import numpy as np
import pytest
from PIL import Image

from meld import views
from meld.media import run_ffmpeg
from meld.project import Project
from meld.timeline import ClipEntry, Timeline


def scene(seed: int, size: int = 900) -> Image.Image:
    """Texture at every scale, as a real scene has: features that can be found again from another place."""
    r = np.random.default_rng(seed)
    coarse = Image.fromarray((r.random((size // 8, size // 8, 3)) * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return Image.fromarray((0.7 * np.asarray(coarse).astype(float) + 0.3 * r.random((size, size, 3)) * 255).astype(np.uint8))


def view(img: Image.Image, shift=(0, 0), zoom=1.0, out=(640, 360)) -> Image.Image:
    """What a phone at another place sees of the scene: a window of it, moved and zoomed."""
    w, h = img.size
    cw, ch = out[0] / zoom, out[1] / zoom
    x0, y0 = (w - cw) / 2 + shift[0], (h - ch) / 2 + shift[1]
    return img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize(out, Image.BICUBIC)


def still_video(path, image: Image.Image, seconds: float = 8.0) -> None:
    png = path.with_suffix(".png")
    image.save(png)
    run_ffmpeg(["-loop", "1", "-i", png, "-t", seconds, "-r", 30, "-pix_fmt", "yuv420p", path])
    png.unlink()


A, B = scene(1), scene(2)


# ---- pairs of frames

def test_views_of_one_scene_have_matches_and_a_view_of_another_has_none(tmp_path):
    frames = []
    for k, img in enumerate([view(A), view(A, (60, 20), 1.1), view(A, (-80, 10), 1.25), view(B)]):
        f = tmp_path / f"f{k}.jpg"
        img.save(f, quality=90)
        frames.append(f)
    pairs = views.pair_matches(frames)
    assert pairs[(0, 1)] > 200 and pairs[(0, 2)] > 200 and pairs[(1, 2)] > 200  # the same scene, from other places
    assert pairs[(0, 3)] == pairs[(1, 3)] == pairs[(2, 3)] == 0  # another scene
    assert set(pairs) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


def test_a_view_that_hardly_overlaps_is_weaker_than_one_that_does(tmp_path):
    frames = []
    for k, img in enumerate([view(A), view(A, (40, 10)), view(A, (330, 200))]):
        f = tmp_path / f"f{k}.jpg"
        img.save(f, quality=90)
        frames.append(f)
    pairs = views.pair_matches(frames)
    assert pairs[(0, 1)] > pairs[(0, 2)] >= 0 and pairs[(0, 1)] > 400


# ---- sets of views

def test_connected_sets_join_views_through_others_and_leave_the_rest_alone():
    pairs = {(0, 1): 300, (1, 2): 40, (0, 2): 0, (3, 4): 500, (2, 3): 10, (0, 5): 24}
    sets = views.connected_sets(6, pairs, min_matches=25)
    assert sets == [[0, 1, 2], [3, 4], [5]]  # 2 is joined to 1, not to 0; 5 (24 matches) and 3-2 (10) are not enough
    assert views.connected_sets(3, {}, 25) == [[0], [1], [2]]
    assert views.connected_sets(0, {}, 25) == []


# ---- moments in a timeline

@pytest.fixture
def show(tmp_path):
    """Phones 0, 1 and 2 film scene A from three places, phone 3 scene B, all 8 s long; phone 4 films A, but from 20 s on."""
    project = Project(tmp_path)
    looks = [view(A), view(A, (50, 20), 1.1), view(A, (-70, 10), 1.2), view(B), view(A, (10, -30), 1.05)]
    starts = [0.0, 0.0, 0.0, 0.0, 20.0]
    entries = []
    for k, (img, at) in enumerate(zip(looks, starts)):
        still_video(project.clips_dir / f"p{k}.mp4", img)
        entries.append(ClipEntry(f"p{k}.mp4", at, 8.0, True, 640, 360))
    return project, Timeline(entries)


def test_the_phones_that_see_the_same_thing_are_found_and_the_one_that_does_not_is_left_out(show):
    project, tl = show
    m = views.look_at(project, tl, 4.0)
    assert m.clips == [0, 1, 2, 3] and m.best == [0, 1, 2] and m.sets == [[0, 1, 2], [3]]
    assert m.strength > 100  # they are held together by a good many matches


def test_a_moment_with_too_few_phones_says_so(show):
    project, tl = show
    m = views.look_at(project, tl, 24.0)  # only phone 4 is filming
    assert m.clips == [4] and m.best == [4] and m.strength == 0 and len(m.best) < views.MIN_VIEWS


def test_the_moments_are_listed_at_which_enough_phones_are_filming(show):
    project, tl = show
    assert views.candidate_times(tl, step=2.0) == [2.0, 4.0, 6.0]  # four phones for the first 8 s (not at 0 s: the margin); then one
    assert views.candidate_times(tl, step=2.0, at_least=5) == []


def test_a_scan_puts_the_moment_with_the_most_phones_that_see_the_same_thing_first(tmp_path):
    project = Project(tmp_path)
    looks = [view(A), view(A, (50, 20), 1.1), view(B), view(A, (-70, 10), 1.2)]
    starts = [0.0, 0.0, 0.0, 10.0]  # phone 3 joins scene A at 10 s, while the others still film (to 12 s): then three see the same
    entries = []
    for k, (img, at) in enumerate(zip(looks, starts)):
        still_video(project.clips_dir / f"p{k}.mp4", img, seconds=12.0)
        entries.append(ClipEntry(f"p{k}.mp4", at, 12.0, True, 640, 360))
    tl = Timeline(entries)
    lines = []
    moments = views.scan(project, tl, [3.0, 11.0], log=lines.append)
    assert [m.t for m in moments] == [11.0, 3.0] and len(moments[0].best) == 3 and len(moments[1].best) == 2
    assert len(lines) == 2 and "see the same thing" in lines[0]


# ---- reconstruct: which phones it hands to the model (the model itself is stood in for)

@pytest.fixture
def modelled(monkeypatch):
    from meld import recon

    seen = []
    monkeypatch.setattr(recon, "reconstruct_frames", lambda frames, names, out, *a, **k: seen.append(names) or out)
    return recon, seen


def test_reconstruct_gives_the_model_only_the_phones_that_see_the_same_thing(show, modelled):
    recon, seen = modelled
    project, tl = show
    tl.save(project.timeline_path)
    recon.reconstruct(project, at=4.0, video=False, log=lambda *_: None)
    assert seen == [["p0.mp4", "p1.mp4", "p2.mp4"]]  # not p3, which looks at something else


def test_reconstruct_says_why_it_will_not_when_too_few_phones_see_the_same_thing(show, modelled):
    recon, seen = modelled
    project, tl = show
    tl.save(project.timeline_path)
    with pytest.raises(SystemExit, match="--scan"):
        recon.reconstruct(project, at=24.0, video=False, log=lambda *_: None)
    assert seen == []
    recon.reconstruct(project, at=24.0, video=False, force=True, log=lambda *_: None)  # unless told to go on anyway
    assert seen == [["p4.mp4"]]


def test_reconstruct_finds_the_moment_itself_when_none_is_given(tmp_path, modelled):
    recon, seen = modelled
    project = Project(tmp_path)
    looks = [view(A), view(A, (50, 20), 1.1), view(A, (-70, 10), 1.2), view(B)]
    entries = []
    for k, img in enumerate(looks):
        still_video(project.clips_dir / f"p{k}.mp4", img, seconds=30.0)
        entries.append(ClipEntry(f"p{k}.mp4", 0.0, 30.0, True, 640, 360))
    Timeline(entries).save(project.timeline_path)
    lines = []
    recon.reconstruct(project, video=False, log=lines.append)
    assert seen == [["p0.mp4", "p1.mp4", "p2.mp4"]]
    assert any("Looking at which phones see the same thing" in line for line in lines)


def test_scan_lists_the_moments_and_does_not_run_the_model(show, modelled, monkeypatch):
    recon, seen = modelled
    project, tl = show
    tl.save(project.timeline_path)
    monkeypatch.setattr(recon, "SCAN_STEP", 2.0)  # the clips here are 8 s long
    lines = []
    assert recon.reconstruct(project, scan=True, log=lines.append) is None
    assert seen == [] and any("most promising" in line for line in lines)
