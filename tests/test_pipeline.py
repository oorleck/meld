import json

from synth import make_clip, phone, synth_music

from meld.audiofuse import fuse_audio
from meld.media import probe
from meld.project import Project
from meld.sync import sync_project
from meld.videocut import render_video

SR = 44100


def test_end_to_end(tmp_path):
    project = Project(tmp_path)
    music = synth_music(100, SR)
    make_clip(project.clips_dir / "a.mp4", phone(music, SR, 0, 60, seed=1), SR, "testsrc2=size=640x360:rate=30")
    make_clip(project.clips_dir / "b.mp4", phone(music, SR, 30, 100, seed=2), SR, "rgbtestsrc=size=640x360:rate=30")
    # vertical, badly clipped
    make_clip(project.clips_dir / "c.mp4", phone(music, SR, 55, 85, seed=3, gain=6), SR, "testsrc2=size=360x640:rate=30")
    # different music entirely: must be rejected
    other = synth_music(30, SR, seed=99)
    make_clip(project.clips_dir / "d.mp4", phone(other, SR, 0, 30, seed=4), SR, "testsrc2=size=640x360:rate=30")

    tl = sync_project(project, log=lambda *_: None)
    off = {c.file: c.offset for c in tl.clips}
    assert set(off) == {"a.mp4", "b.mp4", "c.mp4"}
    assert abs((off["b.mp4"] - off["a.mp4"]) - 30) < 0.005
    assert abs((off["c.mp4"] - off["a.mp4"]) - 55) < 0.005
    assert [r["file"] for r in tl.rejected] == ["d.mp4"]
    assert json.loads(project.timeline_path.read_text())["clips"]

    fuse_audio(project, log=lambda *_: None)
    render_video(project, size=(640, 360), log=lambda *_: None)

    info = probe(project.out_dir / "fused.mp4")
    assert info.has_video and info.has_audio
    assert abs(info.duration - 100) < 0.3
    assert (info.width, info.height) == (640, 360)
