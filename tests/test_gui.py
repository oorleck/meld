"""Tests for the GUI's logic. The window itself needs a display; the pipeline behind it does not."""
from pathlib import Path

import pytest
from synth import make_clip, phone, synth_music

from meld import gui
from meld.gui import Cancelled, Settings, UserError, friendly_error, parse_progress, run_pipeline

SR = 44100


def test_settings_derive_folder_size_and_disk_estimate(tmp_path):
    s = Settings("Metallica 2003 (Paris!)", tmp_path, clips=100, quality=1080)
    assert s.project_dir == tmp_path / "metallica-2003-paris"
    assert s.size == (1920, 1080)
    assert Settings("x", tmp_path, quality=720).size == (1280, 720)
    assert Settings("x", tmp_path, clips=500, quality=1080).disk_needed_gb() > Settings("x", tmp_path, clips=20, quality=720).disk_needed_gb()


@pytest.mark.parametrize("line, expected", [
    ("[3/40] downloaded Metallica - Fuel (Live 2003)", (3, 40)),
    ("[7/40] FAILED something (unavailable)", (7, 40)),
    ("  E4n96UcW1ZQ.mp4 vs b.mp4: z=20.1 | 9/15 placed, 38 comparisons, 12s elapsed", (9, 15)),
    ("  50/120", (50, 120)),
    ("Search 'metallica 2003': kept 12, dropped 3 as not matching", None),
    ("  0.5s shown from 6CfgAaBUjCA.mp4", None),
])
def test_parse_progress(line, expected):
    assert parse_progress(line) == expected


def test_friendly_errors():
    assert friendly_error(UserError("plain words")) == "plain words"
    assert friendly_error(SystemExit("No media files")) == "No media files"
    assert "internet" in friendly_error(OSError("<urlopen error [Errno 11001] getaddrinfo failed>"))
    assert friendly_error(ValueError("boom")).startswith("Something went wrong")


def _make_project_clips(s: Settings) -> None:
    clips = s.project_dir / "clips"
    clips.mkdir(parents=True)
    music = synth_music(70, SR)
    make_clip(clips / "a.mp4", phone(music, SR, 0, 45, seed=1), SR, "testsrc2=size=640x360:rate=30")
    make_clip(clips / "b.mp4", phone(music, SR, 25, 70, seed=2), SR, "rgbtestsrc=size=640x360:rate=30")


def test_pipeline_runs_all_four_stages_and_reports(tmp_path, monkeypatch):
    s = Settings("test show 2024", tmp_path, clips=20, quality=720)
    _make_project_clips(s)
    monkeypatch.setattr(gui, "fetch", lambda *a, **k: None)  # no internet: the clips are already there
    stages = []
    result = run_pipeline(s, log=lambda *_: None, stage=lambda i, name: stages.append((i, name)))

    assert [i for i, _ in stages] == [0, 1, 2, 3]
    assert result["video"].name == "fused.mp4" and result["video"].exists()
    assert result["used"] == 2 and result["skipped"] == 0
    assert 1.0 < result["minutes"] < 1.4  # the two clips together cover 70 s


def test_pipeline_with_no_clips_explains_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "fetch", lambda *a, **k: None)
    with pytest.raises(UserError, match="couldn't find"):
        run_pipeline(Settings("nothing here", tmp_path), log=lambda *_: None)


def test_cancel_unwinds_from_the_log_callback(tmp_path, monkeypatch):
    s = Settings("cancel me", tmp_path)
    _make_project_clips(s)
    monkeypatch.setattr(gui, "fetch", lambda *a, **k: None)
    calls = {"n": 0}

    def log(msg=""):
        calls["n"] += 1
        if calls["n"] > 3:
            raise Cancelled()

    with pytest.raises(Cancelled):
        run_pipeline(s, log=log)
    assert not (s.project_dir / "out" / "fused.mp4").exists()


def test_shorten_path_keeps_both_ends():
    from meld.gui import shorten_path

    short = "C:/Users/me/Documents/Meld"
    assert shorten_path(short) == short
    long = "C:/Users/someone/AppData/Local/Temp/a-very-long-folder-name/another-long-one/results"
    out = shorten_path(long)
    assert len(out) <= 56 and out.startswith(long[:18]) and out.endswith(long[-10:]) and " ... " in out
