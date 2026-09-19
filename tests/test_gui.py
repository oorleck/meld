"""Tests for the GUI's logic. The window itself needs a display; the pipeline behind it does not."""
import json
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
    assert Settings("x", tmp_path, quality=480).size == (854, 480)
    assert Settings("x", tmp_path, quality=360).size == (854, 480)  # below the smallest still gets the smallest
    assert Settings("x", tmp_path, clips=500, quality=1080).disk_needed_gb() > Settings("x", tmp_path, clips=20, quality=720).disk_needed_gb()
    disk = [Settings("x", tmp_path, clips=100, quality=q).disk_needed_gb() for q in (1080, 720, 480)]
    assert disk[0] > disk[1] > disk[2]  # each step down needs less room


def test_every_quality_choice_has_a_size_and_a_download_estimate():
    for _, height in gui.QUALITY_CHOICES:
        assert height in gui.OUTPUT_SIZES and height in gui.MB_PER_CLIP
        assert gui.OUTPUT_SIZES[height][1] == height and gui.OUTPUT_SIZES[height][0] % 2 == 0  # even width for H.264
    assert [h for _, h in gui.QUALITY_CHOICES] == sorted((h for _, h in gui.QUALITY_CHOICES), reverse=True)


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
    # no video titles are known here, so the files are named after the search; they go straight into the save folder
    assert result["video"] == tmp_path / "Test Show 2024.mp4" and result["video"].exists()
    assert result["audio"] == tmp_path / "Test Show 2024.wav" and result["audio"].exists()
    assert result["folder"] == tmp_path
    assert not (s.project_dir / "out" / "fused.mp4").exists()  # moved, not copied
    assert result["used"] == 2 and result["skipped"] == 0
    assert 1.0 < result["minutes"] < 1.4  # the two clips together cover 70 s


def test_smallest_quality_caps_the_download_and_sizes_the_video(tmp_path, monkeypatch):
    from meld.media import probe

    s = Settings("small show 2024", tmp_path, clips=20, quality=480)
    _make_project_clips(s)
    asked = {}
    monkeypatch.setattr(gui, "fetch", lambda *a, **k: asked.update(k))
    result = run_pipeline(s, log=lambda *_: None)
    assert asked["max_height"] == 480  # what YouTube is asked for
    info = probe(result["video"])
    assert (info.width, info.height) == (854, 480)  # what comes out


def test_pipeline_names_the_files_after_the_videos_it_used(tmp_path, monkeypatch):
    s = Settings("test band 2024", tmp_path, clips=20, quality=720)
    _make_project_clips(s)
    (s.project_dir / "sources.json").write_text(json.dumps({
        "a": {"title": "Test Band - Song One (Live at Big Arena 2024) 4K"},
        "b": {"title": "TEST BAND Big Arena 2024 - Song Two"},
    }), encoding="utf-8")
    monkeypatch.setattr(gui, "fetch", lambda *a, **k: None)
    first = run_pipeline(s, log=lambda *_: None)
    assert first["video"].name == "Test Band Big Arena 2024.mp4"
    assert first["audio"].name == "Test Band Big Arena 2024.wav"

    second = run_pipeline(s, log=lambda *_: None)  # the same concert again must not overwrite the first
    assert second["video"].name == "Test Band Big Arena 2024 (2).mp4" and first["video"].exists()


def _fake_working_folder(root: Path) -> Path:
    d = root / "some-search"
    for sub in ("clips", "cache/audio", "out"):
        (d / sub).mkdir(parents=True)
    (d / "sources.json").write_text(json.dumps({"vid1": {"title": "x"}, "vid2": {"title": "y"}}), encoding="utf-8")
    for name in ("vid1.mp4", "vid2.webm", "vid3.f137.mp4", "vid3.mp4.part"):  # downloads; the last two half-finished
        (d / "clips" / name).write_bytes(b"x")
    (d / "cache" / "audio" / "vid1.mp4.16000.f32").write_bytes(b"x")
    (d / "timeline.json").write_text("{}", encoding="utf-8")
    (d / "meld.log").write_text("log", encoding="utf-8")
    return d


def test_remove_working_files_deletes_everything_meld_made(tmp_path):
    d = _fake_working_folder(tmp_path)
    assert gui.remove_working_files(d) is True
    assert not d.exists()


def test_remove_working_files_leaves_anything_that_is_not_meld_s(tmp_path):
    d = _fake_working_folder(tmp_path)
    (d / "clips" / "my own recording.mp4").write_bytes(b"x")  # not a download: no id in sources.json
    (d / "notes.txt").write_text("mine", encoding="utf-8")
    assert gui.remove_working_files(d) is False
    assert sorted(p.name for p in d.rglob("*") if p.is_file()) == ["my own recording.mp4", "notes.txt"]


def test_remove_working_files_survives_a_missing_folder(tmp_path):
    assert gui.remove_working_files(tmp_path / "never-existed") is True


def test_settings_remember_keep_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "_settings_path", lambda: tmp_path / "settings.json")
    assert Settings("x", tmp_path).keep_files is False  # by default only the fused files are kept
    assert gui.load_saved().get("keep") is None
    gui.save_settings(Settings("x", tmp_path, keep_files=True))
    assert gui.load_saved()["keep"] is True


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
