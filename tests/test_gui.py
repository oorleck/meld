"""Tests for the GUI's logic. The window itself needs a display; the pipeline behind it does not."""
import json
import shutil
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


class FakeYouTube:
    """Stands in for fetch(). 'YouTube' is a folder of finished clips (<id>.mp4). A search puts up to max_clips of them
    in the project it is given, a download of links puts just those; every call is recorded."""

    def __init__(self, folder: Path, titles: dict | None = None):
        self.folder = folder
        self.titles = titles or {}
        self.calls: list[tuple[str, bool, list[str], int | None]] = []  # (kind, audio only, ids, max height)
        self.kwargs: list[dict] = []  # what each call was asked, whole
        folder.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, audio, video_src: str = "testsrc2=size=640x360:rate=30") -> None:
        make_clip(self.folder / f"{name}.mp4", audio, SR, video_src)

    def __call__(self, project, urls, queries, **kw) -> None:
        ids = sorted(p.stem for p in self.folder.glob("*.mp4")) if queries else [u.rsplit("=", 1)[1] for u in urls]
        if queries:
            ids = ids[: kw.get("max_clips") or len(ids)]
        for i in ids:
            if not (project.clips_dir / f"{i}.mp4").exists():
                shutil.copy(self.folder / f"{i}.mp4", project.clips_dir / f"{i}.mp4")
        sources = json.loads(project.sources_path.read_text("utf-8")) if project.sources_path.exists() else {}
        sources.update({i: {"title": self.titles.get(i)} for i in ids})
        project.sources_path.write_text(json.dumps(sources), encoding="utf-8")
        kind = "preview" if project.root.name == "preview" else "video"
        self.calls.append((kind, bool(kw.get("audio_only")), ids, kw.get("max_height")))
        self.kwargs.append(kw)

    def downloaded(self, kind: str) -> list[str]:
        return [i for k, _, ids, _ in self.calls if k == kind for i in ids]


def _one_concert(yt: FakeYouTube) -> None:
    """Two phones at the same show: together they cover 70 s of it."""
    music = synth_music(70, SR)
    yt.add("a", phone(music, SR, 0, 45, seed=1))
    yt.add("b", phone(music, SR, 25, 70, seed=2), "rgbtestsrc=size=640x360:rate=30")


def _two_concerts(yt: FakeYouTube) -> None:
    """Three phones at each of two different shows; the first show's clips cover 70 s, the second's 60 s."""
    a, b = synth_music(70, SR), synth_music(70, SR, seed=5)
    for name, music, t0, t1, seed in [
        ("a1", a, 0, 45, 1), ("a2", a, 25, 70, 2), ("a3", a, 10, 55, 3),
        ("b1", b, 0, 40, 4), ("b2", b, 20, 60, 5), ("b3", b, 10, 50, 6),
    ]:
        yt.add(name, phone(music, SR, t0, t1, seed=seed))


def test_pipeline_runs_all_six_stages_and_reports(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    s = Settings("test show 2024", tmp_path, clips=20, quality=720)
    stages = []
    result = run_pipeline(s, log=lambda *_: None, stage=lambda i, name: stages.append((i, name)))

    assert [i for i, _ in stages] == [0, 1, 2, 3, 4, 5] == list(range(len(gui.STAGES)))
    # no video titles are known here, so the files are named after the search; they go straight into the save folder
    assert result["video"] == tmp_path / "Test Show 2024.mp4" and result["video"].exists()
    assert result["audio"] == tmp_path / "Test Show 2024.wav" and result["audio"].exists()
    assert result["folder"] == tmp_path
    assert not (s.project_dir / "out" / "fused.mp4").exists()  # moved, not copied
    assert result["used"] == 2 and result["skipped"] == 0
    assert 1.0 < result["minutes"] < 1.4  # the two clips together cover 70 s
    # first small audio-only previews of the candidates, then the videos themselves
    assert [(kind, audio_only) for kind, audio_only, _, _ in yt.calls] == [("preview", True), ("video", False)]
    assert not (s.project_dir / "preview").exists()  # the previews have done their job


def test_only_the_chosen_groups_videos_are_downloaded(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    _two_concerts(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    shown = []

    def choose_group(options):
        shown.extend(options)
        return options[1]  # the second show

    s = Settings("two shows 2024", tmp_path, clips=20, quality=480)
    result = run_pipeline(s, log=lambda *_: None, choose_group=choose_group)
    assert [(o.label.split(" ")[0:2], o.count) for o in shown] == [(["Group", "1"], 3), (["Group", "2"], 3)]
    assert sorted(yt.downloaded("preview")) == ["a1", "a2", "a3", "b1", "b2", "b3"]  # everything was previewed ...
    assert sorted(yt.downloaded("video")) == ["b1", "b2", "b3"]  # ... but only the chosen group's videos were downloaded
    assert result["used"] == 3 and result["skipped"] == 3
    assert 0.9 < result["minutes"] < 1.1  # the 60 s of the second show


def test_without_a_choice_the_biggest_group_is_used_and_only_its_videos_downloaded(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    _two_concerts(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    result = run_pipeline(Settings("two shows 2024", tmp_path, clips=20), log=lambda *_: None)
    assert sorted(yt.downloaded("video")) == ["a1", "a2", "a3"]  # the show that covers the most time
    assert result["used"] == 3 and result["skipped"] == 3


def test_no_more_videos_are_downloaded_than_asked_for(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    music = synth_music(70, SR)
    for name, t0, t1, seed in [("a1", 0, 50, 1), ("a2", 20, 60, 2), ("a3", 30, 65, 3), ("a4", 10, 35, 4)]:
        yt.add(name, phone(music, SR, t0, t1, seed=seed))
    monkeypatch.setattr(gui, "fetch", yt)
    result = run_pipeline(Settings("few 2024", tmp_path, clips=2), log=lambda *_: None)
    assert sorted(yt.downloaded("video")) == ["a1", "a2"]  # the two longest of the group of four
    assert result["used"] == 2 and result["skipped"] == 2
    assert len(yt.calls[0][2]) <= gui.preview_count(2)  # and no more previewed than the preview allowance


def test_smallest_quality_caps_the_download_and_sizes_the_video(tmp_path, monkeypatch):
    from meld.media import probe

    yt = FakeYouTube(tmp_path / "youtube")
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    result = run_pipeline(Settings("small show 2024", tmp_path, clips=20, quality=480), log=lambda *_: None)
    assert [max_height for kind, _, _, max_height in yt.calls if kind == "video"] == [480]  # what YouTube is asked for
    info = probe(result["video"])
    assert (info.width, info.height) == (854, 480)  # what comes out


def test_pipeline_names_the_files_after_the_videos_it_used(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube", titles={
        "a": "Test Band - Song One (Live at Big Arena 2024) 4K",
        "b": "TEST BAND Big Arena 2024 - Song Two",
    })
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    s = Settings("test band 2024", tmp_path, clips=20, quality=720)
    first = run_pipeline(s, log=lambda *_: None)
    assert first["video"].name == "Test Band Big Arena 2024.mp4"
    assert first["audio"].name == "Test Band Big Arena 2024.wav"

    second = run_pipeline(s, log=lambda *_: None)  # the same concert again must not overwrite the first
    assert second["video"].name == "Test Band Big Arena 2024 (2).mp4" and first["video"].exists()


def _timeline_of(*ids):
    from meld.timeline import ClipEntry, Timeline

    return Timeline([ClipEntry(f"{i}.mp4", k * 100.0, 120.0, True, 1, 1) for k, i in enumerate(ids)])


def test_group_rows_name_the_place_that_tells_the_groups_apart(tmp_path):
    from meld.project import Project

    titles = {
        "a1": "Oasis - Wonderwall (Live at Principality Stadium, Cardiff 2025)",
        "a2": "Oasis Cardiff Principality Stadium 2025 Supersonic",
        "a3": "Oasis live Cardiff 2025 - Principality Stadium - Live Forever",
        "m1": "Oasis Live 25 Manchester Heaton Park 2025 - Champagne Supernova",
        "m2": "Oasis - Heaton Park Manchester 2025 Hello",
        "m3": "Oasis live Heaton Park Manchester 2025 - Slide Away",
    }
    project = Project(tmp_path)
    (tmp_path / "sources.json").write_text(json.dumps({k: {"title": v} for k, v in titles.items()}), encoding="utf-8")
    first, second = gui.describe_groups(project, [_timeline_of("a1", "a2", "a3"), _timeline_of("m1", "m2", "m3")])
    # the band and the year are in both groups' titles, so they say nothing about which is which
    assert first.label.startswith("Group 1: Cardiff") and "Principality" in first.label and "Oasis" not in first.label
    assert second.label.startswith("Group 2:") and "Manchester" in second.label and "Heaton" in second.label
    assert first.count == second.count == 3 and "min" in first.label

    bare = gui.describe_groups(Project(tmp_path / "no-titles"), [_timeline_of("x1", "x2", "x3"), _timeline_of("y1", "y2", "y3")])
    assert [o.label.split(" ")[0:2] for o in bare] == [["Group", "1"], ["Group", "2"]]  # no titles: just numbered


def test_pipeline_asks_which_group_and_hands_the_answer_to_sync(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    yt.add("x", phone(synth_music(20, SR), SR, 0, 15, seed=1))  # something to preview: sync itself is stubbed
    monkeypatch.setattr(gui, "fetch", yt)
    s = Settings("ask me 2024", tmp_path)
    seen = {}

    def fake_sync(project, log, choose):
        seen["index"] = choose([_timeline_of("a1", "a2", "a3"), _timeline_of("b1", "b2", "b3", "b4")])
        raise UserError("stop here")

    monkeypatch.setattr(gui, "sync_project", fake_sync)
    shown = []

    def choose_group(options):
        shown.extend(options)
        return options[1]

    with pytest.raises(UserError, match="stop here"):
        run_pipeline(s, log=lambda *_: None, choose_group=choose_group)
    assert seen["index"] == 1 and [o.count for o in shown] == [3, 4]

    with pytest.raises(UserError, match="stop here"):  # without a chooser, sync is not given one
        monkeypatch.setattr(gui, "sync_project", lambda project, log, choose: (_ for _ in ()).throw(UserError(f"stop here {choose}")))
        run_pipeline(s, log=lambda *_: None)


def _fake_working_folder(root: Path) -> Path:
    d = root / "some-search"
    for sub in ("clips", "cache/audio", "out", "preview/clips"):
        (d / sub).mkdir(parents=True)
    (d / "preview" / "clips" / "vid1.m4a").write_bytes(b"x")  # an audio preview a stopped run left behind
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
    yt = FakeYouTube(tmp_path / "youtube")
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    s = Settings("cancel me", tmp_path)
    calls = {"n": 0}

    def log(msg=""):
        calls["n"] += 1
        if calls["n"] > 3:
            raise Cancelled()

    with pytest.raises(Cancelled):
        run_pipeline(s, log=log)
    assert not (s.project_dir / "out" / "fused.mp4").exists()
    assert not list(tmp_path.glob("*.mp4"))  # nothing was saved


def test_shorten_path_keeps_both_ends():
    from meld.gui import shorten_path

    short = "C:/Users/me/Documents/Meld"
    assert shorten_path(short) == short
    long = "C:/Users/someone/AppData/Local/Temp/a-very-long-folder-name/another-long-one/results"
    out = shorten_path(long)
    assert len(out) <= 56 and out.startswith(long[:18]) and out.endswith(long[-10:]) and " ... " in out


# ---- the YouTube login (for when YouTube asks to confirm you are not a bot)


def test_the_chosen_browser_is_remembered_and_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "_settings_path", lambda: tmp_path / "settings.json")
    assert Settings("x", tmp_path).login is None
    assert gui.load_saved().get("login") is None
    gui.save_settings(Settings("x", tmp_path, login="firefox"))
    assert gui.load_saved()["login"] == "firefox"
    gui.save_settings(Settings("x", tmp_path))  # switched off again
    assert gui.load_saved()["login"] is None


def test_with_a_login_fewer_previews_are_asked_for_and_less_room_is_needed(tmp_path):
    assert gui.preview_count(500) == gui.PREVIEW_MAX and gui.preview_count(20) == 40  # as before without a login
    assert gui.preview_count(500, login=True) == gui.LOGIN_MAX_PREVIEWS < gui.PREVIEW_MAX
    assert gui.preview_count(20, login=True) == 40  # small runs are not touched
    assert Settings("x", tmp_path, clips=500, login="firefox").disk_needed_gb() < Settings("x", tmp_path, clips=500).disk_needed_gb()


def test_the_login_reaches_both_downloads_and_caps_the_previews(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    run_pipeline(Settings("login show 2024", tmp_path, clips=500, quality=480, login="firefox"), log=lambda *_: None)
    previews, videos = yt.kwargs
    assert previews["login_browser"] == videos["login_browser"] == "firefox"  # the previews and the full videos
    assert previews["max_clips"] == gui.LOGIN_MAX_PREVIEWS


def test_without_a_login_nothing_is_passed_on(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    run_pipeline(Settings("no login 2024", tmp_path, clips=500, quality=480), log=lambda *_: None)
    previews, videos = yt.kwargs
    assert previews["login_browser"] is None and videos["login_browser"] is None
    assert previews["max_clips"] == gui.PREVIEW_MAX


def test_a_login_that_cannot_be_read_is_shown_in_plain_words(tmp_path, monkeypatch):
    message = "I couldn't read the login from Google Chrome because it is open. Close Google Chrome completely."

    def refused(*a, **k):
        raise SystemExit(message)

    monkeypatch.setattr(gui, "fetch", refused)
    with pytest.raises(SystemExit) as e:
        run_pipeline(Settings("x", tmp_path, login="chrome"), log=lambda *_: None)
    assert friendly_error(e.value) == message
