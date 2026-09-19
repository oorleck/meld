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
        return [{"id": i, "title": self.titles.get(i), "url": f"https://www.youtube.com/watch?v={i}"} for i in ids]

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
    # the previews stay (a run of the same search again then needs no downloading or sorting), but not their decoded sound
    assert (s.project_dir / "preview" / "clips").is_dir() and len(list((s.project_dir / "preview" / "clips").iterdir())) == 2
    assert (s.project_dir / "preview" / "saved" / "sorted.json").exists()
    assert not (s.project_dir / "preview" / "cache" / "audio").exists()


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

    def fake_sync(project, log, choose, **kw):
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
        monkeypatch.setattr(gui, "sync_project", lambda project, log, choose, **kw: (_ for _ in ()).throw(UserError(f"stop here {choose}")))
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


# ---- the live view of the matching


def test_titles_are_read_from_the_downloads_record_by_video_id(tmp_path):
    from meld.project import Project

    project = Project(tmp_path)
    assert gui.load_titles(project) == {}  # nothing downloaded yet
    project.sources_path.write_text(json.dumps({"abc": {"title": "Band - Song"}, "def": {"title": None}, "ghi": {}}), encoding="utf-8")
    assert gui.load_titles(project) == {"abc": "Band - Song", "def": "", "ghi": ""}


def test_the_pipeline_shows_the_matching_as_it_happens_and_gives_the_titles(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube", titles={f"{n}{i}": f"Band {n.upper()} - Song {i}" for n in "ab" for i in (1, 2, 3)})
    _two_concerts(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    seen = []
    watched = run_pipeline(
        Settings("watched 2024", tmp_path / "w", clips=20, quality=480), log=lambda *_: None,
        choose_group=lambda options: options[0], on_match=lambda state, titles: seen.append((state, dict(titles))),
    )
    assert len(seen) > 10  # many steps, not just the end
    first, last = seen[0][0], seen[-1][0]
    assert first.groups == () and len(first.pending) == 6  # every clip waiting
    assert last.finished and last.chosen and len(last.groups) == 2  # two concerts, one of them used
    assert all(titles["a1"] == "Band A - Song 1" for _, titles in seen)  # the titles came with it, by video id
    assert {Path(n).stem for n in last.durations} == {"a1", "a2", "a3", "b1", "b2", "b3"}

    plain = run_pipeline(  # and watching does not change what is made
        Settings("watched 2024", tmp_path / "p", clips=20, quality=480), log=lambda *_: None, choose_group=lambda options: options[0],
    )
    assert (watched["used"], watched["skipped"]) == (plain["used"], plain["skipped"])


# ---- running the same search again


class _Log:
    """The log of runs, and how many comparisons of clips were made in sorting them (the line-up of each downloaded
    video with its own preview is another matter, and is done every time)."""

    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        self.lines.append(str(msg))

    @property
    def compared(self) -> int:
        return sum(1 for line in self.lines if " vs " in line and ": z=" in line and "comparisons" in line)


def test_the_same_search_again_does_not_sort_again_and_another_group_can_be_picked(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    _two_concerts(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    log = _Log()
    s = Settings("two shows 2024", tmp_path, clips=20, quality=480)

    first = run_pipeline(s, log=log, choose_group=lambda options: options[0])
    sorting = log.compared
    assert sorting > 0 and first["used"] == 3
    gui.clean_up_after(s)  # what the window does when a run is over: the videos go, the rest stays
    assert not (s.project_dir / "clips").exists()

    second = run_pipeline(s, log=log, choose_group=lambda options: options[1])  # the other show this time
    assert log.compared == sorting  # not sorted again
    assert any("before: using that" in line for line in log.lines)
    assert sorted(yt.downloaded("video")) == ["a1", "a2", "a3", "b1", "b2", "b3"]  # each video downloaded once, when chosen
    assert first["video"].exists() and second["video"].exists() and first["video"] != second["video"]
    assert 0.9 < second["minutes"] < 1.1  # the 60 s of the second show
    assert yt.kwargs[0]["remember"] is True and not yt.kwargs[1].get("remember")  # the search is kept; the downloads are not


def test_only_what_this_search_found_is_sorted_not_previews_left_by_another_search(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    _two_concerts(yt)
    s = Settings("two shows 2024", tmp_path, clips=20, quality=480)
    (s.project_dir / "preview" / "clips").mkdir(parents=True)
    shutil.copy(yt.folder / "b1.mp4", s.project_dir / "preview" / "clips" / "b1.mp4")  # left by an earlier, bigger search

    def searching(project, urls, queries, **kw):
        found = yt(project, urls, queries, **kw)
        return [t for t in found if t["id"] in {"a1", "a2", "a3"}] if queries else found  # this search found only show A

    monkeypatch.setattr(gui, "fetch", searching)
    result = run_pipeline(s, log=lambda *_: None, choose_group=lambda options: pytest.fail(f"one group only: {options}"))
    assert sorted(yt.downloaded("video")) == ["a1", "a2", "a3"] and result["used"] == 3


def test_a_newer_meld_does_not_use_what_an_older_one_saved(tmp_path, monkeypatch):
    from meld import cache

    yt = FakeYouTube(tmp_path / "youtube")
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    log = _Log()
    s = Settings("test show 2024", tmp_path, clips=20)
    run_pipeline(s, log=log)
    sorting = log.compared
    assert sorting > 0
    run_pipeline(s, log=log)
    assert log.compared == sorting and not any("newer Meld" in line for line in log.lines)  # the same Meld: all is used

    monkeypatch.setattr(cache, "__version__", "9.9.9")
    run_pipeline(s, log=log)
    assert log.compared == 2 * sorting and any("newer Meld" in line for line in log.lines)  # a new one sorts it again
    assert len(list((s.project_dir / "preview" / "clips").iterdir())) == 2  # what YouTube sent was not thrown away


def test_running_a_search_clears_the_previews_of_the_oldest_searches(tmp_path, monkeypatch):
    import os
    import time

    from meld import cache

    for i in range(7):  # seven earlier searches, s0 the newest
        d = tmp_path / f"s{i}" / "preview"
        (d / "clips").mkdir(parents=True)
        (d / "clips" / "x.m4a").write_bytes(b"x")
        (d / cache.MARKER).write_text("0.1.3", encoding="utf-8")
        stamp = time.time() - (i + 1) * 3600
        os.utime(d / cache.MARKER, (stamp, stamp))
    yt = FakeYouTube(tmp_path / "youtube")
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    lines = []
    run_pipeline(Settings("test show 2024", tmp_path, clips=20), log=lines.append)
    left = sorted(p.name for p in tmp_path.glob("s*") if (p / "preview").exists())
    assert left == ["s0", "s1", "s2", "s3"]  # this search and four more: five in all
    assert any("Cleared the saved previews" in line for line in lines)


# ---- what is kept when it is over


def _finished_search(root: Path) -> Path:
    """A search folder as a run leaves it: downloaded videos, working files, previews and what was sorted from them."""
    d = root / "some-search"
    for sub in ("clips", "cache/audio", "out", "preview/clips", "preview/saved"):
        (d / sub).mkdir(parents=True)
    (d / "preview" / "clips" / "vid1.m4a").write_bytes(b"x")
    (d / "preview" / "saved" / "sorted.json").write_text("{}", encoding="utf-8")
    (d / "preview" / "sources.json").write_text("{}", encoding="utf-8")
    (d / "preview" / "meld-version").write_text("0.1.3", encoding="utf-8")
    (d / "sources.json").write_text(json.dumps({"vid1": {"title": "x"}}), encoding="utf-8")
    (d / "clips" / "vid1.mp4").write_bytes(b"video")
    (d / "cache" / "audio" / "vid1.mp4.16000.f32").write_bytes(b"x")
    (d / "timeline.json").write_text("{}", encoding="utf-8")
    (d / "meld.log").write_text("log", encoding="utf-8")
    return d


def test_the_videos_are_kept_only_if_asked_and_the_previews_always(tmp_path):
    d = _finished_search(tmp_path)
    assert gui.clean_up_after(Settings("some search", tmp_path)) is True  # "keep the downloaded videos" is off
    assert not (d / "clips").exists() and not (d / "cache").exists() and not (d / "timeline.json").exists()
    assert (d / "preview" / "clips" / "vid1.m4a").exists() and (d / "preview" / "saved" / "sorted.json").exists()
    assert (d / "meld.log").exists()  # the log too: it is what to look at if something went wrong

    d = _finished_search(tmp_path / "kept")
    assert gui.clean_up_after(Settings("some search", tmp_path / "kept", keep_files=True)) is True  # ... and on
    assert (d / "clips" / "vid1.mp4").exists() and (d / "cache" / "audio" / "vid1.mp4.16000.f32").exists()
    assert (d / "preview" / "clips" / "vid1.m4a").exists()


def test_keeping_the_previews_leaves_the_persons_own_files_and_says_so_if_it_cannot_finish(tmp_path):
    d = _finished_search(tmp_path)
    (d / "clips" / "my own recording.mp4").write_bytes(b"x")  # not a download: no id in sources.json
    assert gui.remove_working_files(d, keep_previews=True) is False  # something that was to go is still there ...
    assert (d / "clips" / "my own recording.mp4").exists() and not (d / "clips" / "vid1.mp4").exists()  # ... it is theirs


def test_removing_everything_still_removes_the_previews_and_what_was_saved(tmp_path):
    d = _finished_search(tmp_path)
    assert gui.remove_working_files(d) is True and not d.exists()


def test_a_long_log_is_cut_to_its_end_from_a_whole_line(tmp_path):
    log = tmp_path / "meld.log"
    log.write_text("".join(f"line {i:06d}\n" for i in range(200_000)), encoding="utf-8")
    gui.trim_log(log, limit=1_000_000, keep=200_000)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert 15_000 < len(lines) < 20_000 and lines[-1] == "line 199999" and lines[0].startswith("line ")
    short = tmp_path / "short.log"
    short.write_text("a\nb\n", encoding="utf-8")
    gui.trim_log(short)
    assert short.read_text(encoding="utf-8") == "a\nb\n"


# ---- a stricter search


def test_settings_remember_strict_and_it_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "_settings_path", lambda: tmp_path / "settings.json")
    assert Settings("x", tmp_path).strict is False
    assert gui.load_saved().get("strict") is None
    gui.save_settings(Settings("x", tmp_path, strict=True))
    assert gui.load_saved()["strict"] is True
    gui.save_settings(Settings("x", tmp_path))
    assert gui.load_saved()["strict"] is False


def test_the_strict_choice_reaches_the_search(tmp_path, monkeypatch):
    yt = FakeYouTube(tmp_path / "youtube")
    _one_concert(yt)
    monkeypatch.setattr(gui, "fetch", yt)
    run_pipeline(Settings("test show 2024", tmp_path / "loose", clips=20), log=lambda *_: None)
    run_pipeline(Settings("test show 2024", tmp_path / "strict", clips=20, strict=True), log=lambda *_: None)
    previews = [k for k in yt.kwargs if k.get("audio_only")]
    assert [k["all_words"] for k in previews] == [False, True]


def test_when_strict_finds_nothing_the_message_says_what_to_turn_off(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "fetch", lambda *a, **k: [])
    with pytest.raises(UserError, match="couldn't find") as loose:
        run_pipeline(Settings("nothing here", tmp_path / "a"), log=lambda *_: None)
    with pytest.raises(UserError, match="turn off Strict search") as strict:
        run_pipeline(Settings("nothing here", tmp_path / "b", strict=True), log=lambda *_: None)
    assert "Strict" not in str(loose.value) and "Strict" in str(strict.value)
