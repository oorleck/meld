"""A download that yt-dlp reports as failed half way, but returns the video's info for all the same, is not a download."""
import json
import types

import pytest

from meld import fetch as fetch_mod
from meld.fetch import DOWNLOAD_TRIES, fetch
from meld.project import Project

NETWORK = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
BOT = "ERROR: [youtube] jNQXAC9IVRw: Sign in to confirm you’re not a bot."


def fake_yt_dlp(script, folder):
    """A yt_dlp whose i-th download of a url does script[url][i]: ("ok",) puts the file there, ("error", message) reports the
    error and, as `ignoreerrors` has it do, returns the info without any file, ("error+file", message) reports one and has
    the file, ("partial", message) reports one and leaves half a file. `calls` counts the downloads of each url."""
    calls = {}

    class YoutubeDL:
        def __init__(self, opts):
            self.logger = opts["logger"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            n = calls[url] = calls.get(url, 0) + 1
            outcome = script[url][min(n, len(script[url])) - 1]
            vid = url.rsplit("=", 1)[1]
            if outcome[0] in ("ok", "error+file"):
                (folder / f"{vid}.webm").write_bytes(b"x")
            if outcome[0] == "partial":
                (folder / f"{vid}.f251.webm").write_bytes(b"x")
                (folder / f"{vid}.webm.part").write_bytes(b"x")
            if outcome[0] != "ok":
                self.logger.error(outcome[1])
            return {"id": vid, "title": f"Video {vid}", "uploader": "", "webpage_url": url, "duration": 100}

    return types.SimpleNamespace(YoutubeDL=YoutubeDL), calls


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(fetch_mod, "DOWNLOAD_RETRY_PAUSE", 0)


def opts_for(folder):
    return {"outtmpl": str(folder / "%(id)s.%(ext)s")}


def test_a_download_that_left_no_file_is_tried_again_and_then_fails_with_the_reason(tmp_path, monkeypatch, quick):
    yt, calls = fake_yt_dlp({"https://x/watch?v=aaaaaaaaaaa": [("error", NETWORK)]}, tmp_path)
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: yt)
    with pytest.raises(RuntimeError, match="HTTP Error 403"):
        fetch_mod._download(opts_for(tmp_path), "https://x/watch?v=aaaaaaaaaaa")
    assert calls["https://x/watch?v=aaaaaaaaaaa"] == DOWNLOAD_TRIES  # asked for a few times, not given up at once
    assert not list(tmp_path.glob("aaaaaaaaaaa.*"))


def test_a_download_that_works_the_second_time_is_a_download(tmp_path, monkeypatch, quick):
    url = "https://x/watch?v=bbbbbbbbbbb"
    yt, calls = fake_yt_dlp({url: [("error", NETWORK), ("ok",)]}, tmp_path)
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: yt)
    info = fetch_mod._download(opts_for(tmp_path), url)
    assert info["id"] == "bbbbbbbbbbb" and calls[url] == 2 and (tmp_path / "bbbbbbbbbbb.webm").exists()


def test_a_refusal_is_not_asked_again_at_once(tmp_path, monkeypatch, quick):
    url = "https://x/watch?v=ccccccccccc"
    yt, calls = fake_yt_dlp({url: [("error", BOT)]}, tmp_path)
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: yt)
    with pytest.raises(RuntimeError, match="not a bot"):
        fetch_mod._download(opts_for(tmp_path), url)
    assert calls[url] == 1


def test_an_error_that_left_the_file_there_is_not_a_failure(tmp_path, monkeypatch, quick):
    url = "https://x/watch?v=ddddddddddd"
    yt, calls = fake_yt_dlp({url: [("error+file", "ERROR: unable to download video thumbnail")]}, tmp_path)
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: yt)
    assert fetch_mod._download(opts_for(tmp_path), url)["id"] == "ddddddddddd" and calls[url] == 1


def test_half_a_file_is_not_a_finished_one(tmp_path, monkeypatch, quick):
    url = "https://x/watch?v=eeeeeeeeeee"
    yt, calls = fake_yt_dlp({url: [("partial", NETWORK)]}, tmp_path)
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: yt)
    with pytest.raises(RuntimeError):
        fetch_mod._download(opts_for(tmp_path), url)
    assert calls[url] == DOWNLOAD_TRIES  # .f251.webm and .part do not count


def test_a_video_with_no_file_is_counted_as_failed_and_not_recorded_as_downloaded(tmp_path, monkeypatch, quick):
    ids = ["fffffffffff", "ggggggggggg"]
    urls = {i: f"https://www.youtube.com/watch?v={i}" for i in ids}
    project = Project(tmp_path)
    yt, calls = fake_yt_dlp({urls["fffffffffff"]: [("ok",)], urls["ggggggggggg"]: [("error", NETWORK)]}, project.clips_dir)
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: yt)
    monkeypatch.setattr(fetch_mod, "search", lambda *a, **k: ([
        {"id": i, "title": f"Band Live {i}", "uploader": "", "duration": 100, "url": urls[i]} for i in ids
    ], []))
    lines = []
    fetch(project, [], ["band live"], match_query="band live", log=lines.append, max_clips=None, audio_only=True, workers=1)
    text = "\n".join(lines)
    assert "Downloaded 1, failed 1" in text and "FAILED Band Live ggggggggggg (unable to download video data" in text
    assert sorted(json.loads(project.sources_path.read_text(encoding="utf-8"))) == ["fffffffffff"]  # only what is there
    assert [f.name for f in project.clip_files()] == ["fffffffffff.webm"]

    # the next run tries the one that is missing again (it is not "already have"), and is given it this time
    yt2, calls2 = fake_yt_dlp({urls["ggggggggggg"]: [("ok",)], urls["fffffffffff"]: [("ok",)]}, project.clips_dir)
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: yt2)
    lines.clear()
    fetch(project, [], ["band live"], match_query="band live", log=lines.append, max_clips=None, audio_only=True, workers=1)
    assert calls2 == {urls["ggggggggggg"]: 1}  # only the missing one
    assert sorted(f.name for f in project.clip_files()) == ["fffffffffff.webm", "ggggggggggg.webm"]
