"""When YouTube refuses downloads ('Sign in to confirm you're not a bot'), say so, and stop asking."""
import json
import time
import types

import pytest

from meld import fetch as fetch_mod
from meld.fetch import BLOCK_LIMIT, BLOCKED_MESSAGE, fetch, is_blocked, short_reason
from meld.gui import Settings, UserError, friendly_error, run_pipeline
from meld.project import Project

# What yt-dlp really says (with the curly apostrophe YouTube uses)
BOT = (
    "ERROR: [youtube] jNQXAC9IVRw: Sign in to confirm you’re not a bot. Use --cookies-from-browser or --cookies "
    "for the authentication. See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
)
PRIVATE = "ERROR: [youtube] abc123DEF45: Private video. Sign in if you've been granted access to this video"
GONE = "ERROR: [youtube] abc123DEF45: Video unavailable"


@pytest.mark.parametrize("message, expected", [
    (BOT, True),
    ("Sign in to confirm you're not a bot", True),  # a straight apostrophe too
    ("ERROR: unable to download video data: HTTP Error 429: Too Many Requests", True),
    (PRIVATE, False),  # also says 'sign in', but it is this one video
    (GONE, False),
    ("<urlopen error [Errno 11001] getaddrinfo failed>", False),
    ("", False),
])
def test_only_a_refusal_of_the_connection_counts_as_blocked(message, expected):
    assert is_blocked(message) is expected


def test_short_reason_is_one_readable_line():
    assert short_reason(GONE) == "Video unavailable"
    assert short_reason("ERROR: [youtube:tab] Something odd\nmore lines") == "Something odd"
    assert short_reason("") == "unavailable"
    assert len(short_reason("ERROR: [youtube] abc123DEF45: " + "x" * 500)) == 120


# ---- _download keeps yt-dlp's own message


def _fake_yt_dlp(outcomes):
    """A stand-in for the yt_dlp module: url -> ("info", dict) | ("error", message) | ("silent", None)."""

    class YoutubeDL:
        def __init__(self, opts):
            self.logger = opts["logger"]  # _download must hand yt-dlp somewhere to report errors to

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            kind, value = outcomes[url]
            if kind == "error":
                self.logger.debug("[debug] noise")
                self.logger.warning("a warning")
                self.logger.error(value)  # what ignoreerrors does: report it and return None
                return None
            return value

    return types.SimpleNamespace(YoutubeDL=YoutubeDL)


def test_download_raises_with_the_message_yt_dlp_reported(monkeypatch):
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: _fake_yt_dlp({"u": ("error", BOT)}))
    with pytest.raises(RuntimeError) as e:
        fetch_mod._download({}, "u")
    assert is_blocked(str(e.value))


def test_download_returns_the_info_when_it_worked_and_none_when_nothing_was_said(monkeypatch):
    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: _fake_yt_dlp({
        "ok": ("info", {"id": "x"}), "quiet": ("silent", None),
    }))
    assert fetch_mod._download({}, "ok") == {"id": "x"}
    assert fetch_mod._download({}, "quiet") is None  # no message to raise with: the caller says 'unavailable'


# ---- fetch


def _candidates(n):
    return [{"id": f"{i:011d}", "title": f"Band Live {i}", "uploader": "", "duration": 100,
             "url": f"https://www.youtube.com/watch?v={i:011d}"} for i in range(n)]


@pytest.fixture
def youtube(monkeypatch):
    """A search that finds what a test sets, and downloads that succeed or fail as it says (by position in the list)."""
    state = {"results": [], "fail": {}, "calls": []}
    monkeypatch.setattr(fetch_mod, "search", lambda *a, **k: (state["results"], []))

    def download(opts, url):
        state["calls"].append(url)
        time.sleep(0.03)  # a real download takes long enough for the run to notice what happened and cancel the rest
        i = int(url.rsplit("=", 1)[1])
        message = state["fail"].get(i)
        if message:
            raise RuntimeError(message)
        return {"id": f"{i:011d}", "title": f"Band Live {i}", "uploader": "", "webpage_url": url, "duration": 100}

    monkeypatch.setattr(fetch_mod, "_download", download)
    return state


def run(tmp_path, youtube, results, fail, **kw):
    youtube["results"], youtube["fail"] = _candidates(results), fail
    lines = []
    project = Project(tmp_path)
    try:
        fetch(project, [], ["q"], match_query="q", log=lines.append, max_clips=None, workers=kw.pop("workers", 1), **kw)
        return project, lines, None
    except SystemExit as e:
        return project, lines, e


def test_a_blocked_connection_stops_the_run_early_with_a_plain_message(tmp_path, youtube):
    project, lines, exit_ = run(tmp_path, youtube, 40, {i: BOT for i in range(40)})
    assert exit_ is not None and exit_.code == BLOCKED_MESSAGE
    assert BLOCK_LIMIT <= len(youtube["calls"]) <= BLOCK_LIMIT + 2  # not all 40: the rest would be refused as well
    failed = [line for line in lines if "FAILED" in line]
    assert len(failed) == BLOCK_LIMIT and all("confirm you're not a bot" in line for line in failed)
    assert not any("unavailable" in line for line in failed)  # the old, misleading label
    assert BLOCKED_MESSAGE in lines
    assert json.loads(project.sources_path.read_text("utf-8")) == {}  # what was done is still saved


def test_with_workers_a_few_requests_already_running_may_finish_but_not_more(tmp_path, youtube):
    _, _, exit_ = run(tmp_path, youtube, 60, {i: BOT for i in range(60)}, workers=4)
    assert exit_ is not None
    assert len(youtube["calls"]) <= BLOCK_LIMIT + 4 + 4  # the block limit, plus those in flight when it was reached


def test_a_few_links_that_are_all_refused_are_reported_too(tmp_path, youtube):
    _, _, exit_ = run(tmp_path, youtube, 3, {i: BOT for i in range(3)})
    assert exit_ is not None and exit_.code == BLOCKED_MESSAGE  # fewer than the limit, but nothing worked


def test_when_some_downloads_work_the_run_goes_on(tmp_path, youtube):
    project, lines, exit_ = run(tmp_path, youtube, 10, {0: BOT, 1: BOT, 2: BOT})
    assert exit_ is None and len(youtube["calls"]) == 10
    assert len(json.loads(project.sources_path.read_text("utf-8"))) == 7
    assert any("3 of the failures were YouTube asking" in line for line in lines)


def test_refusals_that_come_and_go_never_stop_the_run(tmp_path, youtube):
    fail = {i: BOT for i in range(30) if i % 4 != 0}  # three refused, one works, three refused ...
    project, _, exit_ = run(tmp_path, youtube, 30, fail)
    assert exit_ is None and len(youtube["calls"]) == 30  # never BLOCK_LIMIT in a row
    assert len(json.loads(project.sources_path.read_text("utf-8"))) == 8


def test_when_refusals_start_after_some_worked_the_rest_is_skipped_but_what_worked_is_kept(tmp_path, youtube):
    project, lines, exit_ = run(tmp_path, youtube, 30, {i: BOT for i in range(3, 30)})
    assert exit_ is None  # three videos did download: go on with them, do not throw them away
    assert 3 + BLOCK_LIMIT <= len(youtube["calls"]) <= 3 + BLOCK_LIMIT + 2  # and stop asking for the other 27
    assert len(json.loads(project.sources_path.read_text("utf-8"))) == 3
    assert any("Going on with the 3 that worked" in line for line in lines)
    assert BLOCKED_MESSAGE not in lines


def test_the_run_is_stopped_even_if_a_lucky_one_worked_first(tmp_path, youtube):
    # what a real partial block looked like: one video of eight went through, the rest were refused
    _, _, exit_ = run(tmp_path, youtube, 40, {i: BOT for i in range(1, 40)})
    assert exit_ is None and len(youtube["calls"]) <= 1 + BLOCK_LIMIT + 2


def test_videos_that_are_simply_gone_do_not_count_as_a_block(tmp_path, youtube):
    project, lines, exit_ = run(tmp_path, youtube, 12, {i: GONE for i in range(12)})
    assert exit_ is None and len(youtube["calls"]) == 12  # every one is tried, as before
    assert all("(Video unavailable)" in line for line in lines if "FAILED" in line)  # and the real reason is shown


def test_private_videos_are_not_mistaken_for_a_block(tmp_path, youtube):
    _, _, exit_ = run(tmp_path, youtube, 8, {i: PRIVATE for i in range(8)})
    assert exit_ is None and len(youtube["calls"]) == 8


# ---- what the window says


def test_the_window_explains_a_block_in_plain_words():
    assert friendly_error(SystemExit(BLOCKED_MESSAGE)) == BLOCKED_MESSAGE  # from fetch
    assert friendly_error(RuntimeError(BOT)) == BLOCKED_MESSAGE  # e.g. the search itself was refused
    assert "phone hotspot" in BLOCKED_MESSAGE and "bot" in BLOCKED_MESSAGE
    assert "couldn't reach the internet" in friendly_error(OSError("<urlopen error getaddrinfo failed>"))  # unchanged


def test_a_blocked_run_is_not_reported_as_finding_no_videos(tmp_path, monkeypatch):
    from meld import gui

    def refused(*a, **k):
        raise SystemExit(BLOCKED_MESSAGE)

    monkeypatch.setattr(gui, "fetch", refused)
    with pytest.raises(SystemExit) as e:  # not the UserError 'I couldn't find any matching videos' it used to end in
        run_pipeline(Settings("metallica 2003", tmp_path), log=lambda *_: None)
    assert friendly_error(e.value) == BLOCKED_MESSAGE
    assert not isinstance(e.value, UserError)
