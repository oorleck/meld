"""Using the YouTube login stored in a browser, to get past 'confirm you're not a bot'."""
import types

import pytest

from meld import fetch as fetch_mod
from meld.fetch import (
    BLOCKED_MESSAGE, BLOCKED_WITH_LOGIN_MESSAGE, LOGIN_BROWSERS, LOGIN_PAUSE, LOGIN_WORKERS, fetch, installed_browsers,
    load_login, login_error,
)
from meld.project import Project

BOT = "ERROR: [youtube] jNQXAC9IVRw: Sign in to confirm you\u2019re not a bot. Use --cookies-from-browser or --cookies"


def cookie(name):
    return types.SimpleNamespace(name=name, value="SECRET-VALUE-NEVER-SHOWN")


# ---- load_login: read once, and say what to do when it cannot be had


def reads(monkeypatch, jar=None, error=None):
    calls = []

    def read(browser):
        calls.append(browser)
        if error:
            raise error
        return jar

    monkeypatch.setattr(fetch_mod, "_read_browser_cookies", read)
    return calls


def test_a_login_with_youtube_sign_in_cookies_is_returned(monkeypatch):
    jar = [cookie("PREF"), cookie("LOGIN_INFO")]
    reads(monkeypatch, jar=jar)
    assert load_login("firefox") is jar


@pytest.mark.parametrize("signed_in_cookie", ["LOGIN_INFO", "SAPISID", "__Secure-3PAPISID", "__Secure-1PSID", "__Secure-3PSID"])
def test_any_of_the_google_sign_in_cookies_counts(monkeypatch, signed_in_cookie):
    reads(monkeypatch, jar=[cookie(signed_in_cookie)])
    load_login("edge")


def test_a_browser_that_is_not_signed_in_is_explained(monkeypatch):
    reads(monkeypatch, jar=[cookie("PREF"), cookie("VISITOR_INFO1_LIVE")])  # cookies, but no sign-in
    with pytest.raises(SystemExit) as e:
        load_login("chrome")
    assert "don't seem to be signed in to YouTube in Google Chrome" in e.value.code
    assert "close Google Chrome" in e.value.code


def test_a_browser_that_is_not_installed_is_explained(monkeypatch):
    reads(monkeypatch, error=FileNotFoundError('could not find brave cookies database in "C:\\x"'))
    with pytest.raises(SystemExit) as e:
        load_login("brave")
    assert "couldn't find a YouTube login in Brave" in e.value.code and "another browser" in e.value.code


@pytest.mark.parametrize("failure, expected", [
    ("Could not copy Chrome cookie database. See  https://github.com/yt-dlp/yt-dlp/issues/7271  for more info",
     "because it is open. Close Google Chrome completely"),
    (PermissionError(13, "Permission denied"), "because it is open"),
    ("Failed to decrypt with DPAPI. See  https://github.com/yt-dlp/yt-dlp/issues/10927  for more info",
     "Firefox works best"),
    ("something odd happened\nwith a second line", "(something odd happened). Firefox is the most reliable"),
])
def test_each_way_reading_a_login_can_fail_says_what_to_do(monkeypatch, failure, expected):
    reads(monkeypatch, error=failure if isinstance(failure, Exception) else RuntimeError(failure))
    with pytest.raises(SystemExit) as e:
        load_login("chrome")
    assert expected in e.value.code


def test_the_messages_never_contain_cookie_values(monkeypatch):
    reads(monkeypatch, jar=[cookie("PREF")])
    with pytest.raises(SystemExit) as e:
        load_login("firefox")
    assert "SECRET" not in e.value.code
    assert "SECRET" not in login_error("Firefox", "failed")


# ---- which browsers are offered


def test_only_browsers_that_have_a_profile_folder_are_offered(tmp_path, monkeypatch):
    roaming, local = tmp_path / "Roaming", tmp_path / "Local"
    (roaming / "Mozilla" / "Firefox" / "Profiles").mkdir(parents=True)
    (local / "Google" / "Chrome" / "User Data").mkdir(parents=True)
    monkeypatch.setattr(fetch_mod.os, "name", "nt")
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    assert installed_browsers() == ["firefox", "chrome"]  # in the order of how reliable they are

    monkeypatch.delenv("APPDATA")
    monkeypatch.delenv("LOCALAPPDATA")
    assert installed_browsers() == []  # nothing to look in: none, rather than looking in the current folder


def test_the_offered_browsers_are_all_known_ones():
    assert list(LOGIN_BROWSERS)[0] == "firefox"  # the reliable one comes first
    assert set(LOGIN_BROWSERS) <= {"firefox", "edge", "chrome", "brave"}


# ---- fetch uses it


def _candidates(n):
    return [{"id": f"{i:011d}", "title": f"Band Live {i}", "uploader": "", "duration": 100,
             "url": f"https://www.youtube.com/watch?v={i:011d}"} for i in range(n)]


@pytest.fixture
def youtube(monkeypatch):
    state = {"results": _candidates(8), "seen_opts": [], "fail": False, "pool_sizes": []}
    monkeypatch.setattr(fetch_mod, "search", lambda *a, **k: (state["results"], []))

    def download(opts, url):
        state["seen_opts"].append(opts)
        if state["fail"]:
            raise RuntimeError(BOT)
        i = int(url.rsplit("=", 1)[1])
        return {"id": f"{i:011d}", "title": f"Band Live {i}", "uploader": "", "webpage_url": url, "duration": 100}

    monkeypatch.setattr(fetch_mod, "_download", download)
    real_pool = fetch_mod.ThreadPoolExecutor

    def pool(max_workers=None, **kw):
        state["pool_sizes"].append(max_workers)
        return real_pool(max_workers=max_workers, **kw)

    monkeypatch.setattr(fetch_mod, "ThreadPoolExecutor", pool)
    return state


def run(tmp_path, **kw):
    lines = []
    fetch(Project(tmp_path), [], ["q"], match_query="q", log=lines.append, max_clips=None, workers=kw.pop("workers", 4), **kw)
    return lines


def test_without_a_login_nothing_changes(tmp_path, monkeypatch, youtube):
    calls = reads(monkeypatch, jar=[cookie("LOGIN_INFO")])
    run(tmp_path)
    assert calls == []  # no browser was touched
    assert all("_meld_cookies" not in o and "sleep_interval" not in o for o in youtube["seen_opts"])
    assert 4 in youtube["pool_sizes"]  # all the workers asked for


def test_the_login_is_read_once_and_shared_by_every_download(tmp_path, monkeypatch, youtube):
    jar = [cookie("LOGIN_INFO")]
    calls = reads(monkeypatch, jar=jar)
    lines = run(tmp_path, login_browser="firefox")
    assert calls == ["firefox"]  # once, not once per video
    assert len(youtube["seen_opts"]) == 8 and all(o["_meld_cookies"] is jar for o in youtube["seen_opts"])
    assert any("Using the YouTube login from Firefox" in line for line in lines)
    assert not any("SECRET" in line for line in lines)


def test_with_a_login_downloads_are_gentler(tmp_path, monkeypatch, youtube):
    reads(monkeypatch, jar=[cookie("LOGIN_INFO")])
    run(tmp_path, login_browser="firefox", workers=4)
    assert max(youtube["pool_sizes"]) == LOGIN_WORKERS < 4  # fewer at once
    assert all((o["sleep_interval"], o["max_sleep_interval"]) == LOGIN_PAUSE for o in youtube["seen_opts"])  # and paused

    youtube["pool_sizes"].clear()
    run(tmp_path / "again", login_browser="firefox", workers=1)
    assert max(youtube["pool_sizes"]) == 1  # never more than asked for


def test_a_login_that_cannot_be_read_stops_before_any_download(tmp_path, monkeypatch, youtube):
    reads(monkeypatch, error=RuntimeError("Failed to decrypt with DPAPI"))
    with pytest.raises(SystemExit) as e:
        run(tmp_path, login_browser="edge")
    assert "Firefox works best" in e.value.code
    assert youtube["seen_opts"] == []  # not one request was made


def test_nothing_to_download_means_the_browser_is_not_touched(tmp_path, monkeypatch, youtube):
    calls = reads(monkeypatch, jar=[cookie("LOGIN_INFO")])
    youtube["results"] = []
    run(tmp_path, login_browser="firefox")
    assert calls == []


def test_still_refused_with_a_login_says_so_and_does_not_suggest_the_login_again(tmp_path, monkeypatch, youtube):
    reads(monkeypatch, jar=[cookie("LOGIN_INFO")])
    youtube["fail"] = True
    with pytest.raises(SystemExit) as e:
        run(tmp_path, login_browser="firefox")
    assert e.value.code == BLOCKED_WITH_LOGIN_MESSAGE
    assert "YouTube login" not in BLOCKED_WITH_LOGIN_MESSAGE  # it is already in use


def test_refused_without_a_login_points_at_the_login(tmp_path, youtube):
    youtube["fail"] = True
    with pytest.raises(SystemExit) as e:
        run(tmp_path)
    assert e.value.code == BLOCKED_MESSAGE and "YouTube login" in BLOCKED_MESSAGE
    assert "--cookies-from-browser" in BLOCKED_MESSAGE  # for the command line


# ---- _download hands the shared login to yt-dlp, and keeps it out of its options


def test_download_attaches_the_shared_login_and_does_not_pass_it_as_an_option(monkeypatch):
    made = []

    class YoutubeDL:
        def __init__(self, opts):
            made.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            self.used = getattr(self, "cookiejar", None)
            made.append(self.used)
            return {"id": "x"}

    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: types.SimpleNamespace(YoutubeDL=YoutubeDL))
    jar = object()
    opts = {"format": "ba/b", "_meld_cookies": jar}
    assert fetch_mod._download(opts, "u") == {"id": "x"}
    assert made[1] is jar  # the login was attached to the downloader
    assert "_meld_cookies" not in made[0] and "cookiesfrombrowser" not in made[0]  # not re-read from the browser each time
    assert "_meld_cookies" in opts  # the caller's own options are left alone (other threads use them)


def test_a_failure_while_downloading_keeps_the_reason_yt_dlp_gave_the_logger(monkeypatch):
    class YoutubeDL:
        def __init__(self, opts):
            self.logger = opts["logger"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            self.logger.error("Could not copy Chrome cookie database.")
            raise RuntimeError("failed to load cookies")  # what yt-dlp raises after logging the real reason

    monkeypatch.setattr(fetch_mod.ytdlp, "import_yt_dlp", lambda: types.SimpleNamespace(YoutubeDL=YoutubeDL))
    with pytest.raises(RuntimeError, match="Could not copy Chrome cookie database"):
        fetch_mod._download({}, "u")


# ---- the command line


def test_the_command_line_accepts_a_browser_and_nothing_else(tmp_path, monkeypatch):
    from meld import cli

    seen = []
    monkeypatch.setattr(fetch_mod, "fetch", lambda *a, **k: seen.append(k.get("login_browser")))
    cli.main(["fetch", "-p", str(tmp_path), "--cookies-from-browser", "firefox", "https://youtu.be/aaaaaaaaaaa"])
    cli.main(["fetch", "-p", str(tmp_path), "https://youtu.be/aaaaaaaaaaa"])
    assert seen == ["firefox", None]  # given, and left alone by default

    with pytest.raises(SystemExit) as e:  # only the browsers Meld knows how to read
        cli.main(["fetch", "-p", str(tmp_path), "--cookies-from-browser", "netscape", "https://youtu.be/aaaaaaaaaaa"])
    assert e.value.code == 2
