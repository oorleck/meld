import io
import os
import sys
import time
import zipfile

import pytest

from meld import ytdlp


def make_zip(path, version="2026.01.02", with_package=True):
    with zipfile.ZipFile(path, "w") as z:
        if with_package:
            z.writestr("yt_dlp/__init__.py", "")
            z.writestr("yt_dlp/version.py", f"__version__ = '{version}'\n")
        else:
            z.writestr("something_else.py", "")


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    return tmp_path / "Meld"


def test_zip_version_is_read_without_importing(tmp_path):
    z = tmp_path / "yt-dlp.zip"
    make_zip(z, "2025.12.31")
    assert ytdlp.zip_version(z) == "2025.12.31"
    assert ytdlp.zip_version(tmp_path / "missing.zip") is None


def test_the_users_updated_copy_wins_over_the_bundled_one(data_home, monkeypatch):
    bundled = data_home.parent / "bundled"
    bundled.mkdir()
    make_zip(bundled / "yt-dlp.zip", "1.0")
    make_zip(ytdlp.user_zip(), "2.0")
    monkeypatch.setattr(sys, "_MEIPASS", str(bundled), raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    ytdlp.prepare()
    assert sys.path[0] == str(ytdlp.user_zip())
    assert sys.path[1] == str(bundled / "yt-dlp.zip")
    assert ytdlp.current_version() == "2.0"


def test_download_refuses_something_that_is_not_yt_dlp(data_home, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "hi")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(ytdlp.urllib.request, "urlopen", lambda *a, **k: Resp(buf.getvalue()))
    with pytest.raises(RuntimeError, match="not a yt-dlp"):
        ytdlp.update(lambda *_: None)
    assert not ytdlp.user_zip().exists()


def test_download_installs_a_good_release_atomically(data_home, monkeypatch):
    buf = io.BytesIO()
    make_zip(buf, "2026.09.19")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(ytdlp.urllib.request, "urlopen", lambda *a, **k: Resp(buf.getvalue()))
    assert ytdlp.update(lambda *_: None) == "2026.09.19"
    assert ytdlp.zip_version(ytdlp.user_zip()) == "2026.09.19"
    assert not ytdlp.user_zip().with_suffix(".part").exists()


def test_update_if_stale_only_downloads_when_old(data_home, monkeypatch):
    calls = []
    monkeypatch.setattr(ytdlp, "update", lambda log=print: calls.append(1) or "9.9")
    make_zip(ytdlp.user_zip())
    assert ytdlp.update_if_stale(14) is None and not calls  # fresh copy: nothing to do

    old = time.time() - 30 * 86400
    os.utime(ytdlp.user_zip(), (old, old))
    assert ytdlp.update_if_stale(14) == "9.9" and calls  # a month old: update


def test_selftest_passes_offline():
    from meld import selftest

    lines = []
    assert selftest.run(lines.append, online=False), "\n".join(lines)
    assert lines[-1] == "ALL OK"
