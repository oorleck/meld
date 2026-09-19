"""Find (and update) yt-dlp.

YouTube changes often and old yt-dlp versions stop working, and a packaged app cannot be pip-upgraded. So the app
keeps a copy of yt-dlp's official release (a zip that Python can import directly) in the user's data folder and
puts it ahead of whatever yt-dlp came with the app.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = Path(base) / "Meld"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_zip() -> Path:
    return data_dir() / "yt-dlp.zip"


def bundled_zip() -> Path | None:
    """The copy shipped inside the installed app, if this is the packaged build."""
    root = getattr(sys, "_MEIPASS", None)
    p = Path(root) / "yt-dlp.zip" if root else None
    return p if p and p.exists() else None


def prepare() -> None:
    """Put the newest available yt-dlp first on sys.path. Must run before yt_dlp is first imported."""
    for p in (bundled_zip(), user_zip()):  # later insert(0) wins, so the user's copy ends up first
        if p and p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def import_yt_dlp():
    prepare()
    import yt_dlp

    return yt_dlp


def zip_version(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as z:
            m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", z.read("yt_dlp/version.py").decode("utf-8", "replace"))
            return m[1] if m else None
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def current_version() -> str | None:
    for p in (user_zip(), bundled_zip()):
        if p and p.exists() and (v := zip_version(p)):
            return v
    try:
        import yt_dlp.version

        return yt_dlp.version.__version__
    except ImportError:
        return None


def download_to(dest: Path) -> str:
    """Download the newest yt-dlp release to `dest` (checked to be a usable yt-dlp). Returns its version."""
    req = urllib.request.Request(RELEASE_URL, headers={"User-Agent": "meld"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:  # refuse anything that is not a usable yt-dlp
        if "yt_dlp/__init__.py" not in z.namelist():
            raise RuntimeError("The downloaded file is not a yt-dlp release.")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return zip_version(dest) or "unknown"


def update(log=print) -> str:
    """Update the user's copy of yt-dlp. Returns its version."""
    version = download_to(user_zip())
    log(f"YouTube downloader updated to {version}.")
    return version


def update_if_stale(max_age_days: float = 14, log=print) -> str | None:
    """Update when there is no user copy yet or it is older than max_age_days. Returns the new version or None."""
    p = user_zip()
    if p.exists() and time.time() - p.stat().st_mtime < max_age_days * 86400:
        return None
    return update(log)
