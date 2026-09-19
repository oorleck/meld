"""What Meld remembers from one run of a search to the next, so that running it again does not repeat the slow steps.

Two kinds of thing are kept for a search:

- Downloads, which are what YouTube sent and so do not depend on Meld: the small audio previews. (The videos are
  a different matter: they are big, and only kept when the person asks for that.)
- Results worked out from them: the search (which videos it found), the sorting by sound (which belong together) and every
  score of a pair of clips. These live in `saved/` next to the previews, one small JSON file each, and are used only
  if they were made by this very version of Meld (a new version may find things differently, and a fix must not be
  hidden behind results from before it) and from the same inputs (the `key`). Anything else is worked out afresh.

`stamp` is called once for a folder at the start of a run: it drops what an older Meld made and notes that this folder
was used now, which is what `trim` goes by when it clears out the searches that have not been used for a while.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from . import __version__

SAVED = "saved"  # the folder in a project for what was worked out
MARKER = "meld-version"  # the file in a project that says which Meld last used it, and when (its modified time)
KEEP_SEARCHES = 5  # `trim` keeps the previews and results of this many searches ...
KEEP_DAYS = 30  # ... and of none that has not been used for this long


def key_of(*parts: Any) -> str:
    """A short fingerprint of what a result was worked out from: the same parts, the same key."""
    text = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def file_ids(files) -> dict[str, int]:
    """name -> size of each file: enough to tell whether a downloaded clip is the same one as before."""
    return {Path(f).name: Path(f).stat().st_size for f in files}


def _entry(root: Path, name: str) -> Path:
    return Path(root) / SAVED / f"{name}.json"


def load(root: Path, name: str, key: str, max_age: float | None = None) -> Any | None:
    """What was saved as `name` in the project at `root`, if it was made by this version of Meld from the same inputs
    (`key`) and, when `max_age` (seconds) is given, not longer ago than that. Otherwise None."""
    try:
        data = json.loads(_entry(root, name).read_text(encoding="utf-8"))
        if data["meld"] != __version__ or data["key"] != key:
            return None
        if max_age is not None and time.time() - data["at"] > max_age:
            return None
        return data["value"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(root: Path, name: str, key: str, value: Any) -> None:
    """Remember `value` as `name`. Never raises: not being able to save costs speed next time, nothing else."""
    path = _entry(root, name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"meld": __version__, "key": key, "at": time.time(), "value": value}), encoding="utf-8",
        )
        os.replace(tmp, path)  # a run stopped half way through saving leaves the old file, not half of a new one
    except (OSError, TypeError, ValueError):
        pass


def stamp(project) -> bool:
    """Start of a run in `project`: throw away what a different version of Meld worked out (the saved results and the
    `cache` folder of decoded audio and the like), and note that the project is used now. Returns whether anything
    was thrown away. The downloaded clips are left alone."""
    root = Path(project.root)
    marker = root / MARKER
    try:
        before = marker.read_text(encoding="utf-8").strip()
    except OSError:
        before = None
    dropped = False
    if before is not None and before != __version__:
        for folder in (root / SAVED, project.cache_dir):
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
                dropped = True
        project.cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        marker.write_text(__version__, encoding="utf-8")  # also sets the time it was used
    except OSError:
        pass
    return dropped


def folder_size(path: Path) -> int:
    """Bytes in the files under `path`."""
    total = 0
    for base, _, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(base, n))
            except OSError:
                pass
    return total


def searches(folder: Path) -> list[Path]:
    """The search folders in `folder` that hold previews and results Meld saved, the most recently used first."""
    found = []
    try:
        children = list(Path(folder).iterdir())
    except OSError:
        return []
    for d in children:
        marker = d / "preview" / MARKER
        if marker.is_file():
            found.append((marker.stat().st_mtime, d))
    return [d for _, d in sorted(found, key=lambda t: -t[0])]


def forget(search_dir: Path) -> None:
    """Delete the previews and saved results of one search, and the folder too if that leaves nothing of the person's
    (a kept video, a note) in it. Videos that were kept are never touched."""
    d = Path(search_dir)
    shutil.rmtree(d / "preview", ignore_errors=True)
    shutil.rmtree(d / SAVED, ignore_errors=True)
    if not d.is_dir():
        return
    ours = {"meld.log", MARKER}  # what Meld itself leaves in the folder of a search
    theirs = [  # anything else, however deep: a video that was kept, a file the person put there
        os.path.join(base, n) for base, _, names in os.walk(d) for n in names if not (Path(base) == d and n in ours)
    ]
    if not theirs:
        for n in ours:
            (d / n).unlink(missing_ok=True)
    for sub in ("clips", "cache", "out", "."):  # empty ones only: rmdir refuses otherwise
        try:
            (d / sub).rmdir()
        except OSError:
            pass


def trim(folder: Path, keep: Path | None = None, searches_kept: int = KEEP_SEARCHES, days: float = KEEP_DAYS) -> list[Path]:
    """Clear out the saved previews and results of searches in `folder` that were used least recently: all but the
    `searches_kept` newest, and any not used for `days`. The search at `keep` (the one being run) is never cleared.
    Returns the searches cleared."""
    cleared = []
    kept = 0
    now = time.time()
    for d in searches(folder):
        if keep is not None and d.resolve() == Path(keep).resolve():
            kept += 1
            continue
        age = (now - (d / "preview" / MARKER).stat().st_mtime) / 86400
        if kept >= searches_kept or age > days:
            forget(d)
            cleared.append(d)
        else:
            kept += 1
    return cleared
