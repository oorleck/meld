"""Find and download concert clips from YouTube with yt-dlp.

Search results are only candidates: the sync step keeps the ones whose audio actually matches
the rest and rejects everything else, so it is fine to cast a wide net.
"""
from __future__ import annotations

import itertools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import cache, ytdlp
from .concerts import group_concerts
from .media import ffmpeg_exe
from .project import _PARTIAL, Project
from .relevance import Relevance

QUERY_SUFFIXES = (
    "live", "fan video", "front row", "crowd", "4K", "full song",
    # how the videos wanted describe themselves: phone footage from the audience, as filmed
    "iPhone footage", "filmed on iPhone", "shot on iPhone", "Samsung phone footage", "smartphone footage",
    "audience recording", "handheld footage", "live phone recording", "raw footage", "unedited footage",
)
SEARCH_WORKERS = 6  # searches made at once: with this many variations, one thread each would be a flood to YouTube
_VIDEO_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/)([\w-]{11})")

BLOCKED_MESSAGE = (
    "YouTube is refusing downloads from this connection: it asks to confirm you're not a bot. "
    "This usually passes after some hours. Trying another network, such as a phone hotspot, often works right away. "
    "Or let Meld use the YouTube login from your browser (YouTube login, at the bottom of the window; "
    "--cookies-from-browser on the command line)."
)
BLOCKED_WITH_LOGIN_MESSAGE = (
    "YouTube is still refusing downloads, even with the login from your browser. Make sure you are signed in to "
    "YouTube there, wait a few hours, or try another network, such as a phone hotspot."
)
SEARCH_MAX_AGE = 7 * 86400  # a saved search is used for this long (seconds): after that YouTube may have new videos
BLOCK_LIMIT = 6  # this many downloads in a row refused: YouTube is blocking us, so stop asking for the rest
_BLOCK_MARKERS = ("not a bot", "too many requests", "http error 429")


# Browsers whose YouTube login Meld can be asked to use, most reliable first (Chrome and Edge lock or encrypt theirs).
LOGIN_BROWSERS = {"firefox": "Firefox", "edge": "Microsoft Edge", "chrome": "Google Chrome", "brave": "Brave"}
LOGIN_WORKERS = 2  # with a login, fewer downloads at once and a pause between them: heavy automatic use of an account
LOGIN_PAUSE = (2, 5)  # (seconds, at least and at most) can get it limited
_LOGIN_COOKIES = {"LOGIN_INFO", "SAPISID", "__Secure-3PAPISID", "__Secure-1PSID", "__Secure-3PSID"}  # signed in to Google


def is_blocked(message: str) -> bool:
    """Whether a yt-dlp error says YouTube is refusing this connection (its 'confirm you're not a bot' check, or rate
    limiting), as opposed to a video that is private, removed or otherwise unavailable."""
    text = message.lower()
    return any(marker in text for marker in _BLOCK_MARKERS)


def short_reason(message: str) -> str:
    """One readable line from a yt-dlp error such as 'ERROR: [youtube] abc123DEF45: Video unavailable'."""
    lines = message.strip().splitlines()
    line = re.sub(r"^ERROR:\s*", "", lines[0].strip()) if lines else ""
    line = re.sub(r"^\[[^\]]+\]\s*(?:[\w-]{11}:\s*)?", "", line)
    return line[:120] or "unavailable"


def video_id(url: str) -> str | None:
    """The YouTube id in a video link, or None."""
    m = _VIDEO_ID.search(url)
    return m[1] if m else None


def expand_queries(query: str) -> list[str]:
    """One query becomes several so the results cover different uploaders' titles."""
    return [query] + [f"{query} {s}" for s in QUERY_SUFFIXES]


def search(
    query: str,
    limit: int,
    min_duration: float,
    max_duration: float,
    require: list[str] = (),
    relevance: Relevance | None = None,
    log=print,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Returns (kept results, [(title, reason it was dropped)])."""
    yt_dlp = ytdlp.import_yt_dlp()

    # Strict filtering throws most results away, so ask for more than we intend to keep.
    raw = min(limit * 2, 600) if relevance else limit
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
        info = ydl.extract_info(f"ytsearch{raw}:{query}", download=False)
    found, dropped = [], []
    for e in info.get("entries") or []:
        dur = e.get("duration")
        if dur is not None and not (min_duration <= dur <= max_duration):
            continue
        title = e.get("title") or ""
        uploader = e.get("uploader") or e.get("channel") or ""
        reason = relevance.check(title, uploader) if relevance else None
        if not reason:
            missing = [w for w in require if w.lower() not in title.lower()]
            reason = f"missing {', '.join(missing)}" if missing else None
        if reason:
            dropped.append((title, reason))
        elif len(found) < limit:
            found.append({
                "id": e["id"],
                "title": title,
                "uploader": uploader,
                "duration": dur,
                "url": e.get("url") or f"https://www.youtube.com/watch?v={e['id']}",
            })
    log(f"Search '{query}': kept {len(found)}, dropped {len(dropped)} as not matching")
    return found, dropped


def installed_browsers() -> list[str]:
    """The browsers (keys of LOGIN_BROWSERS) that have a profile on this computer, most reliable first. Only whether
    the folder exists is looked at, never what is in it."""
    if os.name != "nt":
        return list(LOGIN_BROWSERS)
    roaming, local = os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA")
    where = {
        "firefox": (roaming, "Mozilla", "Firefox", "Profiles"),
        "edge": (local, "Microsoft", "Edge", "User Data"),
        "chrome": (local, "Google", "Chrome", "User Data"),
        "brave": (local, "BraveSoftware", "Brave-Browser", "User Data"),
    }
    return [b for b, (base, *rest) in where.items() if base and Path(base, *rest).is_dir()]


class _Quiet:
    """The cookie reader of yt-dlp reports what it found. None of that is worth showing, and none of it may leak."""

    def debug(self, msg) -> None:
        pass

    info = warning = error = debug


def _read_browser_cookies(browser: str):
    """The cookie jar of `browser`, read once, in memory. (Kept apart so tests need no real browser.)"""
    ytdlp.import_yt_dlp()
    from yt_dlp.cookies import extract_cookies_from_browser

    return extract_cookies_from_browser(browser, logger=_Quiet())


def login_error(name: str, message: str) -> str:
    """What to tell the user when the login of browser `name` could not be read (`message` is what failed)."""
    text = message.lower()
    if any(w in text for w in ("could not copy", "permission", "locked", "being used")):
        return (f"I couldn't read the login from {name} because it is open. Close {name} completely "
                f"(also from the tray at the bottom right of the screen) and try again.")
    if "dpapi" in text or "decrypt" in text:
        return (f"{name} keeps its login in a form Meld can't read. Firefox works best: sign in to YouTube in Firefox "
                f"and choose it under YouTube login.")
    return f"I couldn't read the YouTube login from {name} ({short_reason(message)}). Firefox is the most reliable."


def load_login(browser: str):
    """The YouTube login stored in `browser`, to use for downloads. Raises SystemExit with what to do if there is none
    to be had. It is held in memory for the run: nothing is written to disk or to the log."""
    name = LOGIN_BROWSERS.get(browser, browser)
    try:
        jar = _read_browser_cookies(browser)
    except FileNotFoundError:
        raise SystemExit(
            f"I couldn't find a YouTube login in {name}: it doesn't look installed here, or has never been used. "
            f"Choose another browser under YouTube login."
        ) from None
    except Exception as e:  # noqa: BLE001 - whatever went wrong, the person needs to be told what to do about it
        raise SystemExit(login_error(name, str(e))) from None
    if not any(c.name in _LOGIN_COOKIES for c in jar):
        raise SystemExit(
            f"You don't seem to be signed in to YouTube in {name}. Open youtube.com there, sign in, close {name} "
            f"and try again."
        )
    return jar


class _Errors:
    """A logger for yt-dlp that keeps its error messages. With `ignoreerrors` it would otherwise print them somewhere
    nobody looks (the installed app has no console) and go on, leaving only 'it failed'."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, msg) -> None:
        pass

    info = warning = debug

    def error(self, msg) -> None:
        self.messages.append(str(msg))


def _download(opts: dict, url: str):
    """The downloaded video's info. Raises with yt-dlp's own message when it fails and says why."""
    yt_dlp = ytdlp.import_yt_dlp()

    opts = dict(opts)
    cookies = opts.pop("_meld_cookies", None)  # the login, read once by fetch(): not once per video
    errors = _Errors()
    try:
        with yt_dlp.YoutubeDL({**opts, "logger": errors}) as ydl:  # one per thread; they are not safe to share
            if cookies is not None:
                ydl.cookiejar = cookies
            info = ydl.extract_info(url, download=True)
    except Exception as e:  # noqa: BLE001 - the reason it gave the logger is the useful one
        raise RuntimeError(errors.messages[-1] if errors.messages else str(e)) from e
    if info is None and errors.messages:
        raise RuntimeError(errors.messages[-1])
    return info


def fetch(
    project: Project,
    urls: list[str],
    queries: list[str],
    limit: int = 250,
    min_duration: float = 20,
    max_duration: float = 3600,
    max_height: int = 1080,
    dry_run: bool = False,
    require: list[str] = (),
    max_clips: int | None = 500,
    workers: int = 4,
    strict: bool = True,
    match_query: str | None = None,
    require_date: bool | None = None,
    concert_only: bool = True,
    log=print,
    choose=None,
    audio_only: bool = False,
    login_browser: str | None = None,
    remember: bool = False,
) -> list[dict]:
    """`choose(concerts)`, if given, is asked which one to use when the results turn out to be from several different
    concerts (see concerts.py): it returns one of them, or None for "use everything", or raises to abort.
    `audio_only` downloads just the sound (a fraction of the size): enough to line clips up, not to cut video.
    `login_browser` (a key of LOGIN_BROWSERS) has the downloads use the YouTube login stored in that browser, which is
    what gets past 'confirm you're not a bot'. It is read once and kept in memory only; downloads then run fewer at a
    time, with a pause between them.
    `remember` keeps what the search found in the project (see cache.py), and uses it again if the same search is made
    within a week (the same words and limits), so that going through it again does not ask YouTube again. Videos
    already downloaded are never downloaded twice, whether it is remembered or not.
    Returns the videos wanted: the same list that is logged, each {id, title, uploader, duration, url}."""
    targets = [{"url": u, "title": None, "id": video_id(u), "uploader": None, "duration": None} for u in urls]
    memory = remember and not (choose or dry_run)  # asking which concert, or listing what would be dropped, is not repeated
    search_key = cache.key_of(
        "search", targets, queries, limit, min_duration, max_duration, max_clips, list(require), match_query, strict,
        require_date, concert_only,
    )
    saved = cache.load(project.root, "search", search_key, SEARCH_MAX_AGE) if memory else None

    def run_search(q: str):
        relevance = Relevance.from_query(match_query or q, require_date, concert_only) if strict else None
        return search(q, limit, min_duration, max_duration, require, relevance, log)

    if saved is not None:
        unique = saved
        log(f"Using the search made before: {len(unique)} video(s). (Nothing is asked of YouTube for it.)")
    else:
        per_query, all_dropped = [], {}
        if queries:
            with ThreadPoolExecutor(max_workers=min(len(queries), SEARCH_WORKERS)) as pool:  # network-bound
                for kept, dropped in pool.map(run_search, queries):
                    per_query.append(kept)
                    all_dropped.update(dict(dropped))
        if dry_run and all_dropped:
            log(f"Dropped {len(all_dropped)} (first 40):")
            for title, reason in list(all_dropped.items())[:40]:
                log(f"  x {title}  ({reason})")
        # Round-robin over the queries so the cap keeps the best hits of each rather than all of the first.
        for group in itertools.zip_longest(*per_query):
            targets += [t for t in group if t]

        seen, unique = set(), []
        for t in targets:
            if t["url"] not in seen:
                seen.add(t["url"])
                unique.append(t)

        # Several nights turn up in one search. Ask which to use before the cap, so the cap fills up from that one.
        groups = group_concerts(unique) if (choose or dry_run) else None
        if groups and len(groups.concerts) > 1:
            log(f"The results are from {len(groups.concerts)} different concerts:")
            for c in groups.concerts:
                log(f"  {c.label}: {c.count} videos")
            picked = choose(groups.concerts) if choose and not dry_run else None
            if picked is not None:
                unique = groups.select(unique, picked)
                log(f"Using {picked.label}: {picked.count} videos, and {len(groups.undecided)} that don't say which concert")
        if max_clips:
            unique = unique[:max_clips]

        if memory:
            cache.save(project.root, "search", search_key, unique)
    for t in unique:
        dur = f"{t['duration']:.0f}s" if t["duration"] else "?"
        log(f"  {dur:>6}  {t['title'] or t['url']}  [{t['uploader'] or ''}]")
    log(f"{len(unique)} unique clip(s) to fetch.")
    if dry_run or not unique:
        return unique

    opts = {
        "format": "ba/b" if audio_only else f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "outtmpl": str(project.clips_dir / "%(id)s.%(ext)s"),
        "ffmpeg_location": ffmpeg_exe(),
        "restrictfilenames": True,
        "ignoreerrors": True,
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
    }
    if not audio_only:
        opts["merge_output_format"] = "mp4"
    sources = json.loads(project.sources_path.read_text("utf-8")) if project.sources_path.exists() else {}
    todo = []
    for t in unique:
        if t["id"] and any(not _PARTIAL.search(f.name) for f in project.clips_dir.glob(f"{t['id']}.*")):
            log(f"  already have {t['title']}")
        else:
            todo.append(t)

    if login_browser and todo:
        opts["_meld_cookies"] = load_login(login_browser)
        opts["sleep_interval"], opts["max_sleep_interval"] = LOGIN_PAUSE
        workers = min(workers, LOGIN_WORKERS)
        log(f"Using the YouTube login from {LOGIN_BROWSERS.get(login_browser, login_browser)} (kept in memory only).")

    ok = failed = blocked = refused_in_a_row = 0
    stopped = False
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(_download, opts, t["url"]): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                info = fut.result()
            except Exception as e:  # a single bad video must not stop the batch
                info, message = None, str(e)
            else:
                message = "unavailable"
            if not info:
                failed += 1
                if is_blocked(message):
                    blocked += 1
                    refused_in_a_row += 1
                    reason = "YouTube asked to confirm you're not a bot"
                else:
                    refused_in_a_row = 0
                    reason = short_reason(message)
                log(f"[{i}/{len(todo)}] FAILED {t['title'] or t['url']} ({reason})")
                if refused_in_a_row >= BLOCK_LIMIT:
                    stopped = True  # it is not the videos: the rest would be refused too, and each ask makes it worse
                    break
                continue
            ok += 1
            refused_in_a_row = 0
            sources[info["id"]] = {
                "title": info.get("title"),
                "uploader": info.get("uploader"),
                "url": info.get("webpage_url"),
                "duration": info.get("duration"),
            }
            log(f"[{i}/{len(todo)}] downloaded {info.get('title')}")
    finally:
        # if log() raised (the GUI's Cancel does that), do not keep downloading everything still queued
        pool.shutdown(wait=True, cancel_futures=True)
        project.sources_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    log(f"Downloaded {ok}, failed {failed}. Clips are in {project.clips_dir}")
    if blocked and not ok and (stopped or blocked == failed):
        message = BLOCKED_WITH_LOGIN_MESSAGE if login_browser else BLOCKED_MESSAGE
        log(message)
        raise SystemExit(message)  # nothing can come of this run: say why, in words a person can act on
    if stopped:  # some downloads did work: go on with those rather than throw them away
        log(f"YouTube started refusing downloads (it asks to confirm you're not a bot), so the rest were skipped. "
            f"Going on with the {ok} that worked.")
    elif blocked:
        log(f"({blocked} of the failures were YouTube asking to confirm you're not a bot.)")
    return unique
