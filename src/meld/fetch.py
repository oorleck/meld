"""Find and download concert clips from YouTube with yt-dlp.

Search results are only candidates: the sync step keeps the ones whose audio actually matches
the rest and rejects everything else, so it is fine to cast a wide net.
"""
from __future__ import annotations

import itertools
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import ytdlp
from .concerts import group_concerts
from .media import ffmpeg_exe
from .project import _PARTIAL, Project
from .relevance import Relevance

QUERY_SUFFIXES = ("live", "fan video", "front row", "crowd", "4K", "full song")
_VIDEO_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/)([\w-]{11})")

BLOCKED_MESSAGE = (
    "YouTube is refusing downloads from this connection: it asks to confirm you're not a bot. "
    "This usually passes after some hours. Trying another network, such as a phone hotspot, often works right away."
)
BLOCK_LIMIT = 6  # this many downloads in a row refused: YouTube is blocking us, so stop asking for the rest
_BLOCK_MARKERS = ("not a bot", "too many requests", "http error 429")


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

    errors = _Errors()
    with yt_dlp.YoutubeDL({**opts, "logger": errors}) as ydl:  # one instance per thread; they are not safe to share
        info = ydl.extract_info(url, download=True)
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
) -> None:
    """`choose(concerts)`, if given, is asked which one to use when the results turn out to be from several different
    concerts (see concerts.py): it returns one of them, or None for "use everything", or raises to abort.
    `audio_only` downloads just the sound (a fraction of the size): enough to line clips up, not to cut video."""
    targets = [{"url": u, "title": None, "id": video_id(u), "uploader": None, "duration": None} for u in urls]
    def run_search(q: str):
        relevance = Relevance.from_query(match_query or q, require_date, concert_only) if strict else None
        return search(q, limit, min_duration, max_duration, require, relevance, log)

    per_query, all_dropped = [], {}
    if queries:
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:  # searching is network-bound
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

    for t in unique:
        dur = f"{t['duration']:.0f}s" if t["duration"] else "?"
        log(f"  {dur:>6}  {t['title'] or t['url']}  [{t['uploader'] or ''}]")
    log(f"{len(unique)} unique clip(s) to fetch.")
    if dry_run or not unique:
        return

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
        log(BLOCKED_MESSAGE)
        raise SystemExit(BLOCKED_MESSAGE)  # nothing can come of this run: say why, in words a person can act on
    if stopped:  # some downloads did work: go on with those rather than throw them away
        log(f"YouTube started refusing downloads (it asks to confirm you're not a bot), so the rest were skipped. "
            f"Going on with the {ok} that worked.")
    elif blocked:
        log(f"({blocked} of the failures were YouTube asking to confirm you're not a bot.)")
