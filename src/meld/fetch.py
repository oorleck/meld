"""Find and download concert clips from YouTube with yt-dlp.

Search results are only candidates: the sync step keeps the ones whose audio actually matches
the rest and rejects everything else, so it is fine to cast a wide net.
"""
from __future__ import annotations

import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import ytdlp
from .concerts import group_concerts
from .media import ffmpeg_exe
from .project import Project
from .relevance import Relevance

QUERY_SUFFIXES = ("live", "fan video", "front row", "crowd", "4K", "full song")


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


def _download(opts: dict, url: str):
    yt_dlp = ytdlp.import_yt_dlp()

    with yt_dlp.YoutubeDL(opts) as ydl:  # one instance per thread; they are not safe to share
        return ydl.extract_info(url, download=True)


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
) -> None:
    """`choose(concerts)`, if given, is asked which one to use when the results turn out to be from several different
    concerts (see concerts.py): it returns one of them, or None for "use everything", or raises to abort."""
    targets = [{"url": u, "title": None, "id": None, "uploader": None, "duration": None} for u in urls]
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
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(project.clips_dir / "%(id)s.%(ext)s"),
        "ffmpeg_location": ffmpeg_exe(),
        "restrictfilenames": True,
        "ignoreerrors": True,
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
    }
    sources = json.loads(project.sources_path.read_text("utf-8")) if project.sources_path.exists() else {}
    todo = []
    for t in unique:
        if t["id"] and any(project.clips_dir.glob(f"{t['id']}.*")):
            log(f"  already have {t['title']}")
        else:
            todo.append(t)

    ok = failed = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(_download, opts, t["url"]): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                info = fut.result()
            except Exception as e:  # a single bad video must not stop the batch
                info, err = None, str(e).splitlines()[0][:120]
            else:
                err = "unavailable"
            if not info:
                failed += 1
                log(f"[{i}/{len(todo)}] FAILED {t['title'] or t['url']} ({err})")
                continue
            ok += 1
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
