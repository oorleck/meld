from __future__ import annotations

import argparse
import sys

from .project import Project, slugify


def _size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-p", "--project", default=None,
        help="project directory (default: projects/default, or projects/<query> for auto)",
    )


def _add_fetch_opts(p: argparse.ArgumentParser, limit=250, max_duration=3600, max_height=1080, max_clips=500) -> None:
    p.add_argument("--limit", type=int, default=limit, help="results to keep per search query (default 250)")
    p.add_argument("--min-duration", type=float, default=20, help="skip shorter results (seconds)")
    p.add_argument("--max-duration", type=float, default=max_duration, help="skip longer results (seconds)")
    p.add_argument("--max-height", type=int, default=max_height, help="download resolution cap")
    p.add_argument("--max-clips", type=int, default=max_clips, help="download at most this many clips (default 500)")
    p.add_argument("--require", action="append", default=[], metavar="WORD", help="title must contain WORD (repeatable)")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads")
    p.add_argument(
        "--cookies-from-browser", choices=["firefox", "edge", "chrome", "brave"], default=None, metavar="BROWSER",
        help="use the YouTube login stored in this browser (firefox, edge, chrome, brave), which gets past YouTube's "
             "'confirm you're not a bot'. Read once and kept in memory only; fewer downloads at a time, with a pause "
             "between them. Heavy automatic use can get a Google account limited: consider a spare one. Chrome and "
             "Edge must be closed, and may not work at all; Firefox is the most reliable",
    )
    p.add_argument(
        "--require-date", action="store_true",
        help="drop titles that state no date or year at all (default: only for year-only queries like 'metallica 2003')",
    )
    p.add_argument(
        "--allow-undated", action="store_true",
        help="keep titles that state no date or year, even for a year-only query",
    )
    p.add_argument(
        "--any-video", action="store_true",
        help="also keep results that don't look like concert footage (interviews, commercials, rehearsals, covers, "
             "or no live/concert wording in the title)",
    )
    p.add_argument(
        "--loose", action="store_true",
        help="turn off strict matching (by default every query word must be in the title/channel, "
             "and titles naming a different date are dropped)",
    )
    p.add_argument("--dry-run", action="store_true", help="list candidates without downloading")


def _add_fetch(p: argparse.ArgumentParser) -> None:
    p.add_argument("urls", nargs="*", help="video URLs to download")
    p.add_argument("-s", "--search", action="append", default=[], metavar="QUERY", help="YouTube search query (repeatable)")
    _add_fetch_opts(p)


def _add_auto(p: argparse.ArgumentParser) -> None:
    p.add_argument("query", help='what to search for, e.g. "coldplay wembley 16 august 2022"')
    _add_fetch_opts(p, max_duration=900, max_height=720)


def _add_sync(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min-z", type=float, default=10.0, help="sync confidence needed to accept a clip")
    p.add_argument("--min-overlap", type=float, default=5.0, help="minimum seconds two clips must overlap")
    p.add_argument(
        "--max-compare", type=int, default=25,
        help="compare each clip against at most this many aligned clips (longest first); higher is slower but finds more",
    )
    p.add_argument(
        "--no-clusters", action="store_true",
        help="grow one group from the longest clip and reject everything else, as before (by default clips are sorted "
             "into groups that line up with each other, all at once, and the biggest group is used)",
    )
    p.add_argument(
        "--cluster", type=int, default=1, metavar="N",
        help="fuse the N-th biggest group of clips instead of the biggest (see the list printed by sync)",
    )
    p.add_argument(
        "--match-workers", type=int, default=None, metavar="N",
        help="compare N pairs of clips at once, in separate processes (default: automatic, up to 4 for big jobs; "
             "1 = one at a time). The result is the same whatever N is, only the time differs",
    )


def _add_audio(p: argparse.ArgumentParser) -> None:
    p.add_argument("--power", type=float, default=4.0, help="higher = stick closer to the single best mic")


def _add_video(p: argparse.ArgumentParser) -> None:
    p.add_argument("--size", type=_size, default=(1920, 1080), help="output size, e.g. 1280x720")
    p.add_argument("--min-shot", type=float, default=3.0, help="shortest shot (seconds)")
    p.add_argument("--max-shot", type=float, default=12.0, help="longest shot (seconds)")


def _add_recon(p: argparse.ArgumentParser) -> None:
    p.add_argument("--at", type=float, default=None, help="time (s on the shared timeline); default: when most phones are filming")
    p.add_argument("--max-views", type=int, default=24, help="most phones to use (best-looking first)")
    p.add_argument("--keep", type=float, default=0.5, help="fraction of the most confident pixels to keep")
    p.add_argument("--min-conf", type=float, default=1.5, help="drop pixels the model is less sure about than this")
    p.add_argument("--max-points", type=int, default=1_000_000)
    p.add_argument("--force", action="store_true", help="write the result even when the model reports low confidence")
    p.add_argument("--no-video", action="store_true", help="skip the swing.mp4 fly-around")
    p.add_argument("--size", type=_size, default=(1280, 720), help="fly-around video size")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="meld", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, helptext, adders in [
        ("fetch", "download candidate clips from YouTube", [_add_fetch]),
        ("sync", "align all clips on a shared timeline", [_add_sync]),
        ("audio", "fuse the audio", [_add_audio]),
        ("video", "cut the multicam video", [_add_video]),
        ("reconstruct", "3D-reconstruct one moment from all phones filming it (needs: uv sync --extra recon)", [_add_recon]),
        ("gui", "open the simple window (no terminal needed)", []),
        ("update-downloader", "download the newest YouTube downloader (yt-dlp)", []),
        ("run", "sync, audio and video in one go", [_add_sync, _add_audio, _add_video]),
        ("auto", "search YouTube, download many clips, then sync, fuse and render", [_add_auto, _add_sync, _add_audio, _add_video]),
    ]:
        p = sub.add_parser(name, help=helptext)
        if name not in ("gui", "update-downloader"):
            _add_common(p)
        for add in adders:
            add(p)

    args = ap.parse_args(argv)
    if args.cmd == "gui":
        from .gui import main as gui_main

        return gui_main()
    if args.cmd == "update-downloader":
        from . import ytdlp

        print(f"Current: {ytdlp.current_version()}")
        ytdlp.update(print)
        return
    sys.stdout.reconfigure(errors="replace")  # video titles are often not representable in the Windows console codepage
    if args.cmd == "auto" and not args.project:
        args.project = "projects/" + slugify(args.query, "auto")
    project = Project(args.project or "projects/default")

    if args.cmd in ("fetch", "auto"):
        from .fetch import expand_queries, fetch

        urls, queries = (args.urls, args.search) if args.cmd == "fetch" else ([], expand_queries(args.query))
        fetch(
            project, urls, queries, args.limit, args.min_duration, args.max_duration,
            args.max_height, args.dry_run, args.require, args.max_clips, args.workers,
            not args.loose, args.query if args.cmd == "auto" else None,
            True if args.require_date else (False if args.allow_undated else None), not args.any_video, _log,
            login_browser=args.cookies_from_browser,
        )
        if args.cmd == "fetch" or args.dry_run:
            return

    if args.cmd == "reconstruct":
        from .recon import reconstruct

        reconstruct(
            project, args.at, args.max_views, args.keep, args.min_conf, args.max_points,
            not args.no_video, args.size, args.force, _log,
        )
        return

    if args.cmd in ("sync", "run", "auto"):
        from .sync import sync_project

        sync_project(
            project, args.min_z, args.min_overlap, args.max_compare, _log, not args.no_clusters, args.cluster,
            workers=args.match_workers,
        )
    if args.cmd in ("audio", "run", "auto"):
        from .audiofuse import fuse_audio

        fuse_audio(project, args.power, _log)
    if args.cmd in ("video", "run", "auto"):
        from .videocut import render_video

        render_video(project, args.size, args.min_shot, args.max_shot, log=_log)


if __name__ == "__main__":
    sys.exit(main())
