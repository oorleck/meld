"""A simple window: type what concert to look for, press Start, get one combined video.

The work is done by the same functions the command line uses (fetch, sync, audio, video); this module only
gathers a few plain choices, runs them in a background thread and shows progress.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

from . import cache, ytdlp
from .audiofuse import fuse_audio
from .matchcanvas import MatchCanvas, Theme
from .fetch import BLOCKED_MESSAGE, LOGIN_BROWSERS, expand_queries, fetch, installed_browsers, is_blocked
from .naming import concert_name, shared_words, trim_connectors, unique_stem
from .project import _PARTIAL, Project, slugify
from .sync import line_up_downloads, sync_project
from .videocut import render_video

APP_NAME = "Meld"
STAGES = [
    "Finding and previewing the videos",
    "Sorting them by their sound",
    "Downloading the chosen videos",
    "Lining them up",
    "Mixing the sound",
    "Cutting the video",
]
STEP_NAMES = ["Find videos", "Sort by sound", "Download", "Line them up", "Mix the sound", "Cut the video"]  # of STAGES
IDLE_TEXT = "Ready when you are."
EXAMPLES = ["Metallica 2003", "Coldplay Wembley 16 August 2022"]
CLIP_CHOICES = [("A few (quickest)", 20), ("A good amount", 100), ("As many as I can find (slow)", 500)]
QUALITY_CHOICES = [("Full HD (1080p)", 1080), ("Smaller and faster (720p)", 720), ("Smallest files (480p)", 480)]
# The picture quality caps the download height and is also the size of the finished video.
OUTPUT_SIZES = {1080: (1920, 1080), 720: (1280, 720), 480: (854, 480)}
MAX_GROUPS_SHOWN = 8  # the picker lists this many, biggest first
MB_PER_CLIP = {480: 15, 720: 30, 1080: 60}  # rough download size, to warn before filling the disk
# Candidates are first downloaded as small audio-only previews and sorted into groups by their sound; only the videos
# of the group the user picks are downloaded in full. More candidates are previewed than videos wanted, because only
# one of the groups they fall into is used.
PREVIEW_FACTOR, PREVIEW_MAX, PREVIEW_MB = 2, 250, 6
LOGIN_MAX_PREVIEWS = 100  # with a YouTube login in use, fewer requests: heavy automatic use can get an account limited


def preview_count(clips: int, login: bool = False) -> int:
    """How many candidates get an audio preview when `clips` videos are wanted."""
    n = min(clips * PREVIEW_FACTOR, PREVIEW_MAX)
    return min(n, LOGIN_MAX_PREVIEWS) if login else n


class Cancelled(Exception):
    """Raised from the log callback when the user presses Cancel; unwinds whatever step is running."""


class UserError(Exception):
    """A problem worth explaining to the user in plain words."""


@dataclass
class Settings:
    query: str
    folder: Path  # each search gets its own sub-folder in here
    clips: int = 100
    quality: int = 720
    keep_files: bool = False  # keep the downloaded videos and working files after a successful run
    login: str | None = None  # a browser (a key of LOGIN_BROWSERS) whose YouTube login the downloads use
    strict: bool = False  # only videos whose title has every word typed, and the date (see relevance.py)

    @property
    def project_dir(self) -> Path:
        return Path(self.folder) / slugify(self.query)

    @property
    def size(self) -> tuple[int, int]:
        for height in sorted(OUTPUT_SIZES, reverse=True):  # the biggest size the quality reaches
            if self.quality >= height:
                return OUTPUT_SIZES[height]
        return OUTPUT_SIZES[min(OUTPUT_SIZES)]

    def disk_needed_gb(self) -> float:
        previews = preview_count(self.clips, bool(self.login)) * PREVIEW_MB
        return (previews + self.clips * MB_PER_CLIP.get(self.quality, 60) + 1500) / 1024


def picked_titles(project: Project, tl) -> list[str]:
    """YouTube titles of the videos that ended up on the timeline (not the ones that were rejected)."""
    try:
        sources = json.loads(project.sources_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [t for c in tl.clips if (t := sources.get(Path(c.file).stem, {}).get("title"))]


@dataclass
class Option:
    """One row of the picker dialog: what to show, and how many videos it stands for."""

    label: str
    count: int


def describe_groups(project: Project, groups: list) -> list[Option]:
    """Rows for groups of clips that line up with each other, biggest first: 'Group 1: <place> · 92 min'. The place is
    what the titles of that group share and the other groups' titles do not (the band and year are in all of them)."""
    try:
        sources = json.loads(project.sources_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sources = {}
    titles = [[sources.get(Path(c.file).stem, {}).get("title") or "" for c in g.clips] for g in groups]
    words = [shared_words([t for t in ts if t]) for ts in titles]
    # what every group has (the band, the year) says nothing about which is which; only what sets them apart does
    common = set.intersection(*({w.casefold() for w in ws} for ws in words)) if words else set()
    options = []
    for i, (g, ws) in enumerate(zip(groups, words), 1):
        own = [w for w in ws if w.casefold() not in common and not any(c.isdigit() for c in w)]
        place = " ".join(trim_connectors(own)[:5])
        minutes = sum(b - a for a, b in g.segments()) / 60
        options.append(Option(f"Group {i}" + (f": {place}" if place else "") + f" · {minutes:.0f} min", len(g.clips)))
    return options


def load_titles(project: Project) -> dict[str, str]:
    """YouTube video id -> title, for the clips downloaded into `project`."""
    try:
        sources = json.loads(project.sources_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {vid: info.get("title") or "" for vid, info in sources.items()}


def run_pipeline(s: Settings, log=print, stage=lambda i, name: None, choose_group=None, on_match=None) -> dict:
    """Preview, sort, download, line up, mix and cut. The video and the sound go into `s.folder`, named after the concert.

    Candidates are first downloaded as small audio-only previews and sorted into groups by their sound; if there are
    several groups worth choosing between, `choose_group(options)` is asked which (it returns one of the options).
    `on_match(state, titles)`, if given, is called as they are sorted, to show it happening (see matchview.py).
    Only the videos of that group are then downloaded in full. Returns
    {"video", "audio", "folder", "minutes", "used", "skipped"}."""
    project = Project(s.project_dir)
    preview = project.preview()
    wanted = preview_count(s.clips, bool(s.login))

    if any([cache.stamp(project), cache.stamp(preview)]):  # what an older Meld worked out may not be right by this one
        log("This is a newer Meld than the one that saved this search: working out what was saved again.")
    for old in cache.trim(s.folder, keep=s.project_dir):
        log(f"Cleared the saved previews of an older search: {old.name}")

    stage(0, STAGES[0])
    found = fetch(
        preview, [], expand_queries(s.query),
        limit=max(wanted, 30), min_duration=20, max_duration=900, max_clips=wanted, match_query=s.query,
        log=log, audio_only=True, login_browser=s.login, remember=True, all_words=s.strict,
    )
    # only what this search found: previews left from a bigger search before are not part of it
    ids = {t["id"] for t in found if t.get("id")} if found else None
    previewed = len([f for f in preview.clip_files() if ids is None or f.stem in ids])
    if not previewed:
        raise UserError(
            "I couldn't find any matching videos. Try fewer words, for example just the band and the year"
            + (", or turn off Strict search." if s.strict else ".")
        )

    def pick_group(groups) -> int | None:
        options = describe_groups(preview, groups)
        answer = choose_group(options)
        return options.index(answer) if answer is not None else None

    titles: dict[str, str] = {}

    def watch(state) -> None:
        if not titles:
            titles.update(load_titles(preview))
        on_match(state, titles)

    stage(1, STAGES[1])
    sorted_tl = sync_project(
        preview, log=log, choose=pick_group if choose_group else None, remember=True, only=ids,
        **({"on_state": watch} if on_match else {}),
    )
    if not sorted_tl.clips:
        raise UserError("None of the videos could be lined up with each other, so there is nothing to combine.")
    chosen = sorted(sorted_tl.clips, key=lambda c: -c.duration)[: s.clips]  # the longest, if the group is bigger

    stage(2, STAGES[2])
    fetch(
        project, [f"https://www.youtube.com/watch?v={Path(c.file).stem}" for c in chosen], [],
        max_height=s.quality, max_clips=None, log=log, login_browser=s.login,
    )

    stage(3, STAGES[3])
    tl = line_up_downloads(preview, project, chosen, log=log)
    if not tl.clips:
        raise UserError("I couldn't download the videos of that group. Please check your connection and try again.")

    stage(4, STAGES[4])
    fuse_audio(project, log=log)

    stage(5, STAGES[5])
    video = render_video(project, size=s.size, log=log)

    stem = unique_stem(s.folder, concert_name(picked_titles(project, tl), fallback=s.query))
    log(f"Saving as {stem}")
    final_video = Path(shutil.move(str(video), str(s.folder / f"{stem}.mp4")))
    final_audio = Path(shutil.move(str(project.out_dir / "fused_audio.wav"), str(s.folder / f"{stem}.wav")))
    # The previews stay, with what was worked out from them: running the same search again (to pick another group, say)
    # then needs no search, no download of previews and no sorting. Only the decoded sound, which is big and quick to
    # make again, goes; and the videos, by the caller, unless the person asked to keep them.
    if not s.keep_files:
        shutil.rmtree(preview.cache_dir / "audio", ignore_errors=True)
    return {
        "video": final_video,
        "audio": final_audio,
        "folder": s.folder,
        "minutes": sum(b - a for a, b in tl.segments()) / 60,
        "used": len(tl.clips),
        "skipped": previewed - len(tl.clips),
    }


def remove_working_files(project_dir: Path, keep_previews: bool = False) -> bool:
    """Delete what Meld downloaded and worked out for one search, once the result is safe elsewhere.

    Only Meld's own files go: the videos it downloaded (named by YouTube id in sources.json, plus half-finished
    downloads), the cache, and the timeline/sources. Anything else that happens to be in the folder stays.
    With `keep_previews` the small audio previews and what was worked out from them (see cache.py) stay too, with the
    log, so that the same search can be run again quickly; the videos, which are the big part, still go.
    Returns True when everything that was to go is gone (and, without `keep_previews`, the folder itself)."""
    d = Path(project_dir)
    try:
        ids = set(json.loads((d / "sources.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        ids = set()
    clips = d / "clips"
    for f in clips.iterdir() if clips.is_dir() else ():
        if f.is_file() and (f.name.split(".")[0] in ids or _PARTIAL.search(f.name)):
            f.unlink(missing_ok=True)
    shutil.rmtree(d / "cache", ignore_errors=True)
    names = ["timeline.json", "sources.json"]
    if not keep_previews:
        shutil.rmtree(d / "preview", ignore_errors=True)  # the audio previews, if a run stopped before removing them
        shutil.rmtree(d / cache.SAVED, ignore_errors=True)
        names += ["meld.log", cache.MARKER]
    for name in names:
        (d / name).unlink(missing_ok=True)
    for folder in (clips, d / "out", d):  # only succeeds while they are empty
        try:
            folder.rmdir()
        except OSError:
            pass
    if not keep_previews:
        return not d.exists()
    kept = {"preview", "meld.log", cache.MARKER, cache.SAVED}
    return not any(p for p in d.rglob("*") if p.is_file() and p.relative_to(d).parts[0] not in kept)


def clean_up_after(s: Settings) -> bool:
    """After a run that finished: the downloaded videos go unless the person asked to keep them (they are the big part, and
    the video and sound that were asked for are safe in the save folder). What is small and slow to get again stays,
    for the next run of the same search: the audio previews and what was worked out from them (see cache.py).
    Returns whether everything that was to go is gone."""
    if s.keep_files:
        return True
    return remove_working_files(s.project_dir, keep_previews=True)


def trim_log(path: Path, limit: int = 1_000_000, keep: int = 200_000) -> None:
    """The log of a search is kept from run to run; when it gets long, only the end of it is kept."""
    try:
        if path.stat().st_size > limit:
            data = path.read_bytes()[-keep:]
            path.write_bytes(data[data.find(b"\n") + 1:])  # from a whole line
    except OSError:
        pass


_PROGRESS = (
    re.compile(r"^\[(\d+)/(\d+)\] (?:downloaded|FAILED)"),  # downloads
    re.compile(r"(\d+)/(\d+) placed"),  # lining up
    re.compile(r"^\s+(\d+)/(\d+)$"),  # rendering shots
)


def parse_progress(line: str) -> tuple[int, int] | None:
    """(done, total) if a log line reports progress."""
    for rx in _PROGRESS:
        m = rx.search(line)
        if m and int(m[2]) > 0:
            return int(m[1]), int(m[2])
    return None


def friendly_error(exc: BaseException) -> str:
    if isinstance(exc, UserError):
        return str(exc)
    if isinstance(exc, SystemExit):
        return str(exc.code) if exc.code else "Stopped."
    if is_blocked(str(exc)):  # e.g. the search itself was refused
        return BLOCKED_MESSAGE
    text = str(exc).lower()
    if any(w in text for w in ("urlopen", "getaddrinfo", "timed out", "connection", "network", "resolve")):
        return "I couldn't reach the internet. Please check your connection and try again."
    return f"Something went wrong: {exc}"


# ---------------------------------------------------------------- settings memory


def _settings_path() -> Path:
    return ytdlp.data_dir() / "settings.json"


def load_saved() -> dict:
    try:
        return json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: Settings) -> None:
    try:
        _settings_path().write_text(
            json.dumps({
                "folder": str(s.folder), "clips": s.clips, "quality": s.quality, "query": s.query,
                "keep": s.keep_files, "login": s.login, "strict": s.strict,
            }),
            encoding="utf-8",
        )
    except OSError:
        pass


def format_size(n: float) -> str:
    """Bytes as people say them: '340 MB', '1.2 GB'."""
    for unit, size in (("GB", 1e9), ("MB", 1e6)):
        if n >= size:
            return f"{n / size:.1f} {unit}" if unit == "GB" and n < 10e9 else f"{n / size:.0f} {unit}"
    return f"{max(n, 0) / 1e3:.0f} kB"


def shorten_path(text: str, limit: int = 56) -> str:
    """Keep the start and end of a long path so it does not stretch the window."""
    if len(text) <= limit:
        return text
    head = 18
    return text[:head] + " ... " + text[-(limit - head - 5):]


def icon_path() -> Path | None:
    """meld.ico, next to the packaged app or in packaging/ when running from source."""
    for base in (getattr(sys, "_MEIPASS", None), Path(__file__).resolve().parents[2] / "packaging"):
        if base and (Path(base) / "meld.ico").exists():
            return Path(base) / "meld.ico"
    return None


def default_folder() -> Path:
    return Path.home() / "Documents" / "Meld"


# ---------------------------------------------------------------- look and feel

# Colours are taken from the app icon: a dark navy square with yellow and coral sound-wave bars.
BG = "#12131c"
SURFACE = "#1b1d2b"
SURFACE2 = "#252838"
SURFACE3 = "#30344c"
SELECTED = "#3a3420"  # the chosen toggle button: a dark tint of the accent
BORDER = "#333852"
TEXT = "#eef0f8"
MUTED = "#9399b8"
ACCENT = "#ffd23f"
ACCENT_HI = "#ffe07a"
ACCENT_LO = "#e0b423"
CORAL = "#ff785a"
ON_ACCENT = "#1a1608"
LOG_BG = "#0d0e15"
ICON_BG = "#181a26"
WAVE = [0.25, 0.55, 0.85, 0.5, 1.0, 0.6, 0.8, 0.4, 0.2]  # bar heights, same as packaging/make_icon.py


def apply_theme(style, px) -> None:
    """Dark theme on top of ttk's 'clam' (the only built-in theme that lets every colour be changed)."""
    style.theme_use("clam")
    style.configure(
        ".", background=BG, foreground=TEXT, fieldbackground=SURFACE, bordercolor=BORDER, darkcolor=BORDER,
        lightcolor=BORDER, troughcolor=SURFACE2, focuscolor=BORDER, insertcolor=ACCENT,
        selectbackground=ACCENT, selectforeground=ON_ACCENT,
    )
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Card.TLabel", background=SURFACE)
    style.configure("Title.TLabel", font=("Segoe UI", 26, "bold"))
    style.configure("Heading.TLabel", font=("Segoe UI", 16, "bold"))
    style.configure("Hint.TLabel", foreground=MUTED)
    style.configure("CardHint.TLabel", background=SURFACE, foreground=MUTED)
    style.configure("Section.TLabel", foreground=MUTED, font=("Segoe UI", 9, "bold"))
    style.configure("CardSection.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 9, "bold"))
    style.configure("Status.TLabel", background=SURFACE, font=("Segoe UI", 12, "bold"))
    style.configure("Check.TLabel", background=SURFACE, foreground=ACCENT, font=("Segoe UI", 22, "bold"))
    for name, colour in (("StepIdle", MUTED), ("StepNow", TEXT), ("StepDone", ACCENT)):
        style.configure(f"{name}.TLabel", background=SURFACE, foreground=colour, font=("Segoe UI", 9, "bold" if name == "StepNow" else "normal"))

    def flat_button(name, bg, fg, hover, pressed, pad, font, disabled_bg=SURFACE2, disabled_fg=MUTED, hover_fg=None):
        style.configure(
            name, background=bg, foreground=fg, bordercolor=bg, lightcolor=bg, darkcolor=bg, focuscolor=bg,
            relief="flat", borderwidth=1, padding=pad, font=font,
        )
        states = [("disabled", disabled_bg), ("pressed", pressed), ("active", hover)]
        style.map(
            name,
            background=states, bordercolor=states, lightcolor=states, darkcolor=states,
            foreground=[("disabled", disabled_fg)] + ([("active", hover_fg)] if hover_fg else []),
            relief=[("pressed", "flat")],
        )

    flat_button("Primary.TButton", ACCENT, ON_ACCENT, ACCENT_HI, ACCENT_LO, (px(26), px(11)), ("Segoe UI", 13, "bold"),
                disabled_bg=SURFACE2, disabled_fg="#6b7090")
    flat_button("Secondary.TButton", SURFACE2, TEXT, SURFACE3, BORDER, (px(18), px(10)), ("Segoe UI", 11), disabled_fg="#6b7090")
    flat_button("Small.TButton", SURFACE2, TEXT, SURFACE3, BORDER, (px(14), px(6)), ("Segoe UI", 10))
    flat_button("Chip.TButton", SURFACE, MUTED, SURFACE2, SURFACE3, (px(12), px(4)), ("Segoe UI", 10), hover_fg=TEXT)
    flat_button("Link.TButton", BG, MUTED, BG, BG, (px(8), px(4)), ("Segoe UI", 10), disabled_bg=BG, hover_fg=ACCENT)

    style.configure(
        "Search.TEntry", fieldbackground=SURFACE, foreground=TEXT, padding=(px(14), px(10)), borderwidth=1,
    )
    focused = [("focus", ACCENT)]
    style.map(
        "Search.TEntry", bordercolor=focused, lightcolor=focused, darkcolor=focused,
        fieldbackground=[("disabled", BG)], foreground=[("disabled", MUTED)],
    )

    # Radio buttons drawn as toggle buttons: ttk's own radio dot is a fixed-size bitmap that is tiny on high-DPI screens.
    state_bg = [("selected", SELECTED), ("active", SURFACE3)]
    for name, anchor in (("Choice.Toolbutton", "center"), ("ChoiceRow.Toolbutton", "w")):  # the row one fills a list
        style.configure(
            name, background=SURFACE2, foreground=TEXT, bordercolor=BORDER, lightcolor=SURFACE2, darkcolor=SURFACE2,
            focuscolor=SURFACE2, relief="flat", borderwidth=1, padding=(px(14), px(7)), font=("Segoe UI", 11),
            anchor=anchor,
        )
        style.map(
            name, background=state_bg, lightcolor=state_bg, darkcolor=state_bg,
            bordercolor=[("selected", ACCENT), ("active", MUTED)], foreground=[("selected", ACCENT)],
            relief=[("selected", "flat"), ("pressed", "flat")],
        )

    style.configure(
        "Meld.Horizontal.TProgressbar", troughcolor=SURFACE2, background=ACCENT, bordercolor=SURFACE2,
        lightcolor=ACCENT, darkcolor=ACCENT, thickness=px(8),
    )
    style.configure(
        "Vertical.TScrollbar", background=SURFACE3, troughcolor=LOG_BG, bordercolor=LOG_BG, arrowcolor=MUTED,
        lightcolor=SURFACE3, darkcolor=SURFACE3,
    )
    style.map("Vertical.TScrollbar", background=[("active", BORDER)])


def draw_logo(canvas, size: int) -> None:
    """The app icon, drawn with canvas primitives so no image library is needed at run time."""
    r = size * 0.2
    x1 = y1 = 1
    x2 = y2 = size - 1
    canvas.create_polygon(
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r,
        x1, y1 + r, x1, y1, smooth=True, fill=ICON_BG, outline=BORDER,
    )
    margin = size * 0.16
    slot = (size - 2 * margin) / len(WAVE)
    width = slot * 0.64
    for i, h in enumerate(WAVE):
        cx = margin + i * slot + slot / 2
        half = max(0.0, size * 0.62 * h / 2 - width / 2)  # round caps add width/2 at both ends
        canvas.create_line(
            cx, size / 2 - half, cx, size / 2 + half, width=width, capstyle="round",
            fill=ACCENT if i % 2 == 0 else CORAL,
        )


def style_titlebar(root) -> None:
    """Dark title bar on Windows 10/11 (Windows 11 also gets the window's own colours); silently skipped elsewhere."""
    try:
        import ctypes
        from ctypes import wintypes

        user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
        user32.GetParent.restype = wintypes.HWND
        user32.GetParent.argtypes = [wintypes.HWND]
        dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        root.update_idletasks()
        hwnd = user32.GetParent(root.winfo_id())

        def colorref(hex_colour: str) -> int:  # 0x00BBGGRR
            r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
            return (b << 16) | (g << 8) | r

        for attr, value in ((20, 1), (35, colorref(BG)), (36, colorref(TEXT))):  # dark mode, caption, caption text
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
    except Exception:
        pass


# ---------------------------------------------------------------- the window


def main(hook=None) -> None:
    """Open the window. `hook(root, widgets)` (tests only) is called once the window is up."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import tkinter.font as tkfont

    # A windowed Windows app has no console: make print() harmless.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens
    except Exception:
        pass

    root = tk.Tk()
    root.withdraw()  # shown once it is styled and placed, so it does not flash unstyled
    root.title(APP_NAME)
    root.configure(background=BG)
    icon = icon_path()
    if icon:
        try:
            root.iconbitmap(str(icon))
        except tk.TclError:
            pass

    scale = root.winfo_fpixels("1i") / 96  # pixel sizes below are for a 96-dpi screen; fonts scale by themselves

    def px(n: float) -> int:
        return int(round(n * scale))

    for font_name, size in (("TkDefaultFont", 11), ("TkTextFont", 11), ("TkMenuFont", 10)):
        tkfont.nametofont(font_name).configure(family="Segoe UI", size=size)
    style = ttk.Style()
    apply_theme(style, px)

    saved = load_saved()
    q: "queue.Queue[tuple]" = queue.Queue()
    cancel = threading.Event()
    answers: "queue.Queue" = queue.Queue()  # the picker dialog's answer, for the worker that is waiting for it
    state = {"worker": None, "result": None}

    query_var = tk.StringVar(value=saved.get("query", ""))
    clips_var = tk.IntVar(value=saved.get("clips", 100))
    quality_var = tk.IntVar(value=saved.get("quality", 720))
    keep_var = tk.BooleanVar(value=bool(saved.get("keep", False)))
    strict_var = tk.BooleanVar(value=bool(saved.get("strict", False)))
    login_var = tk.StringVar(value=saved.get("login") or "")  # a browser whose YouTube login to use, or ""
    folder_var =tk.StringVar(value=saved.get("folder", str(default_folder())))
    folder_shown = tk.StringVar(value=shorten_path(folder_var.get()))
    folder_var.trace_add("write", lambda *_: folder_shown.set(shorten_path(folder_var.get())))
    step_var = tk.StringVar(value=IDLE_TEXT)
    detail_var = tk.StringVar(value="")
    show_details = tk.BooleanVar(value=False)

    def card(parent, border: str = BORDER, pad: int = 14):
        """A rounded-looking panel: a 1 px border around a padded surface. Returns (outer, inner)."""
        outer = tk.Frame(parent, background=SURFACE, highlightthickness=1, highlightbackground=border, highlightcolor=border)
        inner = ttk.Frame(outer, style="Card.TFrame", padding=px(pad))
        inner.pack(fill="both", expand=True)
        return outer, inner

    frm = ttk.Frame(root, padding=(px(28), px(16), px(28), px(10)))
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(0, weight=1, minsize=px(680))
    frm.rowconfigure(10, weight=1)  # the details box takes any extra height
    wrapped: list[tuple[ttk.Label, int]] = []  # (label, pixels its wrap width stays short of the window's)

    # -- header
    header = ttk.Frame(frm)
    header.grid(row=0, column=0, sticky="w")
    logo = tk.Canvas(header, width=px(46), height=px(46), background=BG, highlightthickness=0)
    draw_logo(logo, px(46))
    logo.grid(row=0, column=0, padx=(0, px(14)))
    ttk.Label(header, text=APP_NAME, style="Title.TLabel").grid(row=0, column=1, sticky="w")
    intro = ttk.Label(
        frm, justify="left", style="Hint.TLabel",
        text="Type a band and a year (or a date). I'll look for phone videos of that concert online "
             "and combine them into one video with better sound.",
    )
    intro.grid(row=1, column=0, sticky="w", pady=(px(8), px(12)))
    wrapped.append((intro, px(56)))

    # -- what to look for
    entry = ttk.Entry(frm, textvariable=query_var, style="Search.TEntry", font=("Segoe UI", 14))
    entry.grid(row=3, column=0, sticky="ew")

    examples = ttk.Frame(frm)
    examples.grid(row=4, column=0, sticky="w", pady=(px(6), px(14)))
    ttk.Label(examples, text="Try", style="Hint.TLabel").pack(side="left", padx=(0, px(8)))

    def use_example(text: str) -> None:
        if str(entry["state"]) != "normal":
            return
        query_var.set(text)
        entry.focus_set()
        entry.icursor("end")

    for text in EXAMPLES:
        ttk.Button(examples, text=text, style="Chip.TButton", cursor="hand2", command=lambda t=text: use_example(t)).pack(
            side="left", padx=(0, px(6))
        )

    # -- options
    opts_card, opts = card(frm)
    opts_card.grid(row=5, column=0, sticky="ew")
    for row, (title, choices, var) in enumerate((
        ("How many videos", CLIP_CHOICES, clips_var),
        ("Picture quality", QUALITY_CHOICES, quality_var),
    )):
        ttk.Label(opts, text=title.upper(), style="CardSection.TLabel", width=17).grid(
            row=row, column=0, sticky="w", padx=(0, px(12)), pady=(px(10) if row else 0, 0)
        )
        toggles = ttk.Frame(opts, style="Card.TFrame")
        toggles.grid(row=row, column=1, sticky="w", pady=(px(10) if row else 0, 0))
        for text, n in choices:
            ttk.Radiobutton(toggles, text=text, value=n, variable=var, style="Choice.Toolbutton", cursor="hand2").pack(
                side="left", padx=(0, px(8))
            )

    # Strict: only videos whose title has every word typed (and the date, month and year). Fewer, but surely the right ones.
    ttk.Label(opts, text="SEARCH", style="CardSection.TLabel", width=17).grid(
        row=2, column=0, sticky="w", padx=(0, px(12)), pady=(px(10), 0)
    )
    strict_btn = ttk.Checkbutton(opts, style="Choice.Toolbutton", variable=strict_var, cursor="hand2")
    strict_btn.grid(row=2, column=1, sticky="w", pady=(px(10), 0))

    def show_strict(*_) -> None:
        strict_btn.configure(text=("✓ " if strict_var.get() else "") + "Strict: the title must have every word I typed")

    strict_var.trace_add("write", show_strict)
    show_strict()

    where = ttk.Frame(frm)
    where.grid(row=6, column=0, sticky="ew", pady=(px(12), 0))
    where.columnconfigure(1, weight=1)
    ttk.Label(where, text="Save results in").grid(row=0, column=0, sticky="w")
    ttk.Label(where, textvariable=folder_shown, style="Hint.TLabel").grid(row=0, column=1, sticky="w", padx=px(10))

    def choose_folder() -> None:
        d = filedialog.askdirectory(initialdir=folder_var.get() or str(Path.home()), title="Where should I save?")
        if d:
            folder_var.set(str(Path(d)))

    ttk.Button(where, text="Change...", style="Small.TButton", cursor="hand2", command=choose_folder).grid(row=0, column=2)

    def saved_searches() -> None:
        """What Meld keeps in the save folder so that a search run again is quick: say how much, and offer to clear it."""
        found = cache.searches(Path(folder_var.get()))
        if not found:
            messagebox.showinfo(APP_NAME, "Nothing is kept for later in that folder.")
            return
        size = sum(cache.folder_size(d / "preview") for d in found)
        if messagebox.askyesno(
            APP_NAME,
            f"So that a search you run again is quick, Meld keeps the small audio previews and what it worked out from "
            f"them for {len(found)} earlier search{'es' if len(found) != 1 else ''}: {format_size(size)}.\n\n"
            "Delete them? (The next run of those searches will download and sort them again. Your finished videos, "
            "and any videos you chose to keep, are not touched.)",
        ):
            for d in found:
                cache.forget(d)

    saved_btn = ttk.Button(where, text="Saved searches...", style="Small.TButton", cursor="hand2", command=saved_searches)
    saved_btn.grid(row=0, column=3, padx=(px(6), 0))

    buttons = ttk.Frame(frm)
    buttons.grid(row=7, column=0, sticky="ew", pady=(px(12), px(12)))
    start_btn = ttk.Button(buttons, text="Start", style="Primary.TButton", cursor="hand2")
    start_btn.pack(side="left")
    cancel_btn = ttk.Button(buttons, text="Cancel", style="Secondary.TButton", cursor="hand2", state="disabled")
    cancel_btn.pack(side="left", padx=px(12))

    # By default only the finished video and sound are kept; this keeps the downloaded videos too.
    keep_btn = ttk.Checkbutton(buttons, style="Choice.Toolbutton", variable=keep_var, cursor="hand2")
    keep_btn.pack(side="right")

    def show_keep(*_) -> None:
        keep_btn.configure(text=("✓ " if keep_var.get() else "") + "Keep the downloaded videos")

    keep_var.trace_add("write", show_keep)
    show_keep()

    # -- progress: four steps, a status line, a bar and the latest detail
    panel, panel_in = card(frm, pad=16)
    panel.grid(row=8, column=0, sticky="new")
    panel_in.columnconfigure(tuple(range(len(STEP_NAMES))), weight=1, uniform="steps")
    step_bars: list[tuple[tk.Frame, ttk.Label]] = []
    for i, name in enumerate(STEP_NAMES):
        cell = ttk.Frame(panel_in, style="Card.TFrame")
        cell.grid(row=0, column=i, sticky="ew", padx=(0, px(8)) if i < len(STEP_NAMES) - 1 else 0)
        seg = tk.Frame(cell, height=px(4), background=SURFACE3)
        seg.pack(fill="x")
        label = ttk.Label(cell, text=name, style="StepIdle.TLabel")
        label.pack(anchor="w", pady=(px(6), 0))
        step_bars.append((seg, label))

    def set_step(current: int) -> None:
        """Steps before `current` are done, `current` is running, the rest wait. -1: none started, len: all done."""
        for i, (seg, label) in enumerate(step_bars):
            if i < current:
                seg.configure(background=ACCENT)
                label.configure(style="StepDone.TLabel", text="✓ " + STEP_NAMES[i])
            elif i == current:
                seg.configure(background=CORAL)
                label.configure(style="StepNow.TLabel", text=STEP_NAMES[i])
            else:
                seg.configure(background=SURFACE3)
                label.configure(style="StepIdle.TLabel", text=STEP_NAMES[i])

    ttk.Label(panel_in, textvariable=step_var, style="Status.TLabel").grid(
        row=1, column=0, columnspan=len(STEP_NAMES), sticky="w", pady=(px(14), px(8))
    )
    bar = ttk.Progressbar(panel_in, mode="determinate", maximum=100, style="Meld.Horizontal.TProgressbar")
    bar.grid(row=2, column=0, columnspan=len(STEP_NAMES), sticky="ew")
    detail_label = ttk.Label(panel_in, textvariable=detail_var, style="CardHint.TLabel", justify="left")
    detail_label.grid(row=3, column=0, columnspan=len(STEP_NAMES), sticky="w", pady=(px(8), 0))
    wrapped.append((detail_label, px(90)))

    # -- result: replaces the progress panel when finished, so the window keeps its height
    result_card, result_in = card(frm, border=ACCENT, pad=14)
    result_card.grid(row=8, column=0, sticky="new")
    result_card.grid_remove()
    result_in.columnconfigure(1, weight=1)
    ttk.Label(result_in, text="✓", style="Check.TLabel").grid(row=0, column=0, rowspan=2, sticky="n", padx=(0, px(14)))
    ttk.Label(result_in, text="All done!", style="Status.TLabel").grid(row=0, column=1, sticky="w")
    result_label = ttk.Label(result_in, style="Card.TLabel", font=("Segoe UI", 12), justify="left")
    result_label.grid(row=1, column=1, sticky="w", pady=(px(4), 0))
    wrapped.append((result_label, px(130)))
    result_btns = ttk.Frame(result_in, style="Card.TFrame")
    result_btns.grid(row=2, column=1, sticky="w", pady=(px(12), 0))
    play_btn = ttk.Button(result_btns, text="Play the video", style="Primary.TButton", cursor="hand2")
    play_btn.pack(side="left")
    open_btn = ttk.Button(result_btns, text="Open the folder", style="Secondary.TButton", cursor="hand2")
    open_btn.pack(side="left", padx=px(12))

    # -- technical details (hidden until asked for)
    log_frame = tk.Frame(frm, background=LOG_BG, highlightthickness=1, highlightbackground=BORDER, highlightcolor=BORDER)
    log_text = tk.Text(
        log_frame, height=6, wrap="word", state="disabled", font=("Consolas", 9), background=LOG_BG,
        foreground="#c4c8de", insertbackground=ACCENT, selectbackground=SURFACE3, selectforeground=TEXT,
        relief="flat", borderwidth=0, highlightthickness=0, padx=px(10), pady=px(8),
    )
    scroll = ttk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=scroll.set)
    log_text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    footer = ttk.Frame(frm)
    footer.grid(row=11, column=0, sticky="ew", pady=(px(12), 0))
    footer.columnconfigure(2, weight=1)
    # padding=0 on the outer edges so the link text lines up with the content above
    details_btn = ttk.Button(footer, style="Link.TButton", cursor="hand2", text="Show details", padding=(0, px(4)))
    details_btn.grid(row=0, column=0, sticky="w")
    match_btn = ttk.Button(footer, style="Link.TButton", cursor="hand2", text="Show matching")
    match_btn.grid(row=0, column=1)
    login_btn = ttk.Button(footer, style="Link.TButton", cursor="hand2", text="YouTube login")
    login_btn.grid(row=0, column=3)
    update_btn = ttk.Button(footer, style="Link.TButton", cursor="hand2", text="Update the YouTube downloader")
    update_btn.grid(row=0, column=4)
    about_btn = ttk.Button(footer, style="Link.TButton", cursor="hand2", text="About", padding=(px(8), px(4), 0, px(4)))
    about_btn.grid(row=0, column=5)

    def keep_on_screen() -> None:
        """The window just grew: slide it up if it now runs off the bottom of the screen."""
        root.update_idletasks()
        overflow = root.winfo_y() + root.winfo_reqheight() - (root.winfo_screenheight() - px(60))
        if overflow > 0:
            root.geometry(f"+{root.winfo_x()}+{max(0, root.winfo_y() - overflow)}")

    # -- the matching, drawn as it happens: a window of its own beside this one, made the first time it is wanted
    match_theme = Theme(
        bg=BG, surface=SURFACE, surface2=SURFACE2, border=BORDER, text=TEXT, muted=MUTED, accent=ACCENT,
        good="#5ee0a0", bad=CORAL, on_bar="#15161f",
        palette=("#ffd23f", "#ff9f45", "#ff785a", "#f76ec0", "#a78bfa", "#6ec6ff", "#4fd1c5", "#a3e26a"),
    )
    view = {"win": None, "canvas": None, "last": None}  # the window, its canvas, and the newest (state, titles)

    def place_matching(win) -> None:
        """Put the matching window next to this one, as wide as is useful. If the screen has no room beside this
        window, slide this one to the left edge and let the other overlap only its right part, so that the buttons
        and the steps on its left stay visible."""
        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        x, y, w, h = root.winfo_x(), root.winfo_y(), root.winfo_width(), root.winfo_height()
        gap, want = px(14), px(900)
        if x + w + gap + want > sw and x > gap:  # not enough room on the right: slide this window to the left edge
            x = gap
            root.geometry(f"+{x}+{y}")
        right = x + w + gap
        overlap = min(max(0, right + want - (sw - gap)), int(w * 0.6))  # never cover more than 60% of it
        left = right - overlap
        width = max(px(560), min(sw - gap - left, px(1700)))
        height = min(max(h, px(560)), sh - px(80))
        win.geometry(f"{width}x{height}+{left}+{max(0, min(y, sh - height - px(70)))}")

    def ensure_matching():
        if view["win"] is None:
            win = tk.Toplevel(root)
            win.withdraw()
            win.title("Meld - the matching")
            win.configure(background=BG)
            if icon:
                try:
                    win.iconbitmap(str(icon))
                except tk.TclError:
                    pass
            canvas = MatchCanvas(win, match_theme, scale, height=px(420))
            canvas.pack(fill="both", expand=True, padx=px(10), pady=px(10))
            win.minsize(px(640), px(320))
            win.protocol("WM_DELETE_WINDOW", lambda: show_matching(False, by_user=True))
            place_matching(win)
            style_titlebar(win)
            view["win"], view["canvas"] = win, canvas
        return view["canvas"]

    def matching_visible() -> bool:
        return view["win"] is not None and str(view["win"].state()) == "normal"

    def show_matching(show: bool, by_user: bool = False) -> None:
        """Show or hide the window of the matching. It opens by itself when a run starts sorting, unless the person
        has closed it (`by_user`) since the program started."""
        if by_user:
            state["match_closed_by_user"] = not show
        if show == matching_visible():
            return
        if show:
            canvas = ensure_matching()
            view["win"].deiconify()
            if view["last"]:
                canvas.set_state(*view["last"])
            match_btn.configure(text="Hide matching")
        else:
            view["win"].withdraw()
            match_btn.configure(text="Show matching")

    def toggle_details() -> None:
        show_details.set(not show_details.get())
        if show_details.get():
            log_frame.grid(row=10, column=0, sticky="nsew", pady=(px(14), 0))
            details_btn.configure(text="Hide details")
            keep_on_screen()
        else:
            log_frame.grid_remove()
            details_btn.configure(text="Show details")

    details_btn.configure(command=toggle_details)
    match_btn.configure(command=lambda: show_matching(not matching_visible(), by_user=True))

    def add_log(line: str) -> None:
        log_text.configure(state="normal")
        log_text.insert("end", line + "\n")
        log_text.see("end")
        log_text.configure(state="disabled")

    def set_running(running: bool) -> None:
        start_btn.configure(state="disabled" if running else "normal")
        cancel_btn.configure(state="normal" if running else "disabled")
        entry.configure(state="disabled" if running else "normal")
        keep_btn.configure(state="disabled" if running else "normal")
        strict_btn.configure(state="disabled" if running else "normal")
        saved_btn.configure(state="disabled" if running else "normal")
        login_btn.configure(state="disabled" if running else "normal")

    def worker(s: Settings) -> None:
        s.project_dir.mkdir(parents=True, exist_ok=True)
        trim_log(s.project_dir / "meld.log")
        logfile = open(s.project_dir / "meld.log", "a", encoding="utf-8", errors="replace")

        def log(msg="") -> None:
            if cancel.is_set():
                raise Cancelled()
            text = str(msg)
            logfile.write(text + "\n")
            logfile.flush()
            q.put(("log", text))

        def stage(i: int, name: str) -> None:
            log(f"=== {name}")
            q.put(("stage", i, name))

        def choose_group(options):
            """Have the window let the user pick one of the groups, and wait for the answer."""
            q.put(("choose_group", options))
            while True:
                try:
                    answer = answers.get(timeout=0.2)
                except queue.Empty:
                    if cancel.is_set():
                        raise Cancelled()
                    continue
                if answer == "cancel":
                    raise Cancelled()
                return answer

        def on_match(match_state, titles) -> None:
            q.put(("match", match_state, titles))

        result = None
        try:
            result = run_pipeline(s, log, stage, choose_group, on_match)
        except Cancelled:
            q.put(("cancelled",))
        except BaseException as e:  # noqa: BLE001 - everything must reach the user
            logfile.write(traceback.format_exc())
            q.put(("error", friendly_error(e), traceback.format_exc()))
        finally:
            logfile.close()
        if result is not None:  # a cancelled or failed run keeps its downloads, so trying again is quick
            try:  # the log is closed now, so it may be trimmed too
                gone = clean_up_after(s)
            except OSError:
                gone = False
            if not gone:
                q.put(("log", f"Some working files could not be removed from {s.project_dir}"))
            q.put(("done", result))

    def start() -> None:
        query = query_var.get().strip()
        if not query:
            messagebox.showinfo(APP_NAME, "Please type which concert to look for, for example:  Metallica 2003")
            entry.focus_set()
            return
        s = Settings(
            query, Path(folder_var.get()), clips_var.get(), quality_var.get(), keep_var.get(), login_var.get() or None,
            strict_var.get(),
        )
        try:
            s.folder.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(s.folder).free / 1024**3
        except OSError as e:
            messagebox.showerror(APP_NAME, f"I can't use that folder:\n{e}")
            return
        if free < s.disk_needed_gb() and not messagebox.askyesno(
            APP_NAME,
            f"This could need about {s.disk_needed_gb():.0f} GB, but only {free:.0f} GB is free on that drive.\n\n"
            "Continue anyway? (Choosing fewer videos needs less space.)",
        ):
            return
        save_settings(s)
        cancel.clear()
        state["result"] = None
        result_label.configure(text="")
        result_card.grid_remove()
        view["last"] = None
        if view["canvas"] is not None:
            view["canvas"].set_state(None)
        panel.grid()
        set_step(-1)
        step_var.set("Starting...")
        detail_var.set("")
        bar.configure(mode="indeterminate")
        bar.start(12)
        set_running(True)
        state["worker"] = threading.Thread(target=worker, args=(s,), daemon=True)
        state["worker"].start()

    def stop() -> None:
        cancel.set()
        cancel_btn.configure(state="disabled")
        step_var.set("Stopping... (this can take a moment)")

    def finish(text: str) -> None:
        bar.stop()
        bar.configure(mode="determinate", value=0)
        set_running(False)
        step_var.set(text)

    def show_error(message: str, details: str) -> None:
        finish("It stopped")
        add_log(details)
        messagebox.showerror(APP_NAME, message + "\n\n(Choose \"Show details\" for technical information.)")

    def make_modal(title: str):
        """A dialog window over the main one, not shown yet. Returns (window, the frame to put its content in)."""
        win = tk.Toplevel(root)
        win.withdraw()
        win.title(title)
        win.configure(background=BG)
        win.transient(root)
        win.resizable(False, False)
        if icon:
            try:
                win.iconbitmap(str(icon))
            except tk.TclError:
                pass
        body = ttk.Frame(win, padding=(px(26), px(22), px(26), px(18)))
        body.pack(fill="both", expand=True)
        return win, body

    def show_modal(win) -> None:
        win.update_idletasks()  # centre it over the main window, style it, then show it
        x = root.winfo_rootx() + (root.winfo_width() - win.winfo_reqwidth()) // 2
        y = root.winfo_rooty() + max(0, (root.winfo_height() - win.winfo_reqheight()) // 3)
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        style_titlebar(win)
        win.deiconify()
        try:
            win.wait_visibility()
            win.grab_set()  # nothing else in the app to click while it waits
        except tk.TclError:
            pass
        win.focus_force()

    def ask_group(options) -> None:
        """Several groups of videos line up with each other, but not with the others: let the user pick one. The answer
        (one of `options`, or "cancel") goes to `answers`."""
        shown = options[:MAX_GROUPS_SHOWN]
        heading = "Which group?"
        earlier_status = step_var.get()
        step_var.set("Waiting for you to choose a group")

        win, body = make_modal(heading)
        ttk.Label(body, text=heading, style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            body, style="Hint.TLabel", justify="left", wraplength=px(540),
            text=f"By their sound, the videos fall into {len(options)} groups that don't line up with each other: "
                 "different concerts, or parts of one that never overlap. Pick the one to combine. Only its "
                 "videos are downloaded.",
        ).pack(anchor="w", pady=(px(6), px(14)))

        choice = tk.IntVar(value=0)  # index into `shown`
        for i, c in enumerate(shown):
            ttk.Radiobutton(
                body, text=f"{c.label}      {c.count} videos", value=i, variable=choice, style="ChoiceRow.Toolbutton",
                cursor="hand2",
            ).pack(fill="x", pady=(0, px(6)))
        if len(options) > len(shown):
            ttk.Label(
                body, style="Hint.TLabel", text=f"({len(options) - len(shown)} groups with fewer videos are not shown)",
            ).pack(anchor="w", pady=(px(8), 0))

        def finish_choice(answer) -> None:
            answers.put(answer)
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()
            step_var.set(earlier_status)

        row = ttk.Frame(body)
        row.pack(fill="x", pady=(px(18), 0))
        ttk.Button(
            row, text="Continue", style="Primary.TButton", cursor="hand2",
            command=lambda: finish_choice(shown[choice.get()]),
        ).pack(side="left")
        ttk.Button(
            row, text="Cancel", style="Secondary.TButton", cursor="hand2", command=lambda: finish_choice("cancel"),
        ).pack(side="left", padx=px(12))
        win.protocol("WM_DELETE_WINDOW", lambda: finish_choice("cancel"))
        win.bind("<Escape>", lambda e: finish_choice("cancel"))
        win.bind("<Return>", lambda e: finish_choice(shown[choice.get()]))

        show_modal(win)

    def login_label() -> str:
        return f"YouTube login: {LOGIN_BROWSERS.get(login_var.get(), login_var.get())}" if login_var.get() else "YouTube login"

    def ask_login() -> None:
        """Let the user choose a browser whose YouTube login the downloads use, for when YouTube asks to confirm they
        are not a bot. Only saved on Save."""
        win, body = make_modal("YouTube login")
        ttk.Label(body, text="YouTube login", style="Heading.TLabel").pack(anchor="w")
        for text, pad in (
            ("Use this if YouTube keeps saying it wants to confirm you're not a bot. Meld can use the YouTube login that "
             "is already saved in one of your browsers, so sign in to YouTube in that browser first.", (px(6), px(8))),
            ("Meld reads it only for this, keeps it in memory while it runs, and never saves it or sends it anywhere but "
             "to YouTube. Downloads go a little slower and gentler. Downloading a lot with an account can get it limited "
             "by Google, so you may prefer a spare account.", (0, px(8))),
            ("Firefox is the most reliable. Chrome and Edge must be closed while Meld runs, and may not work at all.",
             (0, px(14))),
        ):
            ttk.Label(body, style="Hint.TLabel", justify="left", wraplength=px(540), text=text).pack(anchor="w", pady=pad)

        offered = installed_browsers() or list(LOGIN_BROWSERS)
        if login_var.get() and login_var.get() not in offered:
            offered.append(login_var.get())  # one chosen earlier stays visible, even if it seems to have gone
        choice = tk.StringVar(value=login_var.get())
        ttk.Radiobutton(
            body, text="Don't use a login (the default)", value="", variable=choice, style="ChoiceRow.Toolbutton",
            cursor="hand2",
        ).pack(fill="x", pady=(0, px(6)))
        for browser in offered:
            ttk.Radiobutton(
                body, text=LOGIN_BROWSERS.get(browser, browser), value=browser, variable=choice,
                style="ChoiceRow.Toolbutton", cursor="hand2",
            ).pack(fill="x", pady=(0, px(6)))

        def close(save: bool) -> None:
            if save:
                login_var.set(choice.get())
                login_btn.configure(text=login_label())
                save_settings(Settings(
                    query_var.get().strip(), Path(folder_var.get()), clips_var.get(), quality_var.get(), keep_var.get(),
                    login_var.get() or None, strict_var.get(),
                ))
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()

        row = ttk.Frame(body)
        row.pack(fill="x", pady=(px(14), 0))
        ttk.Button(row, text="Save", style="Primary.TButton", cursor="hand2", command=lambda: close(True)).pack(side="left")
        ttk.Button(
            row, text="Cancel", style="Secondary.TButton", cursor="hand2", command=lambda: close(False),
        ).pack(side="left", padx=px(12))
        win.protocol("WM_DELETE_WINDOW", lambda: close(False))
        win.bind("<Escape>", lambda e: close(False))
        win.bind("<Return>", lambda e: close(True))
        show_modal(win)

    def poll() -> None:
        newest_match = None  # comparisons can come faster than the screen is drawn: only the latest is worth showing
        try:
            while True:
                msg = q.get_nowait()
                kind = msg[0]
                if kind == "match":
                    newest_match = msg
                elif kind == "choose_group":
                    ask_group(msg[1])
                elif kind == "stage":
                    step_var.set(f"Step {msg[1] + 1} of {len(STAGES)}: {msg[2]}")
                    set_step(msg[1])
                    bar.stop()
                    bar.configure(mode="indeterminate")
                    bar.start(12)
                elif kind == "log":
                    add_log(msg[1])
                    line = msg[1].strip()
                    if line and not line.startswith("==="):
                        detail_var.set(line[:140])
                    prog = parse_progress(msg[1])
                    if prog:
                        bar.stop()
                        bar.configure(mode="determinate", value=100 * prog[0] / prog[1])
                elif kind == "done":
                    r = msg[1]
                    state["result"] = r
                    finish("All done!")
                    set_step(len(STAGES))
                    bar.configure(value=100)
                    detail_var.set("")
                    result_label.configure(
                        text=f"Made a {r['minutes']:.0f}-minute video from {r['used']} phone videos"
                             + (f" ({r['skipped']} others didn't match)." if r["skipped"] else ".")
                             + f"\nSaved as {r['video'].name}"
                    )
                    play_btn.configure(command=lambda p=r["video"]: os.startfile(p))
                    open_btn.configure(command=lambda p=r["folder"]: os.startfile(p))
                    panel.grid_remove()
                    result_card.grid()
                elif kind == "cancelled":
                    finish("Stopped")
                    detail_var.set("You can start again any time; videos already downloaded are kept.")
                elif kind == "error":
                    show_error(msg[1], msg[2])
                elif kind == "info":
                    detail_var.set("")
                    messagebox.showinfo(APP_NAME, msg[1])
        except queue.Empty:
            pass
        if newest_match is not None:
            view["last"] = (newest_match[1], newest_match[2])
            if not state.get("match_closed_by_user"):
                show_matching(True)
            if matching_visible():
                view["canvas"].set_state(*view["last"])
        root.after(100, poll)

    def on_close() -> None:
        w = state["worker"]
        if w is not None and w.is_alive() and not messagebox.askyesno(APP_NAME, "I'm still working. Stop and close?"):
            return
        cancel.set()
        root.destroy()

    def update_downloader() -> None:
        def go() -> None:
            try:
                v = ytdlp.update(lambda *_: None)
                q.put(("log", f"YouTube downloader updated to {v}."))
                q.put(("info", f"The YouTube downloader is up to date ({v})."))
            except Exception as e:  # noqa: BLE001
                q.put(("error", friendly_error(e), traceback.format_exc()))

        detail_var.set("Updating the YouTube downloader...")
        threading.Thread(target=go, daemon=True).start()

    def about() -> None:
        messagebox.showinfo(
            APP_NAME,
            f"{APP_NAME}\n\nFinds phone videos of a concert, lines them up by their sound and combines them "
            "into one video with the best sound.\n\nFor personal use only. Downloading videos from YouTube may go "
            "against its terms, and concert footage is usually copyrighted.",
        )

    login_btn.configure(text=login_label(), command=ask_login)
    update_btn.configure(command=update_downloader)
    about_btn.configure(command=about)
    start_btn.configure(command=start)
    cancel_btn.configure(command=stop)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Return>", lambda e: start() if str(start_btn["state"]) == "normal" else None)

    # Keep the YouTube downloader fresh (YouTube breaks old versions); quietly ignore being offline.
    def refresh_downloader() -> None:
        try:
            v = ytdlp.update_if_stale(14, lambda *_: None)
            if v:
                q.put(("log", f"YouTube downloader updated to {v}."))
        except Exception:
            pass

    threading.Thread(target=refresh_downloader, daemon=True).start()

    def fit_text(_event=None) -> None:
        width = max(px(300), frm.winfo_width())
        for label, margin in wrapped:
            label.configure(wraplength=max(px(200), width - margin))

    set_step(-1)
    fit_text()
    frm.bind("<Configure>", fit_text)
    root.minsize(px(560), px(420))

    # Place the finished window in the upper middle of the screen, then show it.
    root.update_idletasks()
    x = max(0, (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2)
    room = root.winfo_screenheight() - root.winfo_reqheight() - px(90)  # title bar and taskbar
    y = max(0, room // 3)
    root.geometry(f"+{x}+{y}")
    style_titlebar(root)
    root.deiconify()

    root.after(100, poll)
    if hook:
        widgets = {
            "query": query_var, "folder": folder_var, "clips": clips_var, "quality": quality_var, "login": login_var, "match_btn": match_btn, "match_view": view,
            "start": start_btn, "step": step_var, "detail": detail_var, "result": result_label, "play": play_btn,
            "queue": q, "keep": keep_var, "strict": strict_var,
        }
        root.after(300, hook, root, widgets)
    entry.focus_set()
    root.mainloop()


if __name__ == "__main__":
    main()
