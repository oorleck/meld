"""What the matching looks like while it runs: a snapshot of its state, and that laid out as timelines.

The matching (sync.py) emits a `MatchState` whenever something changes. `layout()` turns one into rectangles: one lane
per group of clips that line up, each clip a bar where it sits in the group's time, and the clips not sorted yet in a
tray below. It knows nothing of any window, so it can be tested; matchcanvas.py draws it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import PurePath

from .naming import is_filler, shared_words


@dataclass(frozen=True)
class MatchState:
    """The matching at one moment. Clip names are the files' names."""

    durations: dict[str, float]  # every clip: its length in seconds
    groups: tuple[dict[str, float], ...] = ()  # each group: clip -> where it starts in the group's own time (s)
    pending: tuple[str, ...] = ()  # clips not looked at yet, in the order they will be
    current: str | None = None  # the clip being sorted now
    compare: tuple[str, str, float | None] | None = None  # (that clip, the one it is being compared with, score or None)
    phase: int = 1  # 1: the first sweep, 2: the second look at clips that matched nothing
    compared: int = 0  # comparisons made so far
    min_z: float = 10.0  # the score a match needs
    doubted: bool = False  # the pair being compared scored enough, but did not hold through the overlap: not a match
    chosen: frozenset[str] | None = None  # the clips of the group that is used, once sorting is over
    finished: bool = False


@dataclass(frozen=True)
class Metrics:
    """Sizes in pixels. The window scales them for the screen."""

    pad: float = 12
    head_h: float = 26  # the line of text on top
    lane_head_h: float = 18
    bar_h: float = 16
    min_bar_h: float = 7
    max_bar_h: float = 34  # bars grow this thick when there is room to spare
    label_w: float = 74  # room for the words in front of each row of chips in the tray
    min_scale: float = 0.5  # pixels per second: the least a group is stretched to (a longer one scrolls sideways)
    bar_gap: float = 3
    lane_gap: float = 10
    axis_h: float = 18
    tray_h: float = 62
    chip: float = 9
    chip_gap: float = 3
    char_w: float = 6.6  # about how wide a character of a bar's label is
    min_label_chars: int = 5

    def scaled(self, k: float) -> "Metrics":
        return Metrics(**{name: (value if name == "min_label_chars" else value * k) for name, value in self.__dict__.items()})


@dataclass
class Bar:
    name: str
    x: float
    y: float
    w: float
    h: float
    lane: int
    label: str  # what fits on the bar ("" when it is too narrow for any)
    title: str  # the whole title, for hovering
    start: float  # where it starts in its group, seconds
    length: float
    chosen: bool = False


@dataclass
class Lane:
    index: int
    y: float
    h: float
    label: str
    count: int
    seconds: float  # how long the group is
    chosen: bool = False
    dim: bool = False  # another group was chosen


@dataclass
class Chip:
    name: str
    x: float
    y: float
    size: float
    kind: str  # "current", "waiting" or "unmatched"
    title: str


@dataclass
class Link:
    """The comparison going on: from the clip being sorted to the one it is compared with."""

    src: str
    dst: str
    z: float | None
    ok: bool  # whether the score is enough to match
    doubted: bool = False  # the score is enough, but the match did not hold through the overlap


@dataclass
class Layout:
    """Where everything goes. `width` x `height` is the window's view. The lanes, their bars and the time marks are in
    the content, which is as big as it needs to be (`content_w` x `content_h`, never less than the view) and scrolls;
    the headline on top and the tray below (`tray_y`, `axis_y`, the chips' y) stay in the view, whatever is scrolled to,
    so their y is where they are in the view. The chips' x is in the content: a long queue scrolls sideways."""

    width: float
    height: float
    content_w: float = 0.0
    content_h: float = 0.0
    lanes: list[Lane] = field(default_factory=list)
    bars: list[Bar] = field(default_factory=list)
    chips: list[Chip] = field(default_factory=list)
    ticks: list[tuple[float, str]] = field(default_factory=list)
    axis_y: float = 0.0
    tray_y: float = 0.0
    lanes_bottom: float = 0.0  # where the last lane ends, in the content
    scale: float = 0.0  # pixels per second
    zoom: float = 1.0  # how much of the usual scale that is (below 1: shrunk, down to the whole group in view)
    waiting: int = 0
    unmatched: int = 0
    headline: str = ""
    link: Link | None = None


def clock(seconds: float) -> str:
    """'45 s', '12 min', '1 h 05 min'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m = round(s / 60)
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


MAX_CONTENT = 60000.0  # pixels: the widest the content is stretched to, however far one zooms in
_STEPS = [10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400]
_WORD = re.compile(r"[\w'’&]+")
_REMARK = re.compile(r"\([^)]*\)|\[[^\]]*\]")  # (Live at Wembley 2025), [4K]


def make_labels(names: list[str], titles: dict[str, str]) -> dict[str, str]:
    """A short label for each clip: its title without the words nearly all the titles have (the band, the year), and
    without bracketed remarks that are only such words or filler ('(Live at Wembley 2025)'), which say nothing about which
    clip it is. Falls back to the whole title, then to the file's name."""
    def title_of(name: str) -> str:
        return titles.get(name) or titles.get(PurePath(name).stem) or ""

    known = [title_of(n) for n in names if title_of(n)]
    common = {w.casefold() for w in shared_words(known)} if len(known) >= 3 else set()
    def dull(word: str) -> bool:
        return word.casefold() in common or is_filler(word) or word.isdigit()

    def unremarkable(match: re.Match) -> str:  # a bracketed remark with nothing in it that tells clips apart
        return "" if all(dull(w) for w in _WORD.findall(match.group())) else match.group()

    labels = {}
    for name in names:
        title = title_of(name)
        kept = [w for w in _WORD.findall(_REMARK.sub(unremarkable, title)) if w.casefold() not in common]
        labels[name] = " ".join(kept) if kept else (title or PurePath(name).stem)
    return labels


def rank_groups(groups, durations: dict[str, float]) -> list[dict[str, float]]:
    """Biggest first, then longest: the order the lanes go in."""
    def span(members: dict[str, float]) -> float:
        return max(o + durations.get(n, 0.0) for n, o in members.items()) - min(members.values())

    return sorted((dict(g) for g in groups if g), key=lambda g: (-len(g), -span(g)))


def _fit(text: str, width: float, m: Metrics) -> str:
    room = int((width - 8) / m.char_w)
    if room < m.min_label_chars:
        return ""
    return text if len(text) <= room else text[: max(1, room - 1)].rstrip() + "…"


def _rows(members: dict[str, float], durations: dict[str, float]) -> dict[str, int]:
    """Which row each clip goes in so that none overlaps another in its row: first row it fits in."""
    ends: list[float] = []
    rows: dict[str, int] = {}
    for name, start in sorted(members.items(), key=lambda kv: (kv[1], kv[0])):
        for r, end in enumerate(ends):
            if start >= end + 0.5:
                ends[r] = start + durations.get(name, 0.0)
                rows[name] = r
                break
        else:
            ends.append(start + durations.get(name, 0.0))
            rows[name] = len(ends) - 1
    return rows


def tick_text(seconds: float) -> str:
    """'0:00', '1:30', '1:05:00': a mark on the time axis."""
    h, rest = divmod(int(round(seconds)), 3600)
    mins, secs = divmod(rest, 60)
    return f"{h}:{mins:02d}:{secs:02d}" if h else f"{mins}:{secs:02d}"


def _ticks(scale: float, span: float, m: Metrics, x0: float) -> list[tuple[float, str]]:
    step = next((s for s in _STEPS if s * scale >= 64), _STEPS[-1])
    return [(x0 + t * scale, tick_text(t)) for t in range(0, int(span) + 1, step)]


def layout(
    state: MatchState, labels: dict[str, str], titles: dict[str, str], width: float, height: float, m: Metrics | None = None,
    zoom: float = 1.0,
) -> Layout:
    """`state` as rectangles, for a view of `width` x `height`. `zoom` stretches the time scale (1: the usual one)."""
    m = m or Metrics()
    out = Layout(width, height)
    total = len(state.durations)
    placed = {n for g in state.groups for n in g}

    def title_of(name: str) -> str:
        return titles.get(name) or titles.get(PurePath(name).stem) or PurePath(name).stem

    # --- what is in lanes, and what waits in the tray
    ranked = rank_groups(state.groups, state.durations)
    lane_groups = [g for g in ranked if len(g) >= 2]
    unmatched = [n for g in ranked if len(g) == 1 for n in g]
    current_placed = state.current in placed if state.current else True
    out.waiting = len(state.pending)
    out.unmatched = len(unmatched)

    sorted_n = total - len(state.pending) - (0 if current_placed else 1)
    if state.finished:
        out.headline = f"Sorted: {len(lane_groups)} group{'s' if len(lane_groups) != 1 else ''}" + (
            "  ·  the highlighted one is used" if state.chosen and len(lane_groups) > 1 else "")
    elif state.phase == 2:
        out.headline = "Second look at the clips that matched nothing"
    elif total:
        out.headline = f"Sorting {sorted_n} of {total}"
    else:
        out.headline = "Waiting for the clips"
    if state.compared and not state.finished:
        out.headline += f"  ·  {state.compared} comparisons"

    # --- the areas: the headline on top and the tray below stay in the view; between them the lanes scroll
    top = m.head_h
    tray_y = height - m.tray_h
    axis_y = tray_y - m.axis_h
    room = axis_y - 4 + m.lane_gap  # the view's part for lanes, counting the gap that follows the last one
    bottom_h = height - (axis_y - 4)  # what the tray and the time marks take up below the lanes
    out.tray_y, out.axis_y = tray_y, axis_y
    usable = max(40.0, width - 2 * m.pad)
    seen_span = max((max(o + state.durations.get(n, 0.0) for n, o in g.items()) - min(g.values())) for g in lane_groups) if lane_groups else 0.0
    span = max(60.0, seen_span)
    fit = usable / span  # the scale at which the longest group is all in view
    base = max(fit, m.min_scale)  # what is usual: that, unless it would make the bars too small to read
    out.scale = min(max(fit, base * zoom), max(fit, MAX_CONTENT / span))
    out.zoom = out.scale / base

    # --- lanes, biggest first, each as tall as its rows need: the bars are the size that reads well (thicker when there is
    # room to spare) and what does not fit in the view is scrolled to
    rows_of = [_rows(g, state.durations) for g in lane_groups]
    gap = m.bar_gap

    def lane_height(i: int, bar_h: float) -> float:
        return m.lane_head_h + (max(rows_of[i].values()) + 1) * (bar_h + gap) + m.lane_gap

    bar_h = m.bar_h
    if lane_groups:
        while bar_h < m.max_bar_h and top + sum(lane_height(i, bar_h + 1) for i in range(len(lane_groups))) <= room:
            bar_h += 1
    y = top
    for i, g in enumerate(lane_groups):
        h = lane_height(i, bar_h)
        lo = min(g.values())
        length_of_group = max(o + state.durations.get(n, 0.0) for n, o in g.items()) - lo
        chosen = bool(state.chosen and state.chosen & set(g))
        out.lanes.append(Lane(
            i, y, h - m.lane_gap, f"Group {i + 1}  ·  {len(g)} clips  ·  {clock(length_of_group)}",
            len(g), length_of_group, chosen=chosen, dim=bool(state.chosen) and not chosen,
        ))
        for name, start in g.items():
            length = state.durations.get(name, 0.0)
            w = max(3.0, length * out.scale)
            label = labels.get(name) or PurePath(name).stem
            out.bars.append(Bar(
                name, m.pad + (start - lo) * out.scale, y + m.lane_head_h + rows_of[i][name] * (bar_h + gap), w, bar_h, i,
                _fit(label, w, m), title_of(name), start - lo, length, chosen=bool(state.chosen) and name in state.chosen,
            ))
        y += h
    out.lanes_bottom = y - m.lane_gap if lane_groups else top

    # --- the tray: the clip being sorted, then those waiting, then those that matched nothing (yet). Every clip has its
    # chip, in a row that goes on as far as it must
    def place_chips(names: list[str], kind: str, row_y: float, x0: float) -> float:
        for k, name in enumerate(names):
            out.chips.append(Chip(name, x0 + k * (m.chip + m.chip_gap), row_y, m.chip, kind, title_of(name)))
        return x0 + len(names) * (m.chip + m.chip_gap)

    label_w = m.label_w
    row1, row2 = tray_y + 6, tray_y + 6 + m.chip + 16
    if state.current and not current_placed:
        out.chips.append(Chip(state.current, m.pad + label_w, row1, m.chip * 1.5, "current", title_of(state.current)))
    waiting_x = m.pad + label_w + (m.chip * 1.5 + m.chip_gap * 2 if state.current and not current_placed else 0)
    tray_end = max(place_chips(list(state.pending), "waiting", row1, waiting_x), place_chips(unmatched, "unmatched", row2, m.pad + label_w))
    if state.current in unmatched:  # a clip getting its second look: it is in the tray, and it is the one at work
        for chip in out.chips:
            if chip.name == state.current:
                chip.kind = "current"

    out.content_w = max(width, (2 * m.pad + seen_span * out.scale) if lane_groups else 0.0, tray_end + m.pad)
    out.content_h = max(height, out.lanes_bottom + bottom_h)
    out.ticks = _ticks(out.scale, (out.content_w - 2 * m.pad) / out.scale if out.scale else 0, m, m.pad) if lane_groups else []

    # --- the comparison going on
    if state.compare:
        src, dst, z = state.compare
        enough = z is not None and z >= state.min_z
        out.link = Link(src, dst, z, enough and not state.doubted, enough and state.doubted)
    return out


def bar_lookup(lay: Layout) -> dict[str, Bar]:
    return {b.name: b for b in lay.bars}


def chip_lookup(lay: Layout) -> dict[str, Chip]:
    return {c.name: c for c in lay.chips}


def fit_check(lay: Layout) -> list[str]:
    """Problems with a layout (for tests): bars that stick out of the content, or overlap in one row of one lane."""
    problems = []
    by_row: dict[tuple[int, float], list[Bar]] = {}
    for b in lay.bars:
        if b.x < -0.01 or b.x + b.w > lay.content_w + 0.01 or b.y + b.h > lay.content_h + 0.01:
            problems.append(f"{b.name} sticks out")
        by_row.setdefault((b.lane, round(b.y, 3)), []).append(b)
    for bars in by_row.values():
        bars.sort(key=lambda b: b.x)
        for a, b in zip(bars, bars[1:]):
            if a.x + a.w > b.x + 0.01 and not math.isclose(a.x + a.w, b.x, abs_tol=0.01):
                problems.append(f"{a.name} overlaps {b.name}")
    return problems
