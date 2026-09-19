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


@dataclass
class Layout:
    width: float
    height: float
    lanes: list[Lane] = field(default_factory=list)
    bars: list[Bar] = field(default_factory=list)
    chips: list[Chip] = field(default_factory=list)
    hidden_lanes: int = 0  # groups that did not fit
    hidden_bars: int = 0  # clips of the first group that did not fit, even with the thinnest bars
    hidden_chips: dict[str, int] = field(default_factory=dict)  # kind -> how many did not fit
    ticks: list[tuple[float, str]] = field(default_factory=list)
    axis_y: float = 0.0
    tray_y: float = 0.0
    scale: float = 0.0  # pixels per second
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
) -> Layout:
    """`state` as rectangles in a `width` x `height` area."""
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

    # --- the areas
    top = m.head_h
    tray_y = height - m.tray_h
    axis_y = tray_y - m.axis_h
    lanes_bottom = axis_y - 4
    out.tray_y, out.axis_y = tray_y, axis_y
    usable = max(40.0, width - 2 * m.pad)
    seen_span = max((max(o + state.durations.get(n, 0.0) for n, o in g.items()) - min(g.values())) for g in lane_groups) if lane_groups else 0.0
    out.scale = usable / max(60.0, seen_span)

    # --- lanes: as many as fit, biggest first. To get the first one in: thinner bars, then tighter gaps, then thinner
    # still; and if even that is not enough, the rows that do not fit are left out (and counted)
    rows_of = [_rows(g, state.durations) for g in lane_groups]
    room = lanes_bottom + m.lane_gap  # the lanes' area, counting the gap that follows the last one

    def lane_height(i: int, bar_h: float, gap: float) -> float:
        return m.lane_head_h + (max(rows_of[i].values()) + 1) * (bar_h + gap) + m.lane_gap

    def lanes_that_fit(bh: float, gp: float) -> int:
        y, n = top, 0
        for i in range(len(lane_groups)):
            h = lane_height(i, bh, gp)
            if n and y + h > room:
                break
            y, n = y + h, n + 1
        return n

    bar_h, gap = m.bar_h, m.bar_gap
    while lane_groups and top + lane_height(0, bar_h, gap) > room:
        if bar_h > m.min_bar_h:
            bar_h -= 1
        elif gap > 1:
            gap = 1.0
        elif bar_h > 3:
            bar_h -= 1
        else:
            break
    if lane_groups and bar_h == m.bar_h:  # room to spare: thicker bars, as long as every lane that fits still does
        n0 = lanes_that_fit(bar_h, gap)
        while bar_h < m.max_bar_h and top + lane_height(0, bar_h + 1, gap) <= room and lanes_that_fit(bar_h + 1, gap) >= n0:
            bar_h += 1
    y = top
    shown = 0
    for i, g in enumerate(lane_groups):
        h = lane_height(i, bar_h, gap)
        if shown and y + h > room:
            break
        max_rows = max(1, int((room - y - m.lane_gap - m.lane_head_h) // (bar_h + gap)))
        lo = min(g.values())
        span = max(o + state.durations.get(n, 0.0) for n, o in g.items()) - lo
        chosen = bool(state.chosen and state.chosen & set(g))
        hidden = sum(1 for n in g if rows_of[i][n] >= max_rows)
        out.lanes.append(Lane(
            i, y, min(h, room - y) - m.lane_gap,
            f"Group {i + 1}  ·  {len(g)} clips  ·  {clock(span)}" + (f"  ·  {hidden} not shown" if hidden else ""),
            len(g), span, chosen=chosen, dim=bool(state.chosen) and not chosen,
        ))
        out.hidden_bars += hidden
        for name, start in g.items():
            row = rows_of[i][name]
            if row >= max_rows:
                continue
            length = state.durations.get(name, 0.0)
            w = max(3.0, length * out.scale)
            label = labels.get(name) or PurePath(name).stem
            out.bars.append(Bar(
                name, m.pad + (start - lo) * out.scale, y + m.lane_head_h + row * (bar_h + gap), w, bar_h, i,
                _fit(label, w, m), title_of(name), start - lo, length, chosen=bool(state.chosen) and name in state.chosen,
            ))
        y += h
        shown += 1
    out.hidden_lanes = len(lane_groups) - shown
    out.ticks = _ticks(out.scale, usable / out.scale if out.scale else 0, m, m.pad) if lane_groups else []

    # --- the tray: the clip being sorted, then those waiting, then those that matched nothing (yet)
    def place_chips(names: list[str], kind: str, row_y: float, x0: float, room: float) -> None:
        per_row = max(1, int(room // (m.chip + m.chip_gap)))
        fit = len(names) if len(names) <= per_row else max(0, per_row - 3)  # leave room for the '+N'
        for k, name in enumerate(names[:fit]):
            out.chips.append(Chip(name, x0 + k * (m.chip + m.chip_gap), row_y, m.chip, kind, title_of(name)))
        if len(names) > fit:
            out.hidden_chips[kind] = len(names) - fit

    label_w = m.label_w
    row1, row2 = tray_y + 6, tray_y + 6 + m.chip + 16
    if state.current and not current_placed:
        out.chips.append(Chip(state.current, m.pad + label_w, row1, m.chip * 1.5, "current", title_of(state.current)))
    waiting_x = m.pad + label_w + (m.chip * 1.5 + m.chip_gap * 2 if state.current and not current_placed else 0)
    place_chips(list(state.pending), "waiting", row1, waiting_x, width - m.pad - waiting_x - 40)
    place_chips(unmatched, "unmatched", row2, m.pad + label_w, width - 2 * m.pad - label_w - 40)
    if state.current in unmatched:  # a clip getting its second look: it is in the tray, and it is the one at work
        for chip in out.chips:
            if chip.name == state.current:
                chip.kind = "current"

    # --- the comparison going on
    if state.compare:
        src, dst, z = state.compare
        out.link = Link(src, dst, z, z is not None and z >= state.min_z)
    return out


def bar_lookup(lay: Layout) -> dict[str, Bar]:
    return {b.name: b for b in lay.bars}


def chip_lookup(lay: Layout) -> dict[str, Chip]:
    return {c.name: c for c in lay.chips}


def fit_check(lay: Layout) -> list[str]:
    """Problems with a layout (for tests): bars that stick out, or overlap in one row of one lane."""
    problems = []
    by_row: dict[tuple[int, float], list[Bar]] = {}
    for b in lay.bars:
        if b.x < -0.01 or b.x + b.w > lay.width + 0.01 or b.y + b.h > lay.axis_y + 0.01:
            problems.append(f"{b.name} sticks out")
        by_row.setdefault((b.lane, round(b.y, 3)), []).append(b)
    for bars in by_row.values():
        bars.sort(key=lambda b: b.x)
        for a, b in zip(bars, bars[1:]):
            if a.x + a.w > b.x + 0.01 and not math.isclose(a.x + a.w, b.x, abs_tol=0.01):
                problems.append(f"{a.name} overlaps {b.name}")
    return problems
