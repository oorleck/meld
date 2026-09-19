"""The window's live view of the matching: draws what matchview.layout() works out, and animates between states.

One lane per group of clips that line up, a bar for each clip where it sits in its group's time (with its title), the
clips not sorted yet in a tray at the bottom, and a line between the two clips being compared with the score it got.
Bars slide from the tray into their place as they are matched, and slide again when groups merge.
"""
from __future__ import annotations

import math
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import font as tkfont

from .matchview import Layout, MatchState, Metrics, clock, layout, make_labels


@dataclass(frozen=True)
class Theme:
    bg: str
    surface: str
    surface2: str
    border: str
    text: str
    muted: str
    accent: str
    good: str
    bad: str
    on_bar: str  # text on the coloured bars
    palette: tuple[str, ...]  # what the bars are coloured with, by clip


def mix(a: str, b: str, t: float) -> str:
    """The colour `t` of the way from hex colour a to b."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def _stable(name: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(name))  # not hash(): that changes between runs


class MatchCanvas(tk.Canvas):
    """Shows a MatchState as timelines. `set_state()` is all a caller needs; call it from the window's thread."""

    FPS = 30
    EASE = 13.0  # how fast things move to where they belong (bigger = quicker)

    def __init__(self, master, theme: Theme, scale: float = 1.0, height: int = 320, **kw):
        super().__init__(master, height=height, bg=theme.bg, highlightthickness=0, bd=0, **kw)
        self.theme, self.k = theme, scale
        self.m = Metrics().scaled(scale)
        self.f_head = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.f_lane = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self.f_bar = tkfont.Font(family="Segoe UI", size=8)
        self.f_bar_big = tkfont.Font(family="Segoe UI", size=9)
        self.f_bar_huge = tkfont.Font(family="Segoe UI", size=10)
        self.f_small = tkfont.Font(family="Segoe UI", size=8)
        self.state: MatchState | None = None
        self.titles: dict[str, str] = {}
        self.labels: dict[str, str] = {}
        self.lay: Layout | None = None
        self._bars: dict[str, list[float]] = {}  # name -> [x, y, w, h] as drawn now, on their way to the layout's
        self._chips: dict[str, list[float]] = {}  # name -> [x, y, size]
        self._lanes: dict[int, list[float]] = {}  # lane index -> [y, h]
        self._hits: list[tuple[float, float, float, float, str]] = []
        self._hover: tuple[float, float, str] | None = None
        self._job: str | None = None
        self._last = time.monotonic()
        self._names_key: tuple = ()
        self.bind("<Configure>", lambda e: self._relayout())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Destroy>", lambda e: self._stop())
        self._draw()

    # ------------------------------------------------------------------ what a caller uses

    def set_state(self, state: MatchState | None, titles: dict[str, str] | None = None) -> None:
        """Show `state` (None: back to the empty view). Things move to their new places over a fraction of a second."""
        if titles is not None:
            self.titles = titles
        self.state = state
        if state is not None:
            key = (tuple(state.durations), id(self.titles), len(self.titles))
            if key != self._names_key:  # short labels depend on all the titles: only redo them when those change
                self._names_key = key
                self.labels = make_labels(list(state.durations), self.titles)
        else:
            self._bars.clear()
            self._chips.clear()
            self._lanes.clear()
        self._relayout()

    def settle(self) -> None:
        """Put everything where it is going, at once (for a picture, and for tests)."""
        if self.lay:
            for b in self.lay.bars:
                self._bars[b.name] = [b.x, b.y, b.w, b.h]
            for c in self.lay.chips:
                self._chips[c.name] = [c.x, c.y, c.size]
            for ln in self.lay.lanes:
                self._lanes[ln.index] = [ln.y, ln.h]
        self._draw()

    # ------------------------------------------------------------------ layout and motion

    def _relayout(self) -> None:
        w, h = self.winfo_width(), self.winfo_height()
        if self.state is None or w < 60 or h < 60:
            self._draw()
            return
        self.lay = layout(self.state, self.labels, self.titles, w, h, self.m)
        # a bar that appears starts from the chip it was in the tray; a new lane starts where it will be
        for b in self.lay.bars:
            if b.name not in self._bars:
                chip = self._chips.get(b.name)
                self._bars[b.name] = [chip[0], chip[1], chip[2], chip[2]] if chip else [b.x, b.y, 0.0, b.h]
        for c in self.lay.chips:
            self._chips.setdefault(c.name, [c.x, c.y, c.size])
        for ln in self.lay.lanes:
            self._lanes.setdefault(ln.index, [ln.y, ln.h])
        keep_b = {b.name for b in self.lay.bars}
        keep_c = {c.name for c in self.lay.chips}
        self._bars = {n: v for n, v in self._bars.items() if n in keep_b}
        self._chips = {n: v for n, v in self._chips.items() if n in keep_c}
        self._lanes = {i: v for i, v in self._lanes.items() if i in {ln.index for ln in self.lay.lanes}}
        self._start()

    def _start(self) -> None:
        if self._job is None:
            self._last = time.monotonic()
            self._job = self.after(1, self._tick)

    def _stop(self) -> None:
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _tick(self) -> None:
        self._job = None
        now = time.monotonic()
        f = 1 - math.exp(-self.EASE * min(0.1, now - self._last))
        self._last = now
        moving = False
        if self.lay:
            def ease(cur: list[float], goal: list[float]) -> None:
                nonlocal moving
                for i, g in enumerate(goal):
                    d = g - cur[i]
                    if abs(d) > 0.3:
                        cur[i] += d * f
                        moving = True
                    else:
                        cur[i] = g

            for b in self.lay.bars:
                ease(self._bars[b.name], [b.x, b.y, b.w, b.h])
            for c in self.lay.chips:
                ease(self._chips[c.name], [c.x, c.y, c.size])
            for ln in self.lay.lanes:
                ease(self._lanes[ln.index], [ln.y, ln.h])
        self._draw()
        pulsing = bool(self.state and not self.state.finished and (self.state.compare or self.state.current))
        if moving or pulsing:
            self._job = self.after(int(1000 / self.FPS), self._tick)

    # ------------------------------------------------------------------ hover

    def _on_motion(self, e) -> None:
        hit = next((n for x0, y0, x1, y1, n in reversed(self._hits) if x0 <= e.x <= x1 and y0 <= e.y <= y1), None)
        new = (e.x, e.y, hit) if hit else None
        if new != self._hover:
            self._hover = new
            if self._job is None:
                self._draw()

    def _on_leave(self, _e) -> None:
        if self._hover:
            self._hover = None
            if self._job is None:
                self._draw()

    # ------------------------------------------------------------------ drawing

    def _round(self, x0, y0, x1, y1, r, **kw) -> int:
        r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        return self.create_polygon(pts, smooth=True, **kw)

    def _bar_colour(self, name: str) -> str:
        return self.theme.palette[_stable(name) % len(self.theme.palette)]

    def _draw(self) -> None:
        t = self.theme
        self.delete("all")
        self._hits = []
        w, h = self.winfo_width(), self.winfo_height()
        if self.lay is None or self.state is None:
            self.create_text(
                w / 2, h / 2, text="The matching shows up here while the videos are sorted", fill=t.muted, font=self.f_lane,
            )
            return
        lay, m, s = self.lay, self.m, self.state
        pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 6.0)

        self.create_text(m.pad, m.head_h / 2 + 1, text=lay.headline, anchor="w", fill=t.text, font=self.f_head)
        extra = []
        if lay.hidden_lanes:
            extra.append(f"+{lay.hidden_lanes} more group{'s' if lay.hidden_lanes != 1 else ''}")
        if extra:
            self.create_text(w - m.pad, m.head_h / 2 + 1, text="  ".join(extra), anchor="e", fill=t.muted, font=self.f_small)

        # time marks behind the lanes
        for x, text in lay.ticks:
            self.create_line(x, m.head_h, x, lay.axis_y, fill=t.border, dash=(1, 5))
            self.create_text(x + 3, lay.axis_y + m.axis_h / 2, text=text, anchor="w", fill=t.muted, font=self.f_small)

        # lanes
        for ln in lay.lanes:
            y, hh = self._lanes.get(ln.index, [ln.y, ln.h])
            fill = mix(t.surface, t.bg, 0.55) if ln.dim else t.surface
            edge = t.accent if ln.chosen else (mix(t.border, t.bg, 0.5) if ln.dim else t.border)
            self._round(m.pad - 6, y, w - m.pad + 6, y + hh, 8 * self.k, fill=fill, outline=edge, width=2 if ln.chosen else 1)
            self.create_text(
                m.pad + 2, y + m.lane_head_h / 2 + 1, text=ln.label, anchor="w",
                fill=t.muted if ln.dim else (t.accent if ln.chosen else t.text), font=self.f_lane,
            )

        # bars
        link = lay.link
        ends = {link.src, link.dst} if link else set()
        dim_lanes = {ln.index for ln in lay.lanes if ln.dim}
        for b in lay.bars:
            x, y, bw, bh = self._bars.get(b.name, [b.x, b.y, b.w, b.h])
            colour = self._bar_colour(b.name)
            if b.lane in dim_lanes:
                colour = mix(colour, t.bg, 0.68)
            if b.name in ends and link:  # a soft halo round the clips being compared
                halo = t.accent if link.z is None else (t.good if link.ok else t.bad)
                g = (2 + 2 * pulse) * self.k
                self._round(x - g, y - g, x + bw + g, y + bh + g, 5 * self.k, fill=mix(t.surface, halo, 0.35 + 0.25 * pulse), outline="")
            self._round(x, y, x + bw, y + bh, min(4 * self.k, bh / 2), fill=colour, outline=t.accent if b.chosen else "")
            if b.label and bh >= 11 * self.k and bw >= 0.7 * b.w:
                font = self.f_bar_huge if bh >= 30 * self.k else self.f_bar_big if bh >= 21 * self.k else self.f_bar
                dim = b.lane in dim_lanes
                self.create_text(
                    x + 6 * self.k, y + bh / 2, text=b.label, anchor="w", font=font,
                    fill=mix(t.muted, t.bg, 0.35) if dim else t.on_bar,
                )
            self._hits.append((x, y, x + bw, y + bh, b.name))

        # the tray
        self.create_line(m.pad, lay.tray_y, w - m.pad, lay.tray_y, fill=t.border)
        row1, row2 = lay.tray_y + 6, lay.tray_y + 6 + m.chip + 16
        self.create_text(m.pad, row1 + m.chip / 2, text=f"Waiting {lay.waiting}" if lay.waiting else "Waiting", anchor="w", fill=t.muted, font=self.f_small)
        self.create_text(m.pad, row2 + m.chip / 2, text=f"No match {lay.unmatched}" if lay.unmatched else "No match", anchor="w", fill=t.muted, font=self.f_small)
        for c in lay.chips:
            x, y, size = self._chips.get(c.name, [c.x, c.y, c.size])
            if c.kind == "current":
                g = (2 + 3 * pulse) * self.k
                self._round(x - g, y - g, x + size + g, y + size + g, 4 * self.k, fill=mix(t.surface, t.accent, 0.3 + 0.3 * pulse), outline="")
                self._round(x, y, x + size, y + size, 3 * self.k, fill=t.accent, outline="")
            elif c.kind == "unmatched":
                self._round(x, y, x + size, y + size, 2 * self.k, fill=mix(t.bad, t.bg, 0.55), outline=mix(t.bad, t.bg, 0.25))
            else:
                self._round(x, y, x + size, y + size, 2 * self.k, fill=mix(self._bar_colour(c.name), t.bg, 0.6), outline="")
            self._hits.append((x, y, x + size, y + size, c.name))
        for kind, row_y in (("waiting", row1), ("unmatched", row2)):
            n = lay.hidden_chips.get(kind)
            if n:
                shown = [c for c in lay.chips if c.kind == kind]
                x0 = (max(c.x for c in shown) + m.chip + m.chip_gap * 2) if shown else m.pad + 80 * self.k
                self.create_text(x0, row_y + m.chip / 2, text=f"+{n}", anchor="w", fill=t.muted, font=self.f_small)

        # the comparison going on
        if link:
            a, b = self._where(link.src), self._where(link.dst)
            if a and b:
                colour = t.accent if link.z is None else (t.good if link.ok else t.bad)
                (ax, ay), (bx, by) = a, b
                mx, my = (ax + bx) / 2, min(ay, by) - 24 * self.k if ay > by else (ay + by) / 2 - 16 * self.k
                self.create_line(
                    ax, ay, mx, my, bx, by, smooth=True, width=2 * self.k, fill=colour, arrow="last",
                    dash=(5, 4) if link.z is None else (), dashoffset=int(time.monotonic() * 20) % 9 if link.z is None else 0,
                )
                text = "comparing..." if link.z is None else f"z {link.z:.0f}" + ("  match" if link.ok else "  not confirmed" if link.doubted else "")
                tid = self.create_text(mx, my - 8 * self.k, text=text, fill=colour, font=self.f_lane)
                x0, y0, x1, y1 = self.bbox(tid)
                self._round(x0 - 4, y0 - 1, x1 + 4, y1 + 1, 4 * self.k, fill=t.bg, outline="")
                self.tag_raise(tid)

        if self._hover and self._hover[2]:
            self._tooltip(*self._hover)

    def _where(self, name: str) -> tuple[float, float] | None:
        """The middle of a clip's bar, or of its chip in the tray."""
        if name in self._bars:
            x, y, w, h = self._bars[name]
            return x + w / 2, y + h / 2
        if name in self._chips:
            x, y, size = self._chips[name]
            return x + size / 2, y + size / 2
        return None

    def _tooltip(self, mx: float, my: float, name: str) -> None:
        t = self.theme
        bar = next((b for b in (self.lay.bars if self.lay else []) if b.name == name), None)
        title = bar.title if bar else next((c.title for c in (self.lay.chips if self.lay else []) if c.name == name), name)
        title = title if len(title) <= 64 else title[:63] + "…"
        length = self.state.durations.get(name, 0.0) if self.state else 0.0
        detail = f"{clock(length)}" + (f"  ·  starts {clock(bar.start)} into its group" if bar else "")
        a = self.create_text(mx + 14, my + 16, text=title, anchor="nw", fill=t.text, font=self.f_lane)
        b = self.create_text(mx + 14, my + 16 + self.f_lane.metrics("linespace") + 2, text=detail, anchor="nw", fill=t.muted, font=self.f_small)
        x0, y0, x1, y1 = self.bbox(a, b)
        dx = min(0, self.winfo_width() - 8 - (x1 + 8))  # keep it inside the view
        dy = min(0, self.winfo_height() - 4 - (y1 + 6))
        if dx or dy:
            self.move(a, dx, dy)
            self.move(b, dx, dy)
            x0, y0, x1, y1 = self.bbox(a, b)
        box = self._round(x0 - 8, y0 - 5, x1 + 8, y1 + 5, 6 * self.k, fill=t.surface2, outline=t.border)
        self.tag_lower(box, a)
