"""The drawn view of the matching, on a real (invisible) Tk canvas."""
import gc
import time
import tkinter as tk

import pytest

from meld.matchcanvas import MatchCanvas, MatchPanel, Theme, mix
from meld.matchview import MatchState

THEME = Theme(
    bg="#12131c", surface="#1b1d2b", surface2="#252838", border="#333852", text="#eef0f8", muted="#9399b8",
    accent="#ffd23f", good="#5ee0a0", bad="#ff785a", on_bar="#15161f", palette=("#ffd23f", "#ff9f45", "#6ec6ff"),
)
TITLES = {
    "a.m4a": "Oasis - Wonderwall (Live at Wembley 2025)",
    "b.m4a": "Oasis - Supersonic (Live at Wembley 2025)",
    "c.m4a": "Oasis - Live Forever (Live at Wembley 2025)",
    "d.m4a": "Oasis - Slide Away (Live at Wembley 2025)",
}


@pytest.fixture(scope="module")
def root():
    """One Tk window for all these tests: making and destroying several in a process breaks Tcl's own state now and then."""
    try:
        r = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display to draw on: {e}")
    r.attributes("-alpha", 0)  # a real window, but nobody sees it
    yield r
    r.destroy()


@pytest.fixture
def canvas(root):
    root.geometry("1000x520+0+0")
    c = MatchCanvas(root, THEME, 1.0, height=520)
    c.pack(fill="both", expand=True)
    for _ in range(5):
        root.update()
    yield c
    try:
        c.destroy()
    except tk.TclError:  # a test that destroyed it itself
        pass
    root.update()
    gc.collect()  # what Tk objects a test left behind are freed here, on this thread: not by the collector on some worker's


def texts(c):
    return [c.itemcget(i, "text") for i in c.find_all() if c.type(i) == "text"]


def pump(c, seconds=1.5):
    end = time.time() + seconds
    while time.time() < end and c._job is not None:
        c.update()
        time.sleep(0.01)


def state(**kw):
    base = dict(durations={n: 120.0 for n in TITLES}, groups=(), pending=(), phase=1)
    base.update(kw)
    return MatchState(**base)


def test_before_anything_happens_it_says_what_it_is_for(canvas):
    assert any("shows up here" in t for t in texts(canvas))
    canvas.set_state(state(pending=("a.m4a", "b.m4a")), TITLES)
    canvas.settle()
    assert not any("shows up here" in t for t in texts(canvas))


def test_a_state_is_drawn_with_the_headline_the_lane_and_the_titles(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60, "c.m4a": 130},), pending=("d.m4a",)), TITLES)
    canvas.settle()
    shown = texts(canvas)
    assert any(t.startswith("Sorting 3 of 4") for t in shown)
    assert any(t.startswith("Group 1") and "3 clips" in t for t in shown)
    assert {"Wonderwall", "Supersonic", "Live Forever"} <= set(shown)  # the songs, not "Oasis ... Live at Wembley 2025"
    assert "Waiting 1" in shown


def test_going_back_to_nothing_clears_it(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},)), TITLES)
    canvas.settle()
    canvas.set_state(None)
    assert any("shows up here" in t for t in texts(canvas)) and canvas._bars == {} and canvas._chips == {}


def test_a_clip_that_gets_placed_starts_from_where_it_waited_in_the_tray(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},), pending=("c.m4a", "d.m4a"), current=None), TITLES)
    canvas.settle()
    chip = list(canvas._chips["c.m4a"])
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60, "c.m4a": 130},), pending=("d.m4a",)), TITLES)
    x, y, w, h = canvas._bars["c.m4a"]
    assert (x, y) == (chip[0], chip[1]) and w == h == chip[2]  # it begins as the little square it was
    target = next(b for b in canvas.lay.bars if b.name == "c.m4a")
    assert (x, y) != (target.x, target.y)  # and has some way to go


def test_things_slide_into_place_and_then_it_stops_moving(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},), pending=("c.m4a",)), TITLES)
    canvas.settle()
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60, "c.m4a": 130},), finished=True, chosen=frozenset("abc")), TITLES)
    assert canvas._job is not None  # moving
    pump(canvas)
    assert canvas._job is None  # settled, and the timer stopped: a finished view costs nothing
    for b in canvas.lay.bars:
        assert canvas._bars[b.name] == pytest.approx([b.x, b.y, b.w, b.h], abs=0.5)


def test_it_keeps_animating_while_a_comparison_is_going_on(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},), pending=("d.m4a",), current="c.m4a", compare=("c.m4a", "a.m4a", None)), TITLES)
    canvas.settle()
    pump(canvas, 0.3)
    assert canvas._job is not None  # the halo pulses until the comparison is over
    assert any("comparing" in t for t in texts(canvas))
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},), current="c.m4a", compare=("c.m4a", "a.m4a", 42.0)), TITLES)
    canvas.settle()
    assert any(t.startswith("z 42") and "match" in t for t in texts(canvas))
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},), current="c.m4a", compare=("c.m4a", "a.m4a", 3.0)), TITLES)
    canvas.settle()
    assert any(t == "z 3" for t in texts(canvas))  # a score too low to match says only its number


def test_a_view_that_is_over_marks_the_chosen_group(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60, "c.m4a": 130}, {"d.m4a": 0}), finished=True, chosen=frozenset("abc")), TITLES)
    canvas.settle()
    assert any("highlighted" in t for t in texts(canvas)) is False  # one group of two or more: nothing to choose between
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60}, {"c.m4a": 0, "d.m4a": 70}), finished=True, chosen=frozenset({"c.m4a", "d.m4a"})), TITLES)
    canvas.settle()
    assert any("highlighted one is used" in t for t in texts(canvas))


def test_resizing_lays_it_out_again(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},)), TITLES)
    canvas.settle()
    wide = max(b.x + b.w for b in canvas.lay.bars)
    canvas.master.geometry("600x520+0+0")
    for _ in range(8):
        canvas.master.update()
    canvas.settle()
    assert canvas.lay.width < 700 and max(b.x + b.w for b in canvas.lay.bars) < wide


def test_hovering_a_bar_shows_its_whole_title(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},)), TITLES)
    canvas.settle()
    pump(canvas)  # let the animation finish: hovering redraws at once only when nothing is moving
    x0, y0, x1, y1, name, _ = next(h for h in canvas._hits if h[4] == "a.m4a")

    class Event:
        x, y = (x0 + x1) / 2, (y0 + y1) / 2

    canvas._on_motion(Event)
    shown = texts(canvas)
    assert any("Oasis - Wonderwall" in t for t in shown) and any("into its group" in t for t in shown)
    canvas._on_leave(None)
    assert not any("into its group" in t for t in texts(canvas))


# ---- scrolling

def crowd(groups=12, **kw):
    """Many groups: more than the view has room for."""
    names = {f"g{k}{c}.m4a": 120.0 for k in range(groups) for c in "abc"}
    return MatchState(
        durations=names, groups=tuple({f"g{k}a.m4a": 0, f"g{k}b.m4a": 40, f"g{k}c.m4a": 80} for k in range(groups)), **kw,
    )


def long_show(**kw):
    """One group of two hours: too long for the view at a scale the bars can be read at."""
    return MatchState(durations={"a.m4a": 1800.0, "b.m4a": 1800.0}, groups=({"a.m4a": 0, "b.m4a": 5400},), **kw)


class Wheel:
    def __init__(self, delta, x=100, y=100):
        self.delta, self.x, self.y = delta, x, y


def headline_y(c):
    item = next(i for i in c.find_all() if c.type(i) == "text" and c.itemcget(i, "text").startswith("Sorting"))
    return c.coords(item)[1]


def scroll_y(c, notches):
    for _ in range(abs(notches)):
        c._wheel(Wheel(-120 if notches > 0 else 120), "y")
    c.update()


def test_what_does_not_fit_the_view_is_scrolled_to_and_the_scrollregion_is_the_content(canvas):
    canvas.set_state(crowd(), TITLES)
    canvas.settle()
    lay = canvas.lay
    assert len(lay.lanes) == 12 and lay.content_h > canvas.winfo_height()
    x0, y0, x1, y1 = (float(v) for v in canvas.cget("scrollregion").split())
    assert (x0, y0) == (0, 0) and (x1, y1) == pytest.approx((lay.content_w, lay.content_h))
    first, last = canvas.yview()
    assert first == 0 and 0 < last < 1  # some of it is below the view
    assert canvas.xview() == (0.0, 1.0)  # and none of it to the side


def test_the_wheel_scrolls_up_and_down_and_the_headline_and_tray_stay_in_the_view(canvas):
    canvas.set_state(crowd(), TITLES)
    canvas.settle()
    top = headline_y(canvas)
    lane = next(b for b in canvas.lay.bars if b.lane == 11)
    assert canvas.canvasy(canvas.winfo_height()) < lane.y  # the last lane is below the view
    scroll_y(canvas, 2)
    assert canvas.canvasy(0) > 0 and canvas.yview()[0] > 0
    assert headline_y(canvas) == pytest.approx(top + canvas.canvasy(0))  # it went with the view: it is still on top
    scroll_y(canvas, -2)
    assert canvas.canvasy(0) == 0
    scroll_y(canvas, 60)  # all the way down: the last lane comes into view above the tray
    last = max(canvas.lay.lanes, key=lambda ln: ln.y)
    assert canvas.canvasy(canvas.winfo_height()) >= last.y + last.h and canvas.yview()[1] == 1.0


def test_the_wheel_with_shift_scrolls_sideways_in_a_group_too_long_for_the_view(canvas):
    canvas.set_state(long_show(), TITLES)
    canvas.settle()
    assert canvas.lay.content_w > canvas.winfo_width() and canvas.xview()[0] == 0
    canvas._wheel(Wheel(-120), "x")
    canvas.update()
    ox = canvas.canvasx(0)
    assert ox > 0
    lane_label = next(i for i in canvas.find_all() if canvas.type(i) == "text" and canvas.itemcget(i, "text").startswith("Group 1"))
    assert canvas.coords(lane_label)[0] == pytest.approx(ox + canvas.m.pad + 2)  # the name of the lane stays in view too


def test_a_bar_keeps_its_label_in_view_while_its_bar_is_partly_scrolled_past(canvas):
    canvas.set_state(long_show(), TITLES)
    canvas.settle()
    bar = next(b for b in canvas.lay.bars if b.name == "a.m4a")
    assert bar.label  # 1800 s at the least scale is wide enough to write on
    canvas.xview_moveto((bar.x + 100) / canvas.lay.content_w)  # a little way into the bar
    canvas.update()
    ox = canvas.canvasx(0)
    assert bar.x < ox < bar.x + bar.w - 200
    text = next(i for i in canvas.find_all() if canvas.type(i) == "text" and canvas.itemcget(i, "text") == bar.label)
    assert canvas.coords(text)[0] >= ox  # the label is where it can be read


def test_zooming_keeps_the_time_under_the_mouse_where_it_is(canvas):
    canvas.set_state(long_show(), TITLES)
    canvas.settle()
    before = canvas.lay.scale
    at = 300
    seconds = (canvas.canvasx(at) - canvas.m.pad) / before
    canvas._zoom_wheel(Wheel(+120, x=at))
    canvas.update()
    assert canvas.lay.scale == pytest.approx(before * canvas.ZOOM_STEP) and canvas.zoom == pytest.approx(canvas.ZOOM_STEP)
    assert (canvas.canvasx(at) - canvas.m.pad) / canvas.lay.scale == pytest.approx(seconds, abs=0.5)
    for _ in range(60):
        canvas._zoom_wheel(Wheel(-120, x=at))  # zooming out stops at the whole group in view
    canvas.update()
    assert canvas.lay.content_w == pytest.approx(canvas.winfo_width()) and canvas.zoom < 1
    fit = canvas.lay.scale
    canvas._zoom_wheel(Wheel(-120, x=at))
    assert canvas.lay.scale == fit  # no smaller
    canvas._zoom_wheel(Wheel(+120, x=at))
    assert canvas.lay.scale > fit  # and one notch back in works at once, however far out one tried to go


def test_hovering_finds_the_bar_under_the_mouse_when_the_view_is_scrolled(canvas):
    canvas.set_state(crowd(), TITLES)
    canvas.settle()
    scroll_y(canvas, 2)
    pump(canvas)
    bar = next(b for b in canvas.lay.bars if canvas.canvasy(0) + 60 < b.y < canvas.canvasy(canvas.winfo_height()) - 120)
    ex, ey = bar.x + bar.w / 2 - canvas.canvasx(0), bar.y + bar.h / 2 - canvas.canvasy(0)  # where it is in the view
    assert canvas._hit_at(ex, ey) == bar.name
    canvas._on_motion(Wheel(0, ex, ey))
    assert any("into its group" in t for t in texts(canvas))


def test_a_bar_under_the_headline_cannot_be_hovered(canvas):
    canvas.set_state(crowd(), TITLES)
    canvas.settle()
    scroll_y(canvas, 1)
    pump(canvas)
    bar = next(b for b in canvas.lay.bars if 0 < b.y + b.h / 2 - canvas.canvasy(0) < canvas.m.head_h)  # scrolled up under it
    assert canvas._hit_at(bar.x + 5, bar.y + bar.h / 2 - canvas.canvasy(0)) is None


def test_a_clip_placed_while_the_view_is_scrolled_still_starts_from_its_chip(canvas):
    names = {f"g{k}{c}.m4a": 120.0 for k in range(12) for c in "abc"} | {"w.m4a": 120.0}
    groups = tuple({f"g{k}a.m4a": 0, f"g{k}b.m4a": 40, f"g{k}c.m4a": 80} for k in range(12))
    canvas.set_state(MatchState(durations=names, groups=groups, pending=("w.m4a",)), TITLES)
    canvas.settle()
    scroll_y(canvas, 3)
    chip = list(canvas._chips["w.m4a"])
    canvas.set_state(MatchState(durations=names, groups=(*groups[:-1], {**groups[-1], "w.m4a": 100}), pending=()), TITLES)
    x, y, w, h = canvas._bars["w.m4a"]
    assert (x, y) == pytest.approx((chip[0], chip[1] + canvas.canvasy(0)))  # in the content: where the chip is in the view


def test_going_back_to_nothing_goes_back_to_the_top_left_at_the_usual_zoom(canvas):
    canvas.set_state(crowd(), TITLES)
    canvas.settle()
    scroll_y(canvas, 1)
    canvas._zoom_wheel(Wheel(+120))
    canvas.set_state(None)
    canvas.update()
    assert canvas.zoom == 1.0 and canvas.canvasy(0) == 0 and canvas.canvasx(0) == 0
    assert any("shows up here" in t for t in texts(canvas))


def test_the_panel_has_scrollbars_that_follow_the_view(root):
    root.geometry("1000x520+0+0")
    panel = MatchPanel(root, THEME, 1.0, height=520)
    panel.pack(fill="both", expand=True)
    for _ in range(5):
        root.update()
    try:
        panel.set_state(crowd(), TITLES)
        panel.settle()
        root.update()
        assert panel.canvas.winfo_width() > 900  # the bars take a little of the room, not most of it
        first, last = (float(v) for v in panel.ybar.get())
        assert first == 0 and 0 < last < 1 and tuple(float(v) for v in panel.xbar.get()) == (0.0, 1.0)
        scroll_y(panel.canvas, 1)
        assert float(panel.ybar.get()[0]) > 0  # the bar moved with the wheel
        panel.canvas.yview_moveto(0.5)  # and the view with the bar
        root.update()
        assert panel.canvas.yview()[0] == pytest.approx(0.5, abs=0.05)
    finally:
        panel.destroy()
        root.update()
        gc.collect()


def test_destroying_it_in_the_middle_of_an_animation_is_harmless(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},), pending=("c.m4a",), current="c.m4a"), TITLES)
    assert canvas._job is not None
    canvas.destroy()
    canvas.master.update()  # a callback that fires after the canvas is gone must not raise


def test_colours_mix_between_two_hex_colours():
    assert mix("#000000", "#ffffff", 0.5) == "#808080"
    assert mix("#ff0000", "#00ff00", 0) == "#ff0000" and mix("#ff0000", "#00ff00", 1) == "#00ff00"
