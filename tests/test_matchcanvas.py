"""The drawn view of the matching, on a real (invisible) Tk canvas."""
import time
import tkinter as tk

import pytest

from meld.matchcanvas import MatchCanvas, Theme, mix
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
    x0, y0, x1, y1, name = next(h for h in canvas._hits if h[4] == "a.m4a")

    class Event:
        x, y = (x0 + x1) / 2, (y0 + y1) / 2

    canvas._on_motion(Event)
    shown = texts(canvas)
    assert any("Oasis - Wonderwall" in t for t in shown) and any("into its group" in t for t in shown)
    canvas._on_leave(None)
    assert not any("into its group" in t for t in texts(canvas))


def test_destroying_it_in_the_middle_of_an_animation_is_harmless(canvas):
    canvas.set_state(state(groups=({"a.m4a": 0, "b.m4a": 60},), pending=("c.m4a",), current="c.m4a"), TITLES)
    assert canvas._job is not None
    canvas.destroy()
    canvas.master.update()  # a callback that fires after the canvas is gone must not raise


def test_colours_mix_between_two_hex_colours():
    assert mix("#000000", "#ffffff", 0.5) == "#808080"
    assert mix("#ff0000", "#00ff00", 0) == "#ff0000" and mix("#ff0000", "#00ff00", 1) == "#00ff00"
