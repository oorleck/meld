"""The live view of the matching: the states it emits, and how they are laid out as timelines."""
import random

import pytest
from test_groups import A, B, SR, TWO_CONCERTS, project_with, sync

from meld.matchview import MatchState, Metrics, clock, fit_check, layout, make_labels, rank_groups, tick_text
from meld.sync import sync_project


def watch(project, **kw):
    states = []
    tl = sync_project(project, log=lambda *_: None, on_state=states.append, **kw)
    return tl, states


# ---- the states the matching emits


def test_the_first_state_has_every_clip_waiting_and_nothing_sorted(tmp_path):
    _, states = watch(project_with(tmp_path, TWO_CONCERTS), workers=1)
    first = states[0]
    assert first.groups == () and first.current is None and first.phase == 1 and first.compared == 0
    assert sorted(first.pending) == sorted(first.durations) == sorted(f"{k}.wav" for k in TWO_CONCERTS)
    assert list(first.pending) == sorted(first.pending, key=lambda n: -first.durations[n])  # longest first, as sorted


def test_every_state_is_consistent(tmp_path):
    _, states = watch(project_with(tmp_path, TWO_CONCERTS), workers=1)
    last_pending, last_compared = len(states[0].pending), 0
    for st in states:
        in_groups = [n for g in st.groups for n in g]
        assert len(in_groups) == len(set(in_groups))  # a clip is in one group at most
        assert set(in_groups) | set(st.pending) | ({st.current} if st.current else set()) <= set(st.durations)
        assert not set(in_groups) & set(st.pending)  # sorted clips are not waiting
        assert st.compared >= last_compared
        last_compared = st.compared
        if st.phase == 1:
            assert len(st.pending) <= last_pending  # the queue only shrinks
            last_pending = len(st.pending)


def test_a_comparison_is_announced_and_then_reported_with_its_score(tmp_path):
    _, states = watch(project_with(tmp_path, TWO_CONCERTS), workers=1)
    started = [i for i, st in enumerate(states) if st.compare and st.compare[2] is None]
    assert started
    for i in started:
        u, p, _ = states[i].compare
        after = states[i + 1]
        assert after.compare[:2] == (u, p) and isinstance(after.compare[2], float)  # the same pair, now with a score
        assert after.compared == states[i].compared + 1
    scores = [st.compare[2] for st in states if st.compare and st.compare[2] is not None]
    assert max(scores) >= 25 and min(scores) < 10  # matches and non-matches both occur in two concerts


def test_the_last_state_is_the_result_with_the_used_group_marked(tmp_path):
    tl, states = watch(project_with(tmp_path, TWO_CONCERTS), workers=1)
    last = states[-1]
    assert last.finished and last.current is None and last.pending == () and last.compare is None
    assert last.chosen == {c.file for c in tl.clips}
    by_size = sorted(last.groups, key=len, reverse=True)
    assert [len(g) for g in by_size] == [4, 3]  # the two concerts
    assert sum(1 for st in states if st.finished) == 1  # announced once


def test_a_bridge_is_seen_as_two_groups_becoming_one(tmp_path):
    spec = {"x": (A, 0, 70), "y": (A, 100, 170), "z": (A, 60, 110)}
    _, states = watch(project_with(tmp_path, spec), workers=1)
    counts = [len([g for g in st.groups if g]) for st in states]
    assert 2 in counts and counts[-1] == 1 and counts.index(2) < len(counts) - 1  # x and y apart, then joined by z
    joined = states[-1].groups[0]
    assert set(joined) == {"x.wav", "y.wav", "z.wav"}


def test_the_second_look_is_shown_as_such(tmp_path):
    spec = {f"a{i}": (A, i * 30, i * 30 + 60) for i in range(5)}
    spec |= {"b1": (B, 0, 20), "b2": (B, 12, 32)}
    _, states = watch(project_with(tmp_path, spec), workers=1, max_compare=3)
    assert any(st.phase == 2 and st.current for st in states)
    assert all(st.pending == () for st in states if st.phase == 2)


def test_the_states_do_not_depend_on_running_comparisons_in_parallel(tmp_path):
    _, serial = watch(project_with(tmp_path / "s", TWO_CONCERTS), workers=1)
    _, parallel = watch(project_with(tmp_path / "p", TWO_CONCERTS), workers=3)
    assert serial[-1].groups == parallel[-1].groups and serial[-1].chosen == parallel[-1].chosen
    assert [st.compared for st in serial][-1] == [st.compared for st in parallel][-1]


def test_watching_changes_nothing_about_the_result(tmp_path):
    watched, _ = watch(project_with(tmp_path / "w", TWO_CONCERTS), workers=1)
    plain = sync(project_with(tmp_path / "p", TWO_CONCERTS), workers=1)
    assert [(c.file, c.offset, c.z) for c in watched.clips] == [(c.file, c.offset, c.z) for c in plain.clips]


# ---- labels


TITLES = {
    "a.m4a": "Oasis - Wonderwall (Live at Wembley 2025)",
    "b.m4a": "Oasis - Supersonic (Live at Wembley 2025)",
    "c.m4a": "Oasis - Live Forever (Live at Wembley 2025)",
    "d.m4a": "Oasis - Champagne Supernova (Live at Wembley 2025)",
}


def test_labels_drop_the_words_nearly_every_title_has():
    labels = make_labels(list(TITLES), TITLES)
    assert labels["a.m4a"] == "Wonderwall" and labels["d.m4a"] == "Champagne Supernova"  # the song, not the band/year


def test_labels_fall_back_to_the_whole_title_then_the_file():
    same = {n: "Same Title" for n in ("a.m4a", "b.m4a", "c.m4a")}
    assert make_labels(list(same), same)["a.m4a"] == "Same Title"  # nothing tells them apart: keep it whole
    assert make_labels(["x.m4a", "y.m4a"], {"x.m4a": "One", "y.m4a": "Two"}) == {"x.m4a": "One", "y.m4a": "Two"}
    assert make_labels(["abc123.m4a"], {}) == {"abc123.m4a": "abc123"}  # no title known at all


def test_titles_can_be_given_by_video_id_as_well_as_by_file_name():
    labels = make_labels(["abc.m4a", "def.m4a"], {"abc": "First", "def": "Second"})
    assert labels == {"abc.m4a": "First", "def.m4a": "Second"}


# ---- layout


def state(groups, pending=(), durations=None, **kw):
    names = [n for g in groups for n in g] + list(pending) + ([kw["current"]] if kw.get("current") else [])
    durations = durations or {n: 120.0 for n in dict.fromkeys(names)}
    return MatchState(durations=durations, groups=tuple(groups), pending=tuple(pending), **kw)


def lay(st, titles=None, width=900, height=360, **kw):
    titles = titles or {}
    return layout(st, make_labels(list(st.durations), titles), titles, width, height, **kw)


def test_each_group_of_two_or_more_is_a_lane_and_the_biggest_comes_first():
    st = state([{"s.m4a": 0}, {"a.m4a": 0, "b.m4a": 100}, {"c.m4a": 0, "d.m4a": 50, "e.m4a": 90}])
    out = lay(st)
    assert [ln.count for ln in out.lanes] == [3, 2]
    assert [b.name for b in out.bars if b.lane == 0] == ["c.m4a", "d.m4a", "e.m4a"]
    assert [c.name for c in out.chips if c.kind == "unmatched"] == ["s.m4a"]  # a lone clip has no timeline yet
    assert out.lanes[0].label.startswith("Group 1") and "3 clips" in out.lanes[0].label


def test_clips_that_overlap_are_in_different_rows_and_those_that_do_not_share_one():
    st = state([{"a.m4a": 0, "b.m4a": 60, "c.m4a": 300}])  # a and b overlap; c starts long after both end
    bars = {b.name: b for b in lay(st).bars}
    assert bars["a.m4a"].y != bars["b.m4a"].y  # overlapping: stacked
    assert bars["a.m4a"].y == bars["c.m4a"].y  # apart in time: the same row


def test_all_lanes_share_one_time_scale():
    long = {"a.m4a": 0.0, "b.m4a": 500.0}
    short = {"c.m4a": 0.0, "d.m4a": 50.0}
    out = lay(state([long, short], durations={"a.m4a": 400, "b.m4a": 400, "c.m4a": 100, "d.m4a": 100}))
    bars = {b.name: b for b in out.bars}
    assert bars["a.m4a"].w == pytest.approx(400 * out.scale) and bars["c.m4a"].w == pytest.approx(100 * out.scale)
    assert bars["a.m4a"].w == pytest.approx(4 * bars["c.m4a"].w)  # a shorter group looks shorter
    assert bars["d.m4a"].x - bars["c.m4a"].x == pytest.approx(50 * out.scale)  # and offsets keep their proportion


def test_a_group_whose_times_start_below_zero_is_shifted_to_its_left_edge():
    out = lay(state([{"a.m4a": -80.0, "b.m4a": 20.0}]))
    bars = {b.name: b for b in out.bars}
    assert bars["a.m4a"].x == pytest.approx(Metrics().pad) and bars["a.m4a"].start == 0 and bars["b.m4a"].start == 100


def test_the_clip_being_sorted_and_those_waiting_are_in_the_tray():
    st = state([{"a.m4a": 0, "b.m4a": 30}], pending=["w1.m4a", "w2.m4a", "w3.m4a"], current="cur.m4a")
    out = lay(st)
    kinds = {c.name: c.kind for c in out.chips}
    assert kinds["cur.m4a"] == "current" and kinds["w1.m4a"] == kinds["w3.m4a"] == "waiting"
    assert out.waiting == 3 and out.headline.startswith("Sorting 2 of 6")
    assert next(c for c in out.chips if c.kind == "current").size > Metrics().chip  # it stands out


def test_a_lone_clip_getting_its_second_look_is_marked_as_the_one_at_work():
    st = state([{"a.m4a": 0, "b.m4a": 30}, {"lone.m4a": 0}], phase=2, current="lone.m4a")
    out = lay(st)
    assert next(c for c in out.chips if c.name == "lone.m4a").kind == "current"
    assert out.headline.startswith("Second look")


def test_the_comparison_is_a_link_that_says_whether_it_is_enough():
    base = state([{"a.m4a": 0, "b.m4a": 30}], current="c.m4a")
    for z, ok in [(None, False), (4.0, False), (10.0, True), (60.0, True)]:
        st = MatchState(**{**base.__dict__, "compare": ("c.m4a", "a.m4a", z)})
        link = lay(st).link
        assert (link.src, link.dst, link.z, link.ok) == ("c.m4a", "a.m4a", z, ok)
    assert lay(base).link is None


def test_when_it_is_over_the_used_group_is_marked_and_the_others_dimmed():
    st = state([{"a.m4a": 0, "b.m4a": 30, "c.m4a": 60}, {"d.m4a": 0, "e.m4a": 30}], finished=True, chosen=frozenset({"d.m4a", "e.m4a"}))
    out = lay(st)
    assert [(ln.chosen, ln.dim) for ln in out.lanes] == [(False, True), (True, False)]
    assert {b.name for b in out.bars if b.chosen} == {"d.m4a", "e.m4a"}
    assert out.headline.startswith("Sorted: 2 groups") and "highlighted" in out.headline


def test_an_empty_state_says_it_is_waiting():
    out = lay(MatchState(durations={}))
    assert out.lanes == [] and out.bars == [] and out.headline == "Waiting for the clips"


def test_bars_are_labelled_when_they_are_wide_enough_and_shortened_with_an_ellipsis():
    titles = {"a.m4a": "Alpha Song Number One With A Very Long Title", "b.m4a": "B", "c.m4a": "C"}
    out = lay(state([{"a.m4a": 0.0, "b.m4a": 4000.0}], durations={"a.m4a": 300.0, "b.m4a": 5.0}), titles)
    bars = {b.name: b for b in out.bars}
    assert bars["a.m4a"].label.startswith("Alpha Song") and bars["a.m4a"].label.endswith("…")  # cut to fit
    assert bars["b.m4a"].label == "" and bars["b.m4a"].title == "B"  # too narrow to write on; hover shows it


def test_time_marks_are_far_enough_apart_to_read():
    out = lay(state([{"a.m4a": 0.0, "b.m4a": 3000.0}], durations={"a.m4a": 600.0, "b.m4a": 600.0}))
    xs = [x for x, _ in out.ticks]
    assert xs and all(b - a >= 63 for a, b in zip(xs, xs[1:])) and out.ticks[0][1] == "0:00"
    assert len({text for _, text in out.ticks}) == len(out.ticks)  # each mark reads differently


def test_a_long_queue_is_all_there_in_a_tray_that_goes_on_sideways():
    pending = [f"w{i}.m4a" for i in range(300)]
    out = lay(state([{"a.m4a": 0, "b.m4a": 30}], pending=pending))
    shown = [c for c in out.chips if c.kind == "waiting"]
    assert len(shown) == 300 and out.waiting == 300  # none left out
    assert out.content_w > out.width and all(c.x + c.size <= out.content_w for c in shown)  # the content is as wide as they need


def test_more_groups_than_fit_the_view_are_all_there_and_the_content_grows_taller_to_scroll():
    groups = [{f"g{k}a.m4a": 0, f"g{k}b.m4a": 40, f"g{k}c.m4a": 80} for k in range(12)]
    out = lay(state(groups), height=300)
    assert len(out.lanes) == 12  # none hidden
    assert out.content_h > out.height and out.lanes[-1].y + out.lanes[-1].h == pytest.approx(out.lanes_bottom)
    assert out.lanes_bottom <= out.content_h - (out.height - out.axis_y)  # the last one can be scrolled clear of the tray
    assert fit_check(out) == []
    assert out.tray_y == out.height - Metrics().tray_h  # the tray and the headline stay in the view: not in the content


def test_a_group_with_many_overlapping_clips_keeps_readable_bars_and_scrolls():
    crowded = {f"c{i}.m4a": float(i) for i in range(80)}  # eighty clips all at once: eighty rows
    out = lay(state([crowded]), height=300)
    assert len(out.bars) == 80 and all(b.h == Metrics().bar_h for b in out.bars)  # not thinned to nothing, not left out
    assert out.content_h > 80 * Metrics().bar_h and fit_check(out) == []


def test_a_view_with_room_to_spare_is_not_made_larger_than_it_is():
    out = lay(state([{"a.m4a": 0, "b.m4a": 30}]), width=900, height=600)
    assert (out.content_w, out.content_h) == (900, 600)  # nothing to scroll to


def test_a_long_group_is_kept_readable_and_scrolls_sideways_instead_of_being_squeezed_in():
    st = state([{"a.m4a": 0.0, "b.m4a": 5000.0}], durations={"a.m4a": 1200.0, "b.m4a": 1200.0})  # 6200 s
    out = lay(st, width=900)
    assert out.scale == pytest.approx(Metrics().min_scale) and out.content_w == pytest.approx(2 * Metrics().pad + 6200 * out.scale)
    assert out.content_w > 900 and fit_check(out) == []
    short = lay(state([{"a.m4a": 0.0, "b.m4a": 50.0}], durations={"a.m4a": 100.0, "b.m4a": 100.0}), width=900)
    assert short.scale > Metrics().min_scale and short.content_w == 900  # a short one is stretched to the view, as before


def test_zooming_stretches_the_time_scale_and_zooming_out_stops_at_the_whole_group_in_view():
    st = state([{"a.m4a": 0.0, "b.m4a": 5000.0}], durations={"a.m4a": 1200.0, "b.m4a": 1200.0})
    usual, closer, further = lay(st, width=900), lay(st, width=900, zoom=4), lay(st, width=900, zoom=0.001)
    assert closer.scale == pytest.approx(4 * usual.scale) and closer.zoom == pytest.approx(4)
    assert further.content_w == 900 and further.zoom < 1  # all of it in view, and no further
    assert closer.content_w > usual.content_w > further.content_w
    huge = lay(st, width=900, zoom=1e9)
    assert huge.content_w <= 60100  # however far one zooms, the content stays a size a canvas copes with


def test_time_mark_text():
    assert [tick_text(t) for t in (0, 30, 60, 90, 600, 3600, 3900)] == ["0:00", "0:30", "1:00", "1:30", "10:00", "1:00:00", "1:05:00"]


def test_bars_grow_thicker_when_the_panel_has_room_to_spare_and_stay_a_readable_size_when_it_has_not():
    st = state([{"a.m4a": 0, "b.m4a": 30, "c.m4a": 70}])
    cramped, roomy, tall = lay(st, height=150), lay(st, height=320), lay(st, height=900)
    assert cramped.bars[0].h == Metrics().bar_h < roomy.bars[0].h  # no room: the usual size, and the content scrolls
    assert cramped.content_h > cramped.height
    assert Metrics().bar_h < tall.bars[0].h == Metrics().max_bar_h  # the more room, the thicker, up to a limit
    assert all(fit_check(x) == [] for x in (cramped, roomy, tall))


def test_thicker_bars_never_make_what_would_have_fitted_scroll():
    groups = [{f"g{k}a.m4a": 0, f"g{k}b.m4a": 40, f"g{k}c.m4a": 80} for k in range(12)]
    for height in range(260, 900, 40):
        st = state(groups)
        plain = lay(st, height=height, m=Metrics(max_bar_h=Metrics().bar_h))  # no growing
        grown = lay(st, height=height)
        assert (grown.content_h == height) or plain.content_h > height  # it fitted with the usual bars: it fits with these
        assert fit_check(grown) == []


def test_the_tray_labels_get_the_room_they_need_on_a_sharper_screen():
    st = state([{"a.m4a": 0, "b.m4a": 30}], pending=["w1.m4a"], current="cur.m4a")
    sharp = lay(st, m=Metrics().scaled(2.0))
    assert min(c.x for c in sharp.chips) >= Metrics().scaled(2.0).pad + Metrics().scaled(2.0).label_w  # clear of the words


def test_metrics_scale_for_a_sharper_screen():
    big = Metrics().scaled(2.0)
    assert big.bar_h == 2 * Metrics().bar_h and big.min_label_chars == Metrics().min_label_chars


def test_a_layout_never_puts_bars_out_of_bounds_or_on_top_of_each_other():
    rng = random.Random(7)
    for _ in range(60):
        names = [f"c{i}.m4a" for i in range(rng.randint(2, 40))]
        durations = {n: rng.uniform(20, 900) for n in names}
        cut = rng.randint(0, len(names))
        members = names[:cut]
        groups, k = [], 0
        while k < len(members):
            size = rng.randint(1, 8)
            groups.append({n: rng.uniform(-200, 3000) for n in members[k:k + size]})
            k += size
        st = MatchState(durations=durations, groups=tuple(groups), pending=tuple(names[cut:]))
        out = layout(st, make_labels(names, {}), {}, rng.choice([420, 900, 1600]), rng.choice([240, 360, 520]))
        assert fit_check(out) == [], (fit_check(out), groups)


def test_ranking_is_by_size_then_length():
    d = {"a": 100.0, "b": 100.0, "c": 500.0, "d": 100.0, "e": 100.0}
    ranked = rank_groups([{"a": 0, "b": 0}, {"c": 0, "d": 0}, {"e": 0}], d)
    assert [sorted(g) for g in ranked] == [["c", "d"], ["a", "b"], ["e"]]  # the longer pair first, the lone clip last


def test_clock_text():
    assert [clock(x) for x in (0, 45, 59.6, 600, 3900)] == ["0 s", "45 s", "1 min", "10 min", "1 h 05 min"]
