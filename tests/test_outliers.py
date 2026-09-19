"""A weak match must be backed by the rest of the group, and a cluster that hangs on one is looked at again when all is known.

What happened in a real run: a clip of another night joined a big group by a match of z=11.5 that no other clip confirmed,
five clips of its own night then joined it, and there was a cluster at the end of the group that did not belong.
"""
import numpy as np
import pytest
from synth import phone, synth_music, write_wav
from test_groups import names

from meld import sync as sync_mod
from meld.project import Project
from meld.sync import AR, CONFIDENT_Z, UNVERIFIED_Z, _Groups, audit_groups, clusters_agree, sync_project

SR = 16000
A = synth_music(300, SR, seed=11)  # one night
B = synth_music(300, SR, seed=12)  # another (as far as the sound goes: nothing in common)


def make(tmp_path, spec: dict) -> Project:
    """spec: name -> (music, from, to). Every clip is a different length, so that a test can tell which is which."""
    project = Project(tmp_path)
    for i, (name, (music, t0, t1)) in enumerate(spec.items()):
        write_wav(project.clips_dir / f"{name}.wav", phone(music, SR, t0, t1, seed=i + 1), SR)
    return project


def forced(monkeypatch, pairs: dict, believed: bool = True):
    """`correlate` as it is, except that the pairs given (by the length of the two clips, in seconds) get the (lag, z) given.
    With `believed` those pairs also pass the check of `support`, as the same song of another night really does: the
    sound is alike, and it is the rest of the group that does not agree."""
    real_correlate, real_support = sync_mod.correlate, sync_mod.support

    def key(a, b, fs):
        return (round(len(a) / fs), round(len(b) / fs))

    def altered(a, b, fs, band=(150.0, 5000.0), min_overlap=5.0):
        got = pairs.get(key(a, b, fs))
        return got if got is not None else real_correlate(a, b, fs, band, min_overlap)

    def supported(a, b, lag, fs, **kw):
        return [20.0, 20.0, 20.0] if believed and key(a, b, fs) in pairs else real_support(a, b, lag, fs, **kw)

    monkeypatch.setattr(sync_mod, "correlate", altered)
    monkeypatch.setattr(sync_mod, "support", supported)


def sync(project, **kw):
    lines = []
    tl = sync_project(project, log=lines.append, workers=1, **kw)
    return tl, lines


def groups_of(tl):
    return [set(names(tl.clips))] + [set(names(g)) for g in tl.others]


# night A: five clips that overlap well, in a chain with plenty of overlap (each is there while three others are);
# night B: four clips, shorter, so that they are looked at after the clips of A, and b1 the longest of them, so that it
# is looked at first: it is the one that is wrongly matched, and the others come in after it
NIGHT_A = {"a1": (A, 0, 100), "a2": (A, 30, 128), "a3": (A, 60, 155), "a4": (A, 90, 182), "a5": (A, 120, 209)}
NIGHT_B = {"b1": (B, 100, 165), "b2": (B, 100, 152), "b3": (B, 105, 156), "b4": (B, 110, 160)}


def test_a_weak_link_that_no_other_clip_confirms_does_not_bring_a_cluster_in(tmp_path, monkeypatch):
    project = make(tmp_path, {**NIGHT_A, **NIGHT_B})
    # b1, of the other night, matches a3 at z=15 as if it started at 100 s (a3 is at 60 s: 40 s later)
    forced(monkeypatch, {(95, 65): (40.0, 15.0)})
    tl, lines = sync(project)
    assert groups_of(tl) == [{"a1", "a2", "a3", "a4", "a5"}, {"b1", "b2", "b3", "b4"}]  # two nights, not one
    assert any("b1" in line and "not backed up" in line for line in lines)


def test_without_the_check_that_link_would_have_brought_the_cluster_in(tmp_path, monkeypatch):
    # what the test above guards against: it is real that a weak link like this one puts b1 in the group of A
    project = make(tmp_path, {**NIGHT_A, **NIGHT_B})
    forced(monkeypatch, {(95, 65): (40.0, 15.0)})
    monkeypatch.setattr(sync_mod, "CORROBORATE_FROM", 10**6)  # the rule switched off: nothing is asked of the others
    monkeypatch.setattr(sync_mod, "audit_groups", lambda *a, **k: 0)
    tl, _ = sync(project)
    assert {"b1"} <= set(names(tl.clips)) and len(tl.clips) >= 6  # b1 joined A, and the others of its night came in with it


def test_a_weak_link_that_the_other_clips_confirm_is_believed(tmp_path, monkeypatch):
    project = make(tmp_path, NIGHT_A)
    # every match of a4 (92 s long, at 90 s) is a real one, but is scored weakly
    forced(monkeypatch, {(100, 92): (90.0, 15.0), (98, 92): (60.0, 15.0), (95, 92): (30.0, 15.0), (92, 89): (30.0, 15.0)}, believed=False)
    tl, _ = sync(project)
    assert set(names(tl.clips)) == set(NIGHT_A)  # the clips of the group agree with it: it is in
    assert tl.others == []


def test_a_weak_link_with_nobody_there_to_ask_is_believed_only_if_it_is_fairly_strong(tmp_path, monkeypatch):
    # a4 is the only clip of the group that is there at the end (a3 ends at 155 s, a4 is 90 to 182 s: only a3 overlaps it
    # enough, and it is the one it matched, so there is nobody else to ask)
    spec = {"a1": (A, 0, 100), "a2": (A, 30, 128), "a3": (A, 60, 155), "a4": (A, 145, 236)}
    pair = (95, 91)
    forced(monkeypatch, {pair: (85.0, 15.0)})
    tl, lines = sync(make(tmp_path / "weak", spec))
    assert "a4" not in names(tl.clips) and any("a4" in line and "not backed up" in line for line in lines)
    assert UNVERIFIED_Z > 15 and UNVERIFIED_Z < CONFIDENT_Z
    forced(monkeypatch, {pair: (85.0, UNVERIFIED_Z + 1.0)})
    tl, _ = sync(make(tmp_path / "stronger", spec))
    assert "a4" in names(tl.clips)


def test_a_clip_that_matches_two_groups_weakly_does_not_join_them(tmp_path, monkeypatch):
    # night A (five clips) and night B (four); c is a clip of A, short so that it is looked at last, that also matches
    # b1 at z=15 as if it started 8 s into it: it would be the bridge that makes the two nights one
    spec = {**NIGHT_A, **NIGHT_B, "c": (A, 70, 116)}
    forced(monkeypatch, {(65, 46): (8.0, 15.0)})
    tl, lines = sync(make(tmp_path, spec))
    assert groups_of(tl) == [{"a1", "a2", "a3", "a4", "a5", "c"}, {"b1", "b2", "b3", "b4"}]
    assert not any("links two groups" in line for line in lines)
    assert any("c.wav: its match with b1.wav" in line and "not backed up" in line for line in lines)


def test_a_clip_that_is_there_for_both_pieces_of_a_night_still_joins_them(tmp_path):
    # the same night in two pieces that no clip of either covers together, and a shorter clip (so that it is looked at once
    # the pieces are there) that is there for the end of one and the start of the other: it joins them, which is right, so
    # it must still be possible
    long_night = synth_music(600, SR, seed=13)
    spec = {"p1": (long_night, 0, 250), "p2": (long_night, 5, 253), "p3": (long_night, 10, 256),
            "q1": (long_night, 350, 600), "q2": (long_night, 352, 598), "q3": (long_night, 355, 597),
            "br": (long_night, 240, 360)}
    tl, lines = sync(make(tmp_path, spec))
    assert groups_of(tl) == [set(spec)]
    assert any("links two groups" in line for line in lines)
    offsets = {c.file.removesuffix(".wav"): c.offset for c in tl.clips}
    assert offsets["q1"] - offsets["p1"] == pytest.approx(350.0, abs=0.01) and offsets["br"] == pytest.approx(240.0, abs=0.01)


# ---- setting a cluster apart afterwards, when all is known


def audio_of(spec):
    return {n: phone(m, SR, t0, t1, seed=i + 1) for i, (n, (m, t0, t1)) in enumerate(spec.items())}


def group_of(offsets: dict, links: dict) -> _Groups:
    """A group as sorting leaves it: offsets in its own time, and for each clip what it was lined up with, and how well."""
    g = _Groups()
    first = next(iter(offsets))
    gid = g.seed(first)
    for n, off in offsets.items():
        if n != first:
            anchor, z = links[n]
            g.add(gid, n, off, anchor, z)
    return g


def test_a_cluster_hanging_on_one_weak_link_that_does_not_sound_like_the_group_is_set_apart():
    spec = {**NIGHT_A, "b1": (B, 100, 147), "b2": (B, 100, 152), "b3": (B, 105, 156)}
    audio = audio_of(spec)
    offsets = {"a1": 0.0, "a2": 30.0, "a3": 60.0, "a4": 90.0, "a5": 120.0, "b1": 100.0, "b2": 100.0, "b3": 105.0}
    links = {"a2": ("a1", 60.0), "a3": ("a2", 55.0), "a4": ("a3", 50.0), "a5": ("a4", 45.0),
             "b1": ("a3", 11.5), "b2": ("b1", 40.0), "b3": ("b2", 38.0)}  # the cluster hangs on a3 by one link
    g = group_of(offsets, links)
    lines = []
    assert audit_groups(g, audio, lines.append) == 1
    sizes = sorted(len(m) for m in g.members.values())
    assert sizes == [3, 5] and g.home["b1"] == g.home["b2"] == g.home["b3"] != g.home["a1"]
    assert g.link["b1"] == (None, None) and g.link["b2"] == ("b1", 40.0)  # the cluster keeps its own links, from its top
    assert any("b1" in line and "2 clip(s) that hang on it" in line and "does not sound like them" in line for line in lines)


def test_a_cluster_on_a_weak_link_that_does_sound_like_the_group_stays():
    audio = audio_of(NIGHT_A)
    offsets = {"a1": 0.0, "a2": 30.0, "a3": 60.0, "a4": 90.0, "a5": 120.0}
    links = {"a2": ("a1", 60.0), "a3": ("a2", 12.0), "a4": ("a3", 14.0), "a5": ("a4", 45.0)}  # weak links, but the truth
    g = group_of(offsets, links)
    assert audit_groups(g, audio, lambda *_: None) == 0
    assert len(g.members) == 1 and len(next(iter(g.members.values()))) == 5


def test_a_cluster_that_nothing_else_was_there_for_stays_only_if_its_link_is_fairly_strong():
    spec = {"a1": (A, 0, 100), "a2": (A, 30, 128), "a3": (A, 60, 155), "a4": (A, 90, 182), "e1": (A, 200, 250), "e2": (A, 205, 262)}
    audio = audio_of(spec)
    offsets = {"a1": 0.0, "a2": 30.0, "a3": 60.0, "a4": 90.0, "e1": 200.0, "e2": 205.0}  # e1, e2: 200 s on, nobody else there
    for z, kept in ((12.0, False), (UNVERIFIED_Z + 2.0, True)):
        links = {"a2": ("a1", 60.0), "a3": ("a2", 55.0), "a4": ("a3", 50.0), "e1": ("a4", z), "e2": ("e1", 40.0)}
        g = group_of(offsets, links)
        lines = []
        assert audit_groups(g, audio, lines.append) == (0 if kept else 1), z
        if not kept:
            assert any("nothing else in the group was there to agree" in line for line in lines)


def test_strong_links_are_not_questioned():
    spec = {**NIGHT_A, "b1": (B, 100, 147)}
    audio = audio_of(spec)
    offsets = {"a1": 0.0, "a2": 30.0, "a3": 60.0, "a4": 90.0, "a5": 120.0, "b1": 100.0}
    links = {"a2": ("a1", 60.0), "a3": ("a2", 55.0), "a4": ("a3", 50.0), "a5": ("a4", 45.0), "b1": ("a3", CONFIDENT_Z + 1)}
    assert audit_groups(group_of(offsets, links), audio, lambda *_: None) == 0  # a strong link is taken as it stands


def test_small_groups_are_not_audited():
    audio = audio_of({"a1": (A, 0, 100), "a2": (A, 30, 128), "b1": (B, 100, 147)})
    g = group_of({"a1": 0.0, "a2": 30.0, "b1": 100.0}, {"a2": ("a1", 60.0), "b1": ("a2", 11.5)})
    assert audit_groups(g, audio, lambda *_: None) == 0


def test_what_hangs_on_a_cluster_that_is_set_apart_goes_with_it_and_the_outermost_link_is_looked_at_first():
    spec = {**NIGHT_A, "b1": (B, 100, 147), "b2": (B, 100, 152), "b3": (B, 105, 156), "b4": (B, 110, 165)}
    audio = audio_of(spec)
    offsets = {"a1": 0.0, "a2": 30.0, "a3": 60.0, "a4": 90.0, "a5": 120.0, "b1": 100.0, "b2": 100.0, "b3": 105.0, "b4": 110.0}
    links = {"a2": ("a1", 60.0), "a3": ("a2", 55.0), "a4": ("a3", 50.0), "a5": ("a4", 45.0),
             "b1": ("a3", 11.5), "b2": ("b1", 14.0), "b3": ("b2", 13.0), "b4": ("b3", 12.0)}  # weak all the way
    g = group_of(offsets, links)
    lines = []
    assert audit_groups(g, audio, lines.append) == 1  # one cluster, not four: b1 goes, and b2, b3, b4 with it
    assert sorted(len(m) for m in g.members.values()) == [4, 5]
    assert len([line for line in lines if "set apart" in line]) == 1


# ---- the tree of links, and how a merge keeps it a tree

def test_a_merge_turns_the_links_of_the_other_group_round_to_hang_from_the_bridge():
    g = _Groups()
    m = g.seed("m1")
    g.add(m, "m2", 10.0, "m1", 50.0)
    o = g.seed("o1")
    g.add(o, "o2", 5.0, "o1", 40.0)
    g.add(o, "o3", 9.0, "o2", 30.0)  # o1 <- o2 <- o3
    g.add(m, "u", 20.0, "m2", 45.0)  # the bridge, hanging on m2 ...
    g.merge(m, o, 100.0, bridge=("u", "o3", 22.0))  # ... and lined up with o3
    assert g.link["o3"] == ("u", 22.0)  # o3 now hangs on the bridge
    assert g.link["o2"] == ("o3", 30.0) and g.link["o1"] == ("o2", 40.0)  # and o2, o1 below it, the same links turned round
    assert g.subtree("o3") == {"o3", "o2", "o1"} and g.subtree("u") == {"u", "o3", "o2", "o1"}
    assert sum(1 for n in g.members[m] if g.link[n][0] is None) == 1  # one tree: one clip at the top
    assert g.members[m]["o1"] == 100.0 and g.members[m]["o2"] == 105.0  # the times moved with it


def test_detaching_makes_a_group_of_its_own_with_its_own_top():
    g = _Groups()
    m = g.seed("a")
    g.add(m, "b", 1.0, "a", 50.0)
    g.add(m, "c", 2.0, "b", 12.0)
    g.add(m, "d", 3.0, "c", 40.0)
    new = g.detach(g.subtree("c"), "c")
    assert g.members[new] == {"c": 2.0, "d": 3.0} and g.members[m] == {"a": 0.0, "b": 1.0}
    assert g.link["c"] == (None, None) and g.link["d"] == ("c", 40.0)
    assert g.home["c"] == g.home["d"] == new


# ---- do two sets of clips sound the same where they are both there?

def test_clusters_that_are_the_same_night_agree_and_those_of_another_do_not():
    audio = audio_of({**NIGHT_A, **NIGHT_B})
    offsets = {n: float(off) for n, off in zip(audio, [0, 30, 60, 90, 120, 100, 100, 105, 110])}
    assert clusters_agree({"a1", "a2"}, {"a3", "a4"}, offsets, audio) is True
    assert clusters_agree({"b1", "b2"}, {"a3", "a4"}, offsets, audio) is False
    assert clusters_agree({"a1"}, {"a5"}, offsets, audio) is None  # a1 ends at 100 s, a5 begins at 120: never there together


def test_the_saved_sorting_is_the_one_after_the_audit(tmp_path, monkeypatch):
    project = make(tmp_path, {**NIGHT_A, **NIGHT_B})
    forced(monkeypatch, {(95, 65): (40.0, 15.0)})
    first, _ = sync(project, remember=True)
    second, lines = sync(project, remember=True)
    assert any("before: using that" in line for line in lines)
    assert groups_of(first) == groups_of(second) == [{"a1", "a2", "a3", "a4", "a5"}, {"b1", "b2", "b3", "b4"}]
