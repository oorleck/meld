"""Sorting clips into groups that line up with each other, all at once (sync_project's default)."""
import json

import pytest
from synth import phone, synth_music, write_wav

from meld.project import Project
from meld.sync import estimate_remaining_groups, sync_project
from meld.timeline import Timeline

SR = 16000
A = synth_music(200, SR, seed=1)  # one concert
B = synth_music(200, SR, seed=2)  # another


def project_with(tmp_path, spec: dict) -> Project:
    """spec: name -> (music, start, end): a clip of that stretch of that concert's music, as a phone would hear it."""
    project = Project(tmp_path)
    for i, (name, (music, t0, t1)) in enumerate(spec.items()):
        write_wav(project.clips_dir / f"{name}.wav", phone(music, SR, t0, t1, seed=i + 1), SR)
    return project


def names(clips) -> dict:
    return {c.file.removesuffix(".wav"): round(c.offset, 2) for c in clips}


def sync(project, **kw):
    return sync_project(project, log=lambda *_: None, **kw)


TWO_CONCERTS = {
    "a1": (A, 0, 60), "b1": (B, 0, 50), "a2": (A, 40, 90), "b2": (B, 30, 80),
    "a3": (A, 80, 130), "b3": (B, 70, 110), "a4": (A, 120, 170),
}


def test_two_concerts_become_two_groups_and_the_bigger_is_used(tmp_path):
    tl = sync(project_with(tmp_path, TWO_CONCERTS))
    assert names(tl.clips) == {"a1": 0.0, "a2": 40.0, "a3": 80.0, "a4": 120.0}  # times are right within the group
    assert [names(g) for g in tl.others] == [{"b1": 0.0, "b2": 30.0, "b3": 70.0}]
    assert tl.rejected == [] and tl.left_out == 3


def test_the_longest_clip_no_longer_decides_which_concert_wins(tmp_path):
    spec = {"b0": (B, 0, 100), "a1": (A, 0, 60), "a2": (A, 40, 90), "a3": (A, 80, 130), "a4": (A, 120, 170), "b1": (B, 90, 140)}
    tl = sync(project_with(tmp_path, spec))
    assert set(names(tl.clips)) == {"a1", "a2", "a3", "a4"}  # the bigger group, though b0 is the longest clip
    assert [set(names(g)) for g in tl.others] == [{"b0", "b1"}]


def test_the_original_single_group_way_is_still_there(tmp_path):
    spec = {"b0": (B, 0, 100), "a1": (A, 0, 60), "a2": (A, 40, 90), "b1": (B, 90, 140)}
    tl = sync(project_with(tmp_path, spec), clusters=False)
    assert set(names(tl.clips)) == {"b0", "b1"}  # grown from the longest clip, whatever it is
    assert tl.others == [] and {r["file"] for r in tl.rejected} == {"a1.wav", "a2.wav"}


def test_a_clip_that_overlaps_two_groups_joins_them_into_one(tmp_path):
    # x and y do not overlap each other; the shorter z overlaps both. x and y are placed first (they are longer) and
    # cannot match, so they start out as separate groups; z is the bridge.
    spec = {"x": (A, 0, 70), "y": (A, 100, 170), "z": (A, 60, 110)}
    tl = sync(project_with(tmp_path, spec))
    assert names(tl.clips) == {"x": 0.0, "z": 60.0, "y": 100.0}
    assert tl.others == [] and tl.rejected == []


def test_two_short_clips_that_only_meet_in_the_second_look_still_form_a_group(tmp_path, monkeypatch):
    from meld import sync as sync_mod

    # the long clips of concert A use up each clip's comparisons before the two B clips ever meet
    spec = {f"a{i}": (A, i * 30, i * 30 + 60) for i in range(5)}
    spec |= {"b1": (B, 0, 20), "b2": (B, 12, 32)}

    def grouped(where):
        tl = sync(project_with(where, spec), max_compare=3)
        return {"b1", "b2"} <= {n for g in tl.others for n in names(g)}

    assert grouped(tmp_path)
    monkeypatch.setattr(sync_mod, "LOOSE_PASSES", 0)  # control: without the second look they never meet
    assert not grouped(tmp_path / "control")


def test_a_clip_that_matches_nothing_is_rejected_and_says_why(tmp_path):
    spec = {"a1": (A, 0, 60), "a2": (A, 40, 90), "a3": (A, 80, 130), "lonely": (B, 0, 40)}
    tl = sync(project_with(tmp_path, spec))
    assert set(names(tl.clips)) == {"a1", "a2", "a3"} and tl.others == []
    assert [r["file"] for r in tl.rejected] == ["lonely.wav"]
    assert "no confident audio match" in tl.rejected[0]["reason"]


def test_cluster_option_takes_another_group(tmp_path):
    project = project_with(tmp_path, TWO_CONCERTS)
    tl = sync(project, cluster=2)
    assert set(names(tl.clips)) == {"b1", "b2", "b3"} and [set(names(g)) for g in tl.others] == [{"a1", "a2", "a3", "a4"}]
    with pytest.raises(SystemExit, match="no group 3"):
        sync(project, cluster=3)


def test_choose_is_asked_only_between_groups_worth_choosing_between(tmp_path):
    asked = []

    def choose(groups):
        asked.append([len(g.clips) for g in groups])
        return 1  # the smaller one

    tl = sync(project_with(tmp_path, TWO_CONCERTS), choose=choose)
    assert asked == [[4, 3]] and set(names(tl.clips)) == {"b1", "b2", "b3"}

    tl = sync(project_with(tmp_path / "again", TWO_CONCERTS), choose=lambda groups: None)  # None: the biggest
    assert set(names(tl.clips)) == {"a1", "a2", "a3", "a4"}

    # a group of two is not worth offering: with only one group of three or more there is nothing to ask
    spec = {"a1": (A, 0, 60), "a2": (A, 40, 90), "a3": (A, 80, 130), "b1": (B, 0, 50), "b2": (B, 30, 80)}
    tl = sync(project_with(tmp_path / "small", spec), choose=lambda groups: pytest.fail("nothing to choose between"))
    assert set(names(tl.clips)) == {"a1", "a2", "a3"}


def test_the_timeline_file_keeps_the_other_groups_and_old_files_still_load(tmp_path):
    tl = sync(project_with(tmp_path, TWO_CONCERTS))
    loaded = Timeline.load(tmp_path / "timeline.json")
    assert names(loaded.clips) == names(tl.clips) and [names(g) for g in loaded.others] == [names(g) for g in tl.others]
    assert loaded.left_out == 3

    clip = {"file": "a.mp4", "offset": 0.0, "duration": 5.0, "has_video": True, "width": 640, "height": 360}
    (tmp_path / "old.json").write_text(json.dumps({"version": 1, "clips": [clip], "rejected": [{"file": "b.mp4"}]}))
    old = Timeline.load(tmp_path / "old.json")  # written before there were groups
    assert [c.file for c in old.clips] == ["a.mp4"] and old.others == [] and old.left_out == 1


def test_clips_are_compared_with_a_bounded_number_of_others(tmp_path, monkeypatch):
    import numpy as np

    from meld import sync as sync_mod

    calls = []

    def fake_correlate(a, b, fs, band=None, min_overlap=5.0):
        calls.append(1)
        loud = float(np.abs(a).max()) > 0.4 or float(np.abs(b).max()) > 0.4
        return 0.0, (1.0 if loud else 40.0)  # the loud clips are from other concerts and match nothing

    monkeypatch.setattr(sync_mod, "correlate", fake_correlate)
    project = Project(tmp_path)
    for i in range(30):
        tone = (0.6 if i % 3 else 0.2) * np.sin(2 * np.pi * 440 * np.arange(8 * SR) / SR).astype(np.float32)
        write_wav(project.clips_dir / f"c{i:02d}.wav", tone, SR)
    sync(project, max_compare=6)
    # never anything like all-against-all (435 pairs): about max_compare per clip, and twice that at the very most
    assert len(calls) <= 30 * 6 * 2


# ---- the estimator for groups


def test_group_estimate_is_capped_and_exact_once_the_second_look_is_known():
    lengths = [1_000_000] * 30
    expected, worst = estimate_remaining_groups(25, 1e-6, lengths[:10], lengths[10:], 60, [1_000_000] * 3, 40)
    assert 0 < expected <= worst
    # once the second look has begun, its remaining cost is given, and that is the expected time
    expected, worst = estimate_remaining_groups(25, 1e-6, lengths, [], 300, [], 300, second_look=42.0)
    assert expected == 42.0 and worst >= 42.0
    # there are only so many pairs: nothing left to compare means nothing left to wait for
    every_pair = 30 * 29 // 2
    assert estimate_remaining_groups(25, 1e-6, lengths, [], 300, [], every_pair, second_look=42.0) == (0.0, 0.0)


def test_group_estimate_is_not_tiny_before_anything_is_known():
    expected, worst = estimate_remaining_groups(25, 1e-6, [1_000_000], [1_000_000] * 40, 0, [], 0)
    assert expected > 0.3 * worst
