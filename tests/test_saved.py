"""What is remembered between runs of the same search (see cache.py): sorting again, a changed set of clips, a new Meld."""
import os
import time
from pathlib import Path

import pytest
from synth import phone, synth_music, write_wav
from test_groups import A, B, SR, TWO_CONCERTS, names, project_with

from meld import cache
from meld import sync as sync_mod
from meld.matchview import MatchState
from meld.project import Project
from meld.sync import sync_project


@pytest.fixture
def calls(monkeypatch):
    """A count of the comparisons really made (`correlate` calls), for runs one at a time."""
    made = []
    real = sync_mod.correlate

    def counting(a, b, fs, band=(150.0, 5000.0), min_overlap=5.0):
        made.append(1)
        return real(a, b, fs, band, min_overlap)

    monkeypatch.setattr(sync_mod, "correlate", counting)
    return made


def sync(project, **kw):
    return sync_project(project, log=kw.pop("log", lambda *_: None), workers=1, remember=True, **kw)


def groups_of(tl):
    return names(tl.clips), [names(g) for g in tl.others], sorted(r["file"] for r in tl.rejected)


# ---- the same clips, sorted again

def test_sorting_the_same_clips_again_makes_no_comparisons_and_gives_the_same_groups(tmp_path, calls):
    project = project_with(tmp_path, TWO_CONCERTS)
    first = sync(project)
    made = len(calls)
    assert made > 0
    lines = []
    again = sync(project, log=lines.append)
    assert len(calls) == made  # nothing compared, nothing decoded
    assert groups_of(again) == groups_of(first)
    assert [c.z for c in again.clips] == [c.z for c in first.clips] and [c.anchor for c in again.clips] == [c.anchor for c in first.clips]
    assert any("before: using that" in line for line in lines)


def test_another_group_can_be_chosen_from_the_saved_sorting(tmp_path, calls):
    project = project_with(tmp_path, TWO_CONCERTS)
    first = sync(project, choose=lambda groups: 0)
    made = len(calls)
    second = sync(project, choose=lambda groups: 1)  # the other concert this time
    assert len(calls) == made
    assert set(names(first.clips)) == {"a1", "a2", "a3", "a4"} and set(names(second.clips)) == {"b1", "b2", "b3"}
    assert [set(names(g)) for g in second.others] == [{"a1", "a2", "a3", "a4"}]
    assert sorted(r["file"] for r in second.rejected) == sorted(r["file"] for r in first.rejected)


def test_the_view_is_shown_where_the_sorting_ended_even_when_it_was_not_redone(tmp_path):
    project = project_with(tmp_path, TWO_CONCERTS)
    sync(project)
    states: list[MatchState] = []
    sync(project, on_state=states.append)
    (final,) = states
    assert final.finished and final.chosen == frozenset({"a1.wav", "a2.wav", "a3.wav", "a4.wav"})
    assert len(final.groups) == 2 and set(final.durations) == {f"{n}.wav" for n in TWO_CONCERTS}


# ---- when the clips change

def test_a_clip_that_is_new_costs_only_its_own_comparisons_and_the_result_is_that_of_a_fresh_run(tmp_path, calls):
    spec = dict(TWO_CONCERTS)
    fewer = {k: v for k, v in spec.items() if k != "a4"}
    project = project_with(tmp_path / "a", fewer)
    sync(project)
    before = len(calls)
    write_wav(project.clips_dir / "a4.wav", phone(A, SR, 120, 170, seed=7), SR)  # one more clip turns up
    grown = sync(project)
    extra = len(calls) - before

    fresh_project = project_with(tmp_path / "b", {**fewer})
    write_wav(fresh_project.clips_dir / "a4.wav", phone(A, SR, 120, 170, seed=7), SR)
    calls.clear()
    fresh = sync_project(fresh_project, log=lambda *_: None, workers=1)  # no memory at all
    assert len(calls) > extra > 0  # the new clip's own comparisons only
    assert groups_of(grown) == groups_of(fresh)


def test_a_run_that_was_stopped_goes_on_from_where_it_got_to(tmp_path, calls):
    sync_project(project_with(tmp_path / "fresh", TWO_CONCERTS), log=lambda *_: None, workers=1)
    full = len(calls)  # what the whole job costs
    calls.clear()

    project = project_with(tmp_path / "stopped", TWO_CONCERTS)

    def stop_after_a_few(msg=""):
        if len(calls) >= 6:
            raise KeyboardInterrupt  # what the window's Cancel does: an exception out of the log

    with pytest.raises(KeyboardInterrupt):
        sync_project(project, log=stop_after_a_few, workers=1, remember=True)
    done = len(calls)
    calls.clear()
    finished = sync(project)
    assert done >= 6 and 0 < len(calls) < full  # more to do, but not what was done before
    assert groups_of(finished) == groups_of(sync_project(project_with(tmp_path / "again", TWO_CONCERTS), log=lambda *_: None, workers=1))


def test_a_changed_file_is_not_trusted_but_the_others_are(tmp_path, calls):
    project = project_with(tmp_path, TWO_CONCERTS)
    sync(project)
    made = len(calls)
    write_wav(project.clips_dir / "a2.wav", phone(A, SR, 40, 85, seed=9), SR)  # not the clip it was: shorter
    again = sync(project)
    assert 0 < len(calls) - made < made  # a2's own pairs, not everyone's
    assert names(again.clips)["a2"] == 40.0  # and the answer is for the new file


def test_settings_that_change_the_answer_are_part_of_what_is_remembered(tmp_path, calls):
    project = project_with(tmp_path, TWO_CONCERTS)
    sync(project)
    made = len(calls)
    sync_project(project, log=lambda *_: None, workers=1, remember=True, min_z=12.0)  # another threshold: groups redone ...
    assert len(calls) == made  # ... from the same scores, which do not depend on it
    sync_project(project, log=lambda *_: None, workers=1, remember=True, min_overlap=8.0)  # what a score means: all again
    assert len(calls) > made


def test_it_is_only_remembered_when_asked_to(tmp_path, calls):
    project = project_with(tmp_path, TWO_CONCERTS)
    sync_project(project, log=lambda *_: None, workers=1)
    made = len(calls)
    sync_project(project, log=lambda *_: None, workers=1)
    assert len(calls) == 2 * made and not (project.root / cache.SAVED).exists()


def test_only_the_named_clips_are_used(tmp_path):
    project = project_with(tmp_path, TWO_CONCERTS)
    tl = sync(project, only={"a1", "a2", "a3", "b1"})
    assert set(names(tl.clips)) == {"a1", "a2", "a3"}
    assert [r["file"] for r in tl.rejected] == ["b1.wav"]
    with pytest.raises(SystemExit):
        sync(project, only={"nothing"})


# ---- a new version of Meld

def test_what_an_older_meld_worked_out_is_not_used(tmp_path, calls, monkeypatch):
    project = project_with(tmp_path, TWO_CONCERTS)
    sync(project)
    made = len(calls)
    monkeypatch.setattr(cache, "__version__", "9.9.9")  # the next Meld
    sync(project)
    assert len(calls) == 2 * made


def test_stamping_drops_what_an_older_meld_made_but_keeps_the_downloads(tmp_path, monkeypatch):
    project = project_with(tmp_path, TWO_CONCERTS)
    assert cache.stamp(project) is False  # first time
    sync(project)
    decoded = list((project.cache_dir / "audio").iterdir())
    assert decoded and (project.root / cache.SAVED).exists()
    assert cache.stamp(project) is False  # the same version: all stays
    assert list((project.cache_dir / "audio").iterdir()) == decoded

    monkeypatch.setattr(cache, "__version__", "9.9.9")
    assert cache.stamp(project) is True
    assert not (project.root / cache.SAVED).exists() and not list((project.cache_dir).iterdir())
    assert len(list(project.clips_dir.iterdir())) == len(TWO_CONCERTS)  # what YouTube sent is not Meld's to throw away


# ---- the store itself

def test_a_result_is_used_only_for_the_same_inputs_and_version(tmp_path, monkeypatch):
    cache.save(tmp_path, "thing", cache.key_of("a", 1), {"x": [1, 2]})
    assert cache.load(tmp_path, "thing", cache.key_of("a", 1)) == {"x": [1, 2]}
    assert cache.load(tmp_path, "thing", cache.key_of("a", 2)) is None
    assert cache.load(tmp_path, "other", cache.key_of("a", 1)) is None
    monkeypatch.setattr(cache, "__version__", "9.9.9")
    assert cache.load(tmp_path, "thing", cache.key_of("a", 1)) is None


def test_a_result_can_expire(tmp_path):
    cache.save(tmp_path, "thing", "k", 1)
    assert cache.load(tmp_path, "thing", "k", max_age=3600) == 1
    old = time.time() - 7200
    os.utime(tmp_path / cache.SAVED / "thing.json", (old, old))  # the file's time does not matter: what it says does
    text = (tmp_path / cache.SAVED / "thing.json").read_text(encoding="utf-8")
    import json

    data = json.loads(text)
    data["at"] = old
    (tmp_path / cache.SAVED / "thing.json").write_text(json.dumps(data), encoding="utf-8")
    assert cache.load(tmp_path, "thing", "k", max_age=3600) is None and cache.load(tmp_path, "thing", "k") == 1


def test_a_damaged_file_is_just_not_used(tmp_path):
    cache.save(tmp_path, "thing", "k", 1)
    (tmp_path / cache.SAVED / "thing.json").write_text("{not json", encoding="utf-8")
    assert cache.load(tmp_path, "thing", "k") is None
    cache.save(tmp_path, "thing", "k", 2)  # and can be written over
    assert cache.load(tmp_path, "thing", "k") == 2


def _search(folder, name, days_ago=0.0, videos=()):
    project = Project(folder / name).preview()
    cache.stamp(project)
    (project.clips_dir / "a.m4a").write_bytes(b"x" * 100)
    stamp = time.time() - days_ago * 86400
    os.utime(project.root / cache.MARKER, (stamp, stamp))
    for v in videos:
        (project.root.parent / "clips" / v).write_bytes(b"video")
    return project.root.parent


def test_trim_clears_the_searches_used_least_recently_and_never_the_one_being_run(tmp_path):
    made = [_search(tmp_path, f"s{i}", days_ago=i) for i in range(6)]  # s0 is the newest
    cleared = cache.trim(tmp_path, keep=made[5], searches_kept=3)  # s5, the oldest, is the one being run
    assert sorted(p.name for p in cleared) == ["s3", "s4"]
    assert made[5].exists() and (made[5] / "preview").exists()
    assert all((made[i] / "preview").exists() for i in (0, 1, 2))


def test_trim_clears_searches_not_used_for_a_long_time(tmp_path):
    fresh, stale = _search(tmp_path, "fresh", 1), _search(tmp_path, "stale", days_ago=40)
    assert [p.name for p in cache.trim(tmp_path, searches_kept=5, days=30)] == ["stale"]
    assert fresh.exists() and not stale.exists()


def test_forgetting_a_search_never_takes_a_video_that_was_kept(tmp_path):
    kept = _search(tmp_path, "kept", videos=("v1.mp4",))
    cache.forget(kept)
    assert not (kept / "preview").exists() and (kept / "clips" / "v1.mp4").exists()
    empty = _search(tmp_path, "empty")
    (empty / "meld.log").write_text("log", encoding="utf-8")
    cache.forget(empty)
    assert not empty.exists()


def test_folders_that_are_not_melds_are_left_alone(tmp_path):
    (tmp_path / "holiday photos" / "preview").mkdir(parents=True)  # a folder with a preview in it, but not Meld's
    (tmp_path / "holiday photos" / "preview" / "x.jpg").write_bytes(b"x")
    _search(tmp_path, "ours", days_ago=100)
    assert [p.name for p in cache.trim(tmp_path, days=30)] == ["ours"]
    assert (tmp_path / "holiday photos" / "preview" / "x.jpg").exists()


def test_the_command_line_remembers_by_default_and_can_be_told_not_to(tmp_path, monkeypatch):
    from meld import cli

    seen = []
    monkeypatch.setattr(sync_mod, "sync_project", lambda *a, **k: seen.append(k))
    cli.main(["sync", "-p", str(tmp_path)])
    cli.main(["sync", "-p", str(tmp_path), "--no-cache"])
    assert [k["remember"] for k in seen] == [True, False]


# ---- the search

def _fake_youtube(monkeypatch, n=6):
    from meld import fetch as fetch_mod

    state = {"searches": 0, "downloads": []}
    found = [{"id": f"{i:011d}", "title": f"Band Live {i}", "uploader": "", "duration": 100,
              "url": f"https://www.youtube.com/watch?v={i:011d}"} for i in range(n)]

    def search(query, *a, **k):
        state["searches"] += 1
        return list(found), []

    def download(opts, url):
        state["downloads"].append(url)
        i = int(url.rsplit("=", 1)[1])
        (Path(opts["outtmpl"]).parent / f"{i:011d}.m4a").write_bytes(b"x")
        return {"id": f"{i:011d}", "title": f"Band Live {i}", "uploader": "", "webpage_url": url, "duration": 100}

    monkeypatch.setattr(fetch_mod, "search", search)
    monkeypatch.setattr(fetch_mod, "_download", download)
    return state


def _fetch(project, remember=True, **kw):
    from meld.fetch import fetch

    lines = []
    found = fetch(project, [], ["band live"], match_query="band live", log=lines.append, max_clips=None,
                  audio_only=True, remember=remember, **kw)
    return found, lines


def test_a_search_made_before_is_not_made_again_and_its_videos_are_not_downloaded_again(tmp_path, monkeypatch):
    state = _fake_youtube(monkeypatch)
    project = Project(tmp_path)
    first, _ = _fetch(project)
    assert state["searches"] == 1 and len(state["downloads"]) == 6
    again, lines = _fetch(project)
    assert state["searches"] == 1 and len(state["downloads"]) == 6  # nothing asked of YouTube
    assert again == first and len(again) == 6  # the same videos, in the same order
    assert any("search made before" in line for line in lines)


def test_another_search_is_made_afresh(tmp_path, monkeypatch):
    state = _fake_youtube(monkeypatch)
    project = Project(tmp_path)
    _fetch(project)
    _fetch(project, limit=17)  # not the same limits: not the same search
    assert state["searches"] == 2


def test_a_saved_search_runs_out_after_a_week(tmp_path, monkeypatch):
    from meld import fetch as fetch_mod

    state = _fake_youtube(monkeypatch)
    project = Project(tmp_path)
    _fetch(project)
    monkeypatch.setattr(fetch_mod, "SEARCH_MAX_AGE", -1)  # as if it were long ago
    _fetch(project)
    assert state["searches"] == 2 and len(state["downloads"]) == 6  # searched again; the videos it found were there already


def test_a_search_is_only_saved_when_asked_to(tmp_path, monkeypatch):
    state = _fake_youtube(monkeypatch)
    project = Project(tmp_path)
    _fetch(project, remember=False)
    _fetch(project, remember=False)
    assert state["searches"] == 2 and not (project.root / cache.SAVED).exists()


def test_a_stricter_search_is_not_taken_from_a_looser_one_that_was_saved(tmp_path, monkeypatch):
    state = _fake_youtube(monkeypatch)
    project = Project(tmp_path)
    _fetch(project)
    _fetch(project, all_words=True)  # not the same question: it must be asked again
    assert state["searches"] == 2
    _fetch(project, all_words=True)  # but the same one again is remembered
    assert state["searches"] == 2


def test_the_strict_kind_of_matching_reaches_the_filter(tmp_path, monkeypatch):
    from meld import fetch as fetch_mod

    seen = []
    monkeypatch.setattr(fetch_mod, "search", lambda q, limit, lo, hi, require, relevance, log: (seen.append(relevance), ([], []))[1])
    for strict in (False, True):
        fetch_mod.fetch(Project(tmp_path / str(strict)), [], ["coldplay august wembley 2025"], match_query="coldplay august wembley 2025",
                        log=lambda *_: None, all_words=strict)
    assert [r.all_words for r in seen] == [False, True]
    assert seen[1].months == {8} and seen[1].words == ["coldplay", "wembley"]


# ---- sorted again from what is remembered: quietly

def test_a_sort_redone_from_remembered_scores_is_not_a_replay_of_every_comparison(tmp_path, calls):
    project = project_with(tmp_path, TWO_CONCERTS)
    first_lines, again_lines = [], []
    first = sync(project, log=first_lines.append)
    compared = lambda ls: [line for line in ls if " vs " in line and ": z=" in line]
    (project.root / cache.SAVED / "sorted.json").unlink()  # the clips are the same but the sort is asked for again: e.g. a
    made = len(calls)  # new clip turned up and went again
    states = []
    again = sync(project, log=again_lines.append, on_state=states.append)
    assert len(calls) == made  # nothing computed
    assert groups_of(again) == groups_of(first)
    assert 0 < len(compared(again_lines)) < len(compared(first_lines))  # not one line for each of them ...
    assert len(compared(again_lines)) <= len(TWO_CONCERTS)  # ... one for each clip, so that the progress still moves
    assert any("(from before)" in line for line in again_lines)
    assert any("comparisons, " in line and "of them from before" in line for line in again_lines)  # and the summary says so
    assert len(states) < 2 * len(compared(first_lines))  # nor is the window shown every one of them


def test_comparisons_that_are_new_are_all_shown_as_before(tmp_path, calls):
    lines = []
    sync(project_with(tmp_path, TWO_CONCERTS), log=lines.append)
    shown = [line for line in lines if " vs " in line and ": z=" in line]
    assert len(shown) == len(calls) and not any("from before" in line for line in shown)
