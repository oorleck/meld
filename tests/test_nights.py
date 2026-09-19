"""Keeping nights apart by the days the titles name: a show that plays to backing tracks sounds the same every night."""
import json

from synth import phone, synth_music, write_wav
from test_groups import names

from meld import gui
from meld import sync as sync_mod
from meld.project import Project
from meld.relevance import dates_conflict, title_dates
from meld.sync import DATE_OVERRIDE_Z, sync_project

SR = 16000
SHOW = synth_music(300, SR, seed=31)  # the same show, whichever night: its sound is one
NOISE = 0.3  # phones that match as real ones do (z about 65; a phone as clean as a copy of the recording scores 400)


def night(tmp_path, clips: dict, remember=False, **kw):
    """clips: name -> (from, to, title). Every phone records the same show."""
    project = Project(tmp_path)
    titles = {}
    for i, (name, (t0, t1, title)) in enumerate(clips.items()):
        write_wav(project.clips_dir / f"{name}.wav", phone(SHOW, SR, t0, t1, seed=i + 1, noise=NOISE), SR)
        titles[name] = {"title": title}
    project.sources_path.write_text(json.dumps(titles), encoding="utf-8")
    lines = []
    tl = sync_project(project, log=lines.append, workers=1, remember=remember, **kw)
    return tl, lines


def groups_of(tl):
    return sorted(([sorted(names(tl.clips))] + [sorted(names(g)) for g in tl.others]), key=lambda g: g[0])


TWO_NIGHTS = {
    "a1": (0, 100, "Taylor Swift - So Long, London - 15/08/2024"),
    "a2": (20, 118, "Taylor Swift - So Long, London | August 15th 2024"),
    "b1": (5, 95, "Taylor Swift - So Long, London - 20/08/2024"),
    "b2": (25, 116, "So Long London Wembley 20th August 2024"),
}


# ---- what a title names

def test_the_days_a_title_names_are_read_in_the_ways_they_are_written_including_with_bars():
    for text, day in (("21|6|2024 - Ready for It?", (21, 6, 2024)), ("Wembley 15/08/2024", (15, 8, 2024)),
                      ("August 19, 2024", (19, 8, 2024)), ("20th August 2024", (20, 8, 2024)), ("So Long London 23.6.24", (23, 6, 2024)),
                      ("2024-08-19 Wembley", (19, 8, 2024))):
        assert day in [reading for named in title_dates(text) for reading in named], text
    assert title_dates("Taylor Swift - 22 (The Eras Tour London 2024)") == []  # a year, or a month, is not a day
    assert title_dates("Eras Tour | June 2024") == []


def test_titles_name_different_days_only_if_both_name_some_and_none_is_the_same():
    a, b, c = title_dates("15/08/2024"), title_dates("Aug 15th 2024"), title_dates("21|6|2024")
    assert not dates_conflict(a, b) and dates_conflict(a, c)  # the same day in two forms; two days
    assert not dates_conflict(a, []) and not dates_conflict([], []) and not dates_conflict(a, title_dates("no day here"))
    assert not dates_conflict(a, title_dates("15/08 and 16/08 compilation"))  # a title that names several days shares one


# ---- sorting

def test_the_same_show_on_two_nights_is_one_group_by_sound_and_two_by_the_days_in_the_titles(tmp_path):
    together, _ = night(tmp_path / "audio", TWO_NIGHTS)
    assert groups_of(together) == [["a1", "a2", "b1", "b2"]]  # by sound alone they are one: it is the same show

    apart, lines = night(tmp_path / "titles", TWO_NIGHTS, nights=True)
    assert groups_of(apart) == [["a1", "a2"], ["b1", "b2"]]
    assert any("4 of 4 clip(s) name a day in their title" in line for line in lines)


def test_a_clip_that_names_no_day_joins_a_night_and_does_not_join_two(tmp_path):
    clips = {**TWO_NIGHTS, "x": (10, 90, "Taylor Swift Live So Long, London Wembley Stadium 2024")}
    tl, _ = night(tmp_path, clips, nights=True)
    groups = groups_of(tl)
    assert len(groups) == 2 and sum("x" in g for g in groups) == 1  # it joined one of the nights ...
    assert [g for g in groups if "x" not in g][0] in (["a1", "a2"], ["b1", "b2"])  # ... and the other is whole


def test_the_days_are_only_used_when_asked_for(tmp_path):
    tl, lines = night(tmp_path, TWO_NIGHTS, nights=False)
    assert groups_of(tl) == [["a1", "a2", "b1", "b2"]] and not any("name a day" in line for line in lines)


def test_the_very_same_recording_under_another_day_is_still_the_same_recording(tmp_path):
    # an upload again, with another date in its title: identical sound, so it matches far better than phones ever do
    project = Project(tmp_path)
    recording = phone(SHOW, SR, 0, 100, seed=1)
    for name in ("first", "again"):
        write_wav(project.clips_dir / f"{name}.wav", recording, SR)
    write_wav(project.clips_dir / "other.wav", phone(SHOW, SR, 10, 105, seed=7, noise=NOISE), SR)
    project.sources_path.write_text(json.dumps({
        "first": {"title": "So Long London 15/08/2024"}, "again": {"title": "So Long London 20/08/2024"},
        "other": {"title": "So Long London 15/08/2024"},
    }), encoding="utf-8")
    tl = sync_project(project, log=lambda *_: None, workers=1, nights=True)
    assert groups_of(tl) == [["again", "first", "other"]] and DATE_OVERRIDE_Z >= 100  # a copy is a copy, whatever its title says


def test_what_is_saved_of_a_sorting_goes_with_the_days_the_titles_name(tmp_path):
    first, _ = night(tmp_path, TWO_NIGHTS, remember=True, nights=True)
    project = Project(tmp_path)
    same, lines = None, []
    same = sync_project(project, log=lines.append, workers=1, remember=True, nights=True)
    assert any("before: using that" in line for line in lines) and groups_of(same) == groups_of(first)

    titles = json.loads(project.sources_path.read_text(encoding="utf-8"))
    titles["b1"]["title"] = "Taylor Swift - So Long, London - 15/08/2024"  # a title that names another day
    project.sources_path.write_text(json.dumps(titles), encoding="utf-8")
    lines.clear()
    changed = sync_project(project, log=lines.append, workers=1, remember=True, nights=True)
    assert not any("before: using that" in line for line in lines)  # the sorting was not taken as it was
    assert ["a1", "a2", "b1"] in groups_of(changed) or any({"a1", "b1"} <= set(g) for g in groups_of(changed))


# ---- what the person is shown

def test_a_group_is_labelled_with_the_night_its_titles_name():
    assert gui.group_night(["Taylor Swift - So Long, London - 20/08/2024", "So Long London August 20th 2024"]) == "20 Aug 2024"
    assert gui.group_night(["So Long London 15/08/2024", "So Long London 20/08/2024"]) == ""  # nights mixed: none named
    assert gui.group_night(["So Long London", "Eras Tour 2024"]) == ""
    assert sync_mod.DATE_OVERRIDE_Z > sync_mod.IDENTICAL_Z
