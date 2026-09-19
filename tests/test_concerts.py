import json

import pytest

from meld import fetch as fetch_mod
from meld.concerts import group_concerts
from meld.fetch import fetch
from meld.project import Project

# Real titles from a search for "oasis 2025": several nights, dates in every format, some titles with no date at all.
CARDIFF = [  # 4 July
    "Oasis 'Slide Away' LIVE Cardiff Friday July 4th 2025 (PRO SHOT ish in 4K Official Soundboard Audio)",
    "Oasis 'Acquiesce' LIVE Cardiff Friday July 4th 2025 (PRO SHOT in 4K)",
    "Oasis - Introduction / Hello (Live in Cardiff 4/7/25)",  # 4/7 could be 7 April; the others settle it
    "OASIS - F*CKIN’ IN THE BUSHES/HELLO - PRINCIPALITY STADIUM - CARDIFF - 04.07.25",
    "Oasis reunited: Cigarettes & Alcohol [Live at Principality Stadium, Cardiff - 04-07-2025]",
]
CARDIFF_NO_DATE = [
    "Rock ’n’ Roll Star – Oasis LIVE Cardiff 2025 | Principality Stadium",
    "Whatever (Part 2) – Oasis LIVE Cardiff 2025 | Principality Stadium",
    "Supersonic  – Oasis LIVE Cardiff 2025 | Principality Stadium",
]
MANCHESTER = [  # 11 July
    "Oasis - Supersonic (Live in Manchester, July 11th 2025)",
    "Oasis - Live Forever (Manchester 11-7-2025)",
    "Oasis Live 25 - Manchester Heaton Park 11 July 2025 - Intro: Hello + Acquiesce",
    "Oasis - Wonderwall (Live in Manchester, July 11th 2025)",
]
PASADENA = [  # 9 July, written 9/7/25 by everyone
    "Slide Away - Oasis - Front Row - Live ‘25 Tour - Rose Bowl Stadium Pasadena - Los Angeles, CA 9/7/25",
    "Wonderwall - Oasis - Front Row - Live ‘25 Tour - Rose Bowl Stadium Pasadena - Los Angeles 9/7/25",
    "Cast No Shadow - Oasis - Front Row - Live ‘25 Tour - Rose Bowl Pasadena- Los Angeles, CA 9/7/25",
]
ELSEWHERE = [  # no date, and nothing that points at one of the concerts above
    "Some Might Say - Oasis live in Santiago de Chile - 2025",
    "Cigarettes & Alcohol - Oasis live in Sao Paulo - 2025",
]
SINGLE_OTHER_NIGHT = ["Oasis Take Stage in Mexico City (September 12, 2025)"]


def videos(*groups):
    return [{"id": f"v{i}", "title": t, "url": f"https://example.com/v{i}"} for i, t in enumerate(sum(groups, []))]


def test_titles_with_the_same_date_are_one_concert_whatever_the_format():
    g = group_concerts(videos(CARDIFF, MANCHESTER, PASADENA, SINGLE_OTHER_NIGHT))
    assert [c.count for c in g.concerts] == [5, 4, 3]  # a lone video from a fourth night is not offered
    cardiff, manchester, pasadena = g.concerts
    assert cardiff.label.startswith("4 Jul 2025") and "Cardiff" in cardiff.label
    assert manchester.label.startswith("11 Jul 2025") and "Manchester" in manchester.label
    assert pasadena.label.startswith("9 Jul 2025") and "Pasadena" in pasadena.label


def test_the_label_is_the_date_and_the_place_words_that_tell_the_concerts_apart():
    # (a word most of ALL the titles have, like the band or, in a small search, a city, tells nothing and is left out)
    g = group_concerts(videos(CARDIFF, CARDIFF_NO_DATE, MANCHESTER, PASADENA, ELSEWHERE, SINGLE_OTHER_NIGHT, MANCHESTER))
    label = g.concerts[0].label
    assert label == "4 Jul 2025 · Cardiff Principality Stadium"  # not the band, the year, "live", a song or a month
    assert group_concerts(videos(CARDIFF)).concerts[0].label.startswith("4 Jul 2025")


def test_an_ambiguous_date_takes_the_reading_other_titles_state_clearly():
    g = group_concerts(videos(CARDIFF))  # "4/7/25" sits with the "July 4th" ones, not in a 7 April group of its own
    assert len(g.concerts) == 1 and g.concerts[0].count == 5


def test_a_title_without_a_date_joins_the_concert_its_place_words_point_at():
    vids = videos(CARDIFF, CARDIFF_NO_DATE, MANCHESTER, PASADENA)
    g = group_concerts(vids)
    cardiff = g.concerts[0]
    assert cardiff.count == len(CARDIFF) + len(CARDIFF_NO_DATE)
    assert {v["title"] for v in cardiff.videos} >= set(CARDIFF_NO_DATE)
    assert g.undecided == []


def test_titles_that_point_nowhere_are_undecided_and_go_with_any_choice():
    vids = videos(CARDIFF, MANCHESTER, PASADENA, ELSEWHERE, SINGLE_OTHER_NIGHT)
    g = group_concerts(vids)
    assert {v["title"] for v in g.undecided} == set(ELSEWHERE)
    picked = g.select(vids, g.concerts[1])  # Manchester
    titles = [v["title"] for v in picked]
    assert titles == [t for t in (v["title"] for v in vids) if t in set(MANCHESTER) | set(ELSEWHERE)]  # original order
    assert not set(titles) & (set(CARDIFF) | set(PASADENA) | set(SINGLE_OTHER_NIGHT))  # other nights are left out


def test_a_date_without_a_year_takes_the_year_the_others_give():
    g = group_concerts(videos(CARDIFF, ["Oasis Cardiff 4th July - Live Forever"]))
    assert len(g.concerts) == 1 and g.concerts[0].count == 6


def test_nothing_to_choose_between_when_titles_state_no_dates():
    g = group_concerts(videos(CARDIFF_NO_DATE, ELSEWHERE))
    assert g.concerts == []


def test_videos_without_a_title_do_not_break_it():
    g = group_concerts([{"url": "https://example.com/x", "title": None}, *videos(CARDIFF)])
    assert len(g.concerts) == 1 and len(g.undecided) == 1


# ---- fetch: asking which concert, and only downloading that one


def _candidates(*groups):
    return [
        {"id": v["id"], "title": v["title"], "uploader": "", "duration": 100, "url": v["url"]} for v in videos(*groups)
    ]


@pytest.fixture
def fake_youtube(monkeypatch):
    """A search that returns whatever a test sets, and downloads that only report their id."""
    state = {"results": []}
    monkeypatch.setattr(fetch_mod, "search", lambda *a, **k: (state["results"], []))

    def fake_download(opts, url):
        vid = url.rsplit("/", 1)[1]
        return {"id": vid, "title": vid, "uploader": "", "webpage_url": url, "duration": 100}

    monkeypatch.setattr(fetch_mod, "_download", fake_download)
    return state


def _downloaded(project) -> set[str]:
    return set(json.loads(project.sources_path.read_text("utf-8")))


def _run(tmp_path, results, fake_youtube, **kw):
    fake_youtube["results"] = results
    project = Project(tmp_path)
    fetch(project, [], ["q"], match_query="q", log=lambda *_: None, max_clips=kw.pop("max_clips", 500), **kw)
    return project


def test_only_the_chosen_concert_and_the_undecided_are_downloaded(tmp_path, fake_youtube):
    results = _candidates(CARDIFF, MANCHESTER, PASADENA, ELSEWHERE)
    asked = []

    def choose(concerts):
        asked.append([c.label for c in concerts])
        return next(c for c in concerts if "Manchester" in c.label)

    project = _run(tmp_path, results, fake_youtube, choose=choose)
    assert len(asked) == 1 and len(asked[0]) == 3
    wanted = {r["id"] for r in results if r["title"] in set(MANCHESTER) | set(ELSEWHERE)}
    assert _downloaded(project) == wanted


def test_the_cap_is_filled_from_the_chosen_concert(tmp_path, fake_youtube):
    results = _candidates(MANCHESTER, CARDIFF, PASADENA)  # the cap would take Manchester's four, then Cardiff's ...
    project = _run(tmp_path, results, fake_youtube, max_clips=3, choose=lambda cs: next(c for c in cs if "Cardiff" in c.label))
    titles = {r["id"]: r["title"] for r in results}
    assert len(_downloaded(project)) == 3 and all(titles[i] in CARDIFF for i in _downloaded(project))


def test_choosing_all_of_them_downloads_everything(tmp_path, fake_youtube):
    results = _candidates(CARDIFF, MANCHESTER)
    project = _run(tmp_path, results, fake_youtube, choose=lambda concerts: None)
    assert _downloaded(project) == {r["id"] for r in results}


def test_no_question_when_there_is_only_one_concert(tmp_path, fake_youtube):
    results = _candidates(CARDIFF, CARDIFF_NO_DATE)

    def choose(concerts):
        raise AssertionError("should not be asked")

    project = _run(tmp_path, results, fake_youtube, choose=choose)
    assert _downloaded(project) == {r["id"] for r in results}


def test_cancelling_the_question_stops_before_anything_is_downloaded(tmp_path, fake_youtube):
    class Stop(Exception):
        pass

    def choose(concerts):
        raise Stop()

    with pytest.raises(Stop):
        _run(tmp_path, _candidates(CARDIFF, MANCHESTER), fake_youtube, choose=choose)
    assert not (tmp_path / "sources.json").exists()


def test_without_a_chooser_everything_is_downloaded_as_before(tmp_path, fake_youtube):
    results = _candidates(CARDIFF, MANCHESTER, PASADENA)
    project = _run(tmp_path, results, fake_youtube)
    assert _downloaded(project) == {r["id"] for r in results}


def test_a_dry_run_lists_the_concerts_but_never_asks(tmp_path, fake_youtube):
    fake_youtube["results"] = _candidates(CARDIFF, MANCHESTER)
    lines = []

    def choose(concerts):
        raise AssertionError("a dry run must not ask")

    fetch(Project(tmp_path), [], ["q"], match_query="q", dry_run=True, choose=choose, log=lines.append)
    assert any("2 different concerts" in line for line in lines)
