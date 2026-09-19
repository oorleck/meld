import pytest

from meld.relevance import Relevance, find_dates

QUERY = "coldplay yellow wembley 16 august 2022"


@pytest.mark.parametrize("title", [
    "Coldplay - Yellow - Live at Wembley Stadium London 16/08/2022",
    "COLDPLAY WEMBLEY STADIUM 16TH AUGUST 2022 YELLOW",
    "Coldplay Wembley 16th Aug 2022 Yellow",
    "Yellow Live | Wembley Stadium, London | Coldplay | 16 August",
    "coldplay wembley 16-8-2022. #yellow #coldplay #London",
    "Coldplay Yellow Wembley 16.8.22",
    "Coldplay - Yellow @ Wembley Stadium (August 16, 2022)",
    "Coldplay Yellow Wembley Stadium",  # no date in the title: kept
    "Coldplay Yellow Wembley live 2022 (1/2)",  # part 1/2 is not a date
])
def test_relevant_titles_are_kept(title):
    assert Relevance.from_query(QUERY).check(title) is None


@pytest.mark.parametrize("title, reason", [
    ("Yellow - Coldplay Concert @ Wembley 13.08.2022", "different date"),
    ("Coldplay Yellow Wembley 20 Agosto", "different date"),
    ("Coldplay Wembley 21.08.2022 - Yellow FRONT ROW", "different date"),
    ("Coldplay Wembley Stadium 12th August 2022 - Yellow", "different date"),
    ("Coldplay Yellow Wembley 2022-08-21", "different date"),
    ("COLDPLAY - Yellow AMAZING CROWD BEST VIEW Main Stage Wembley August 21,2022", "different date"),
    ("Coldplay, Viva la Vida. #wembleystadium #coldplayconcert", "missing yellow"),
    ("Coldplay - Yellow", "missing wembley"),
    ("Yellow - Wembley 16/08/2022 Chris Martin", "missing coldplay"),
])
def test_irrelevant_titles_are_dropped(title, reason):
    assert Relevance.from_query(QUERY).check(title) == reason


def test_channel_name_can_supply_a_word():
    rel = Relevance.from_query(QUERY)
    assert rel.check("Yellow live Wembley 16 August 2022", uploader="Coldplay Fan Channel") is None


def test_query_without_date_does_not_filter_on_dates():
    rel = Relevance.from_query("coldplay yellow wembley")
    assert rel.check("Coldplay Yellow Wembley 13.08.2022") is None


def test_query_words_and_dates_are_separated():
    rel = Relevance.from_query(QUERY)
    assert rel.words == ["coldplay", "yellow", "wembley"]
    assert len(rel.dates) == 1


def test_numeric_dates_can_be_either_order():
    dates, _ = find_dates("show on 08/12/22")
    assert {(8, 12, 2022), (12, 8, 2022)} == set(dates[0])


def test_require_date_drops_titles_without_a_date():
    rel = Relevance.from_query(QUERY, require_date=True)
    assert rel.check("Coldplay Yellow Wembley Stadium") == "no date in title"
    assert rel.check("Coldplay Yellow Wembley 16/08/22") is None
    assert rel.check("Coldplay Yellow Wembley 13/08/22") == "different date"


YEAR_QUERY = "metallica 2003"


@pytest.mark.parametrize("title", [
    "Metallica - Live in Seattle 2003 Full Concert",
    "Metallica Summer Sanitarium Festival 2003 - Enter Sandman",
    "Metallica live 12/06/2003",
    "Metallica St. Anger tour (2003-2004)",
    "METALLICA - Fuel (Live, June 2003)",
])
def test_year_query_keeps_that_year(title):
    assert Relevance.from_query(YEAR_QUERY).check(title) is None


@pytest.mark.parametrize("title, reason", [
    ("Metallica Live 1989 Moscow", "different year"),
    ("Metallica - Fuel (Live 2008 Rock Am Ring)", "different year"),
    ("Metallica live 12/06/2008", "different year"),
    ("Metallica - Fuel Live", "no year in title"),
    ("Metallica Live 1920x1080 HD", "no year in title"),  # a resolution is not a year
])
def test_year_query_drops_other_or_unstated_years(title, reason):
    assert Relevance.from_query(YEAR_QUERY).check(title) == reason


def test_year_query_can_keep_undated_titles():
    rel = Relevance.from_query(YEAR_QUERY, require_date=False)
    assert rel.check("Metallica - Fuel Live") is None
    assert rel.check("Metallica - Fuel Live 2008") == "different year"


def test_full_date_query_also_rejects_a_different_year_in_the_title():
    rel = Relevance.from_query(QUERY)
    assert rel.check("Coldplay Yellow Wembley 2019") == "different year"
    assert rel.check("Coldplay Yellow Wembley Live 2022") is None


@pytest.mark.parametrize("title", [
    "Metallica -St  Anger Live Rock am Ring 2003",
    "Metallica - The Four Horsemen & Ride The Lightning - Live At The Fillmore (2003)",
    "Metallica Giants stadium 7 8 2003",
    "Metallica: St. Anger (Madrid, Spain - June 22, 2003) (MetOnTour Edit)",
    "Metallica: Sad But True (Konstanz, Germany - August 16, 2003)",
    "Metallica - Damage Inc. [Live in Paris 2003, SBD audio]",
    "METALLICA - Leper Messiah and Damage Inc. - Le Bataclan, Paris 11 June 2003",
])
def test_concerts_are_kept(title):
    assert Relevance.from_query(YEAR_QUERY).check(title) is None


@pytest.mark.parametrize("title, reason", [
    ("AOL 9.0 Optimized Commercial featuring Metallica (2003)", "not a concert (commercial)"),
    ("Metallica video interview 2003 (Part 1).flv", "not a concert (interview)"),
    ("Metallica bassplayer auditions 2003", "not a concert (auditions)"),
    ("[HD] Metallica - The Unnamed Feeling [St. Anger Rehearsals 2003]", "not a concert (rehearsals)"),
    ("Metallica - Jump in the Studio: Truheeo (March 6, 2003)", "not a concert (studio)"),
    ("Metallica: A Day Off in Detroit (July 3, 2003)", "not a concert (day off)"),
    ("Metallica live 2003 guitar lesson", "not a concert (lesson)"),
    ("Old School 2003 Metallica Scene (Master of Puppets)", "no live/concert signal"),
    ("Metallica - Frantic (2003)", "no live/concert signal"),
])
def test_non_concerts_are_dropped(title, reason):
    assert Relevance.from_query(YEAR_QUERY).check(title) == reason


def test_any_video_switches_the_concert_filter_off():
    rel = Relevance.from_query(YEAR_QUERY, concert_only=False)
    assert rel.check("Metallica video interview 2003 (Part 1).flv") is None
    assert rel.check("Metallica interview 2008") == "different year"  # year/word rules still apply


# ---- strict: every word typed, in the title itself

STRICT_QUERY = "coldplay august wembley 2025"


def strict(query=STRICT_QUERY, **kw):
    return Relevance.from_query(query, all_words=True, **kw)


def test_strict_reads_a_month_typed_anywhere_as_a_month_and_not_as_a_word():
    rel = strict()
    assert rel.words == ["coldplay", "wembley"] and rel.months == {8} and rel.years == {2025}


@pytest.mark.parametrize("title", [
    "Coldplay - Yellow - Live at Wembley Stadium 16/08/2025",  # the month is in the date
    "Coldplay Wembley live 16 Aug 2025",
    "Coldplay Wembley live August 2025 - Fix You",
    "COLDPLAY WEMBLEY live 2025-08-16",
])
def test_strict_keeps_a_title_that_says_the_month_and_year(title):
    assert strict().check(title) is None


@pytest.mark.parametrize("title, reason", [
    ("Coldplay Wembley live 2025 Fix You", "no month in title"),
    ("Coldplay Wembley Stadium live 12/06/2025 Yellow", "different month"),
    ("Coldplay Wembley live 20 September 2025 Yellow", "different month"),
    ("Coldplay Wembley live 16 Aug 2024", "different year"),
    ("Coldplay Wembley live August", "no year in title"),
])
def test_strict_drops_a_title_that_does_not_say_the_month_and_year(title, reason):
    assert strict().check(title) == reason


def test_strict_asks_for_the_title_to_have_every_word_not_the_channel_or_the_start_of_a_word():
    rel = strict()
    assert rel.check("Live at Wembley 16/08/2025 - Sky Full of Stars", uploader="Coldplay") == "missing coldplay"
    assert rel.check("Coldplay Wembleystadium live August 2025") == "missing wembley"  # a word, not the start of one
    assert Relevance.from_query(STRICT_QUERY).check("Coldplay Wembleystadium live August 2025") is None  # as it was
    assert rel.check("Coldplay at Wembley live August 2025") is None
    for typed, in_title in (("songs", "song"), ("song", "songs")):  # a plural is the same word, whichever way round
        rel_plural = Relevance.from_query(f"coldplay wembley {typed} 2025", all_words=True)
        assert rel_plural.check(f"Coldplay Wembley {in_title} live 2025") is None
    assert Relevance.from_query("coldplay wembley boss 2025", all_words=True).check("Coldplay Wembley bos live 2025") is not None


def test_strict_asks_for_the_words_that_are_usually_left_out_but_not_for_connectors():
    rel = Relevance.from_query("coldplay at the wembley live 2025", all_words=True)
    assert rel.words == ["coldplay", "wembley", "live"]  # "at" and "the" are never asked for
    assert rel.check("Coldplay Wembley 2025 Fix You concert") == "missing live"
    assert Relevance.from_query("coldplay at the wembley live 2025").words == ["coldplay", "wembley"]  # not so by default


def test_strict_asks_for_the_date_in_the_title_unless_told_otherwise():
    assert Relevance.from_query(QUERY).check("Coldplay Yellow Wembley Stadium") is None  # by default: kept
    assert strict(QUERY).check("Coldplay Yellow Wembley Stadium live") == "no date in title"
    assert strict(QUERY).check("Coldplay Yellow Wembley Stadium live 16/08/2022") is None
    assert strict(QUERY, require_date=False).check("Coldplay Yellow Wembley Stadium live") is None  # asked for explicitly


def test_strict_still_applies_the_rules_that_are_not_about_words():
    rel = strict()
    assert rel.check("Coldplay Wembley 16/08/2025 interview") == "not a concert (interview)"
    assert rel.check("Coldplay Wembley fix you 2025 august") == "no live/concert signal"  # nothing says it is a concert
    assert rel.check("Coldplay Wembley 16/08/2025") is None  # ... but a dated performance does


def test_the_default_is_not_strict():
    rel = Relevance.from_query(STRICT_QUERY)
    assert rel.all_words is False and rel.months == set()
