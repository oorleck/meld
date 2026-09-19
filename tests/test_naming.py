import pytest

from meld.naming import MAX_LEN, concert_name, safe_filename, shared_words, trim_connectors, unique_stem


def test_name_is_what_the_titles_have_in_common():
    titles = [
        "Coldplay - Yellow (Live at Wembley Stadium 2022) 4K",
        "Coldplay Wembley Stadium 16/08/2022 Viva La Vida front row",
        "COLDPLAY - Fix You LIVE Wembley Stadium London 2022",
        "Coldplay live at Wembley Stadium - Clocks (fan video)",
    ]
    # songs, the uploader's tags (4K, front row, fan video) and words only some titles use are left out
    assert concert_name(titles, "coldplay wembley 16 august 2022") == "Coldplay at Wembley Stadium 2022"


def test_accents_and_casing_do_not_split_a_word():
    titles = [
        "håkan hellström - kärlek är ett sätt att vara (live ullevi 2025)",
        "Hakan Hellstrom Ullevi 2025 Känn ingen sorg",
        "HÅKAN HELLSTRÖM LIVE @ ULLEVI, GÖTEBORG 2025",
        "Håkan Hellström - Ullevi 2025 - För sent för Edelweiss",
    ]
    assert concert_name(titles) == "Håkan Hellström Ullevi 2025"


def test_words_come_in_the_order_of_the_most_representative_title():
    titles = [
        "The Cure - Boys Don't Cry (Live in Paris 2016)",
        "The Cure Paris 2016 Lovesong",
        "The Cure - Live Paris 2016 - Friday I'm in Love",
    ]
    assert concert_name(titles) == "The Cure in Paris 2016"
    # a multi-word place stays in one piece however the other titles order their words
    places = [
        "Live at Wembley Stadium, London - 25-07-2025",
        "London Wembley Stadium 25/07/2025 Hello",
        "Hello - Wembley Stadium London 25 July 2025",
    ]
    words = shared_words(places)
    assert {"Wembley", "Stadium", "London"} <= set(words)
    assert words[words.index("Wembley") + 1] == "Stadium"


def test_trim_connectors_drops_small_words_at_the_ends_only():
    assert trim_connectors(["Coldplay", "at"]) == ["Coldplay"]
    assert trim_connectors(["in", "Paris"]) == ["Paris"]
    assert trim_connectors(["The", "Cure"]) == ["The", "Cure"]  # a leading 'The' is part of the name
    assert trim_connectors(["Queens", "of", "the", "Stone", "Age"]) == ["Queens", "of", "the", "Stone", "Age"]
    assert trim_connectors(["at", "the"]) == []


def test_one_title_is_cleaned_up():
    assert concert_name(["Metallica - Fuel (Live 2003, Madrid) HD"]) == "Metallica Fuel 2003 Madrid"


@pytest.mark.parametrize("titles, fallback, expected", [
    (["Song A", "Other thing entirely"], "some band 2001", "Some Band 2001"),  # nothing in common
    (["Foo 2022", "Bar 2022"], "the query 2022", "The Query 2022"),  # only a year in common is not a name
    ([], "Metallica 2003", "Metallica 2003"),
    ([], "", "Concert"),
])
def test_falls_back_to_what_the_user_typed(titles, fallback, expected):
    assert concert_name(titles, fallback) == expected


def test_name_is_short_and_safe_for_a_file_name():
    titles = ["Band " + " ".join(f"word{i}" for i in range(40))] * 3
    name = concert_name(titles)
    assert 0 < len(name) <= MAX_LEN and len(name.split()) <= 8
    assert safe_filename('a<b>:"c/d\\e|f?g*h.  ') == "abcdefgh"
    assert safe_filename("CON") == "CON_"  # a reserved device name on Windows
    assert len(safe_filename("x" * 200)) == MAX_LEN


def test_unique_stem_never_reuses_a_name(tmp_path):
    assert unique_stem(tmp_path, "Show") == "Show"
    (tmp_path / "Show.wav").write_bytes(b"")  # either file of the pair being taken is enough
    assert unique_stem(tmp_path, "Show") == "Show (2)"
    (tmp_path / "Show (2).mp4").write_bytes(b"")
    assert unique_stem(tmp_path, "Show") == "Show (3)"
