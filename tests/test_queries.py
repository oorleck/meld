"""The variations of a search: the words phone footage from an audience describes itself with."""
from meld import fetch as fetch_mod
from meld.fetch import QUERY_SUFFIXES, SEARCH_WORKERS, expand_queries
from meld.project import Project

PHONE_WORDS = [
    "iPhone footage", "filmed on iPhone", "shot on iPhone", "Samsung phone footage", "smartphone footage",
    "audience recording", "handheld footage", "live phone recording", "raw footage", "unedited footage",
]


def test_the_search_is_also_made_the_way_phone_footage_is_described():
    queries = expand_queries("coldplay august wembley 2025")
    assert queries[0] == "coldplay august wembley 2025"  # the plain search comes first: its best hits are kept first
    for words in PHONE_WORDS:
        assert f"coldplay august wembley 2025 {words}" in queries
    for older in ("live", "fan video", "front row", "crowd", "4K", "full song"):
        assert f"coldplay august wembley 2025 {older}" in queries  # and what was there is still there


def test_no_variation_is_asked_for_twice():
    queries = expand_queries("band 2024")
    assert len(queries) == len(set(q.lower() for q in queries)) == 1 + len(QUERY_SUFFIXES)


def test_many_variations_are_not_all_searched_at_once(tmp_path, monkeypatch):
    sizes = []
    real_pool = fetch_mod.ThreadPoolExecutor

    def pool(max_workers=None, **kw):
        sizes.append(max_workers)
        return real_pool(max_workers=max_workers, **kw)

    monkeypatch.setattr(fetch_mod, "ThreadPoolExecutor", pool)
    monkeypatch.setattr(fetch_mod, "search", lambda *a, **k: ([], []))
    queries = expand_queries("band 2024")
    assert len(queries) > SEARCH_WORKERS
    fetch_mod.fetch(Project(tmp_path), [], queries, match_query="band 2024", log=lambda *_: None)
    assert sizes == [SEARCH_WORKERS]  # a few at a time, not one thread per variation
