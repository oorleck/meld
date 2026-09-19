"""Tell apart the concerts in a batch of search results, so the user can pick one.

A search for "oasis 2025" finds clips of many different nights. Titles usually say which night, mostly as a date
(4/7/25, July 4th 2025, 04.07.25 ...), which relevance.find_dates already reads in many formats. So:

- videos whose titles state the same date are one concert;
- a video that states no date joins a concert when its other words (venue, city) point clearly at that one;
- what is left could be from any of them and is tried with whichever concert is picked. Sync rejects it if the
  audio does not match, exactly as before.

When the titles give nothing to tell concerts apart, there is only one group (or none) and the user is not asked.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .naming import content_keys, shared_words, trim_connectors
from .naming import _key as word_key
from .relevance import _MONTH_NUM, find_dates

MIN_VIDEOS = 2  # fewer videos than this is not offered as a concert of its own
MAX_PLACE_WORDS = 6
_PLACE_SHARE = 0.5  # a word belongs to a concert's place when this share of the dated titles that use it are its own
_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
_DATE_WORDS = set(_MONTH_NUM) | {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
_ASSOC_MIN = 2  # a word points at a concert when it is in at least this many of its dated titles ...
_ASSOC_SHARE = 0.8  # ... and at least this share of all dated titles that have it
_TOO_COMMON = 0.5  # words in more than this share of all titles (the band, the year) tell nothing

Key = tuple  # (year | None, month, day)


@dataclass
class Concert:
    label: str  # e.g. "4 Jul 2025 · Cardiff Principality Stadium"
    key: Key
    videos: list[dict] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.videos)


@dataclass
class Grouping:
    concerts: list[Concert]  # biggest first
    undecided: list[dict]  # could be from any of the concerts: no date, and nothing pointing at one

    def select(self, videos: list[dict], concert: Concert) -> list[dict]:
        """`videos` (in their original order) that belong to `concert` or could: the rest are other concerts."""
        keep = {id(v) for v in concert.videos} | {id(v) for v in self.undecided}
        return [v for v in videos if id(v) in keep]


def _readings(title: str) -> list[tuple]:
    """The (day, month, year|None) readings of the first date a title states; several when 4/7 could be either order."""
    dates, _ = find_dates(title)
    return dates[0] if dates else []


def _date_keys(readings: list[list[tuple]]) -> list[Key | None]:
    """One (year, month, day) per title, or None for titles that state no date."""
    sure = Counter((y, m, d) for r in readings if len(r) == 1 for d, m, y in r)

    def support(key: Key) -> int:  # how many titles state this very date without any doubt
        return sum(n for (y, m, d), n in sure.items() if (m, d) == key[1:] and (y == key[0] or None in (y, key[0])))

    keys: list[Key | None] = []
    for r in readings:
        if not r:
            keys.append(None)
            continue
        cands = [(y, m, d) for d, m, y in r]
        # 4/7/25 is 4 July or 7 April: go with the reading other titles state unambiguously, else day-first
        keys.append(max(cands, key=lambda k: (support(k), -cands.index(k))))

    years: dict[tuple, set] = {}
    for k in keys:
        if k and k[0]:
            years.setdefault(k[1:], set()).add(k[0])
    # a date without a year takes the year other titles give that same day, when there is only one
    return [
        (next(iter(years[k[1:]])), k[1], k[2]) if k and k[0] is None and len(years.get(k[1:], ())) == 1 else k
        for k in keys
    ]


def _date_text(key: Key) -> str:
    year, month, day = key
    return f"{day} {_MONTHS[month - 1]}" + (f" {year}" if year else "")


def group_concerts(videos: list[dict]) -> Grouping:
    """Split search results (dicts with a "title") into concerts. Videos without a title are undecided."""
    titles = [v.get("title") or "" for v in videos]
    keys = _date_keys([_readings(t) for t in titles])
    by_key: dict[Key, list[int]] = {}
    for i, k in enumerate(keys):
        if k:
            by_key.setdefault(k, []).append(i)
    nights = {k: idx for k, idx in by_key.items() if len(idx) >= MIN_VIDEOS}

    words = [content_keys(t) for t in titles]
    total = Counter(w for ws in words for w in ws)
    dated = Counter(w for i, k in enumerate(keys) if k for w in words[i])
    assoc: dict[Key, set[str]] = {}
    here_counts: dict[Key, Counter] = {}
    for k, idx in nights.items():
        here = here_counts[k] = Counter(w for i in idx for w in words[i])
        assoc[k] = {
            w for w, n in here.items()
            if n >= _ASSOC_MIN and n / dated[w] >= _ASSOC_SHARE and total[w] / len(videos) <= _TOO_COMMON
        }

    members = {k: list(idx) for k, idx in nights.items()}
    undecided: list[dict] = []
    for i, k in enumerate(keys):
        if k:
            continue
        hits = {c: len(words[i] & a) for c, a in assoc.items()}
        best = max(hits.values(), default=0)
        winners = [c for c, n in hits.items() if n == best]
        if best and len(winners) == 1:
            members[winners[0]].append(i)
        else:
            undecided.append(videos[i])

    common = {w.casefold() for w in shared_words(titles)}  # the band, the year: not what tells the concerts apart
    concerts = []
    for k, idx in members.items():
        # what is left of the words this concert's titles share: its place (the date is shown separately)
        own = [
            w for w in shared_words([titles[i] for i in idx])
            if w.casefold() not in common and word_key(w) not in _DATE_WORDS and not any(c.isdigit() for c in w)
        ]
        # A word other concerts' titles use too (a song: "Hello") is not part of the place. Keep the words that mostly
        # belong to this concert and those right beside one ("Stadium" in "Principality Stadium").
        mine = [dated[word_key(w)] > 0 and here_counts[k][word_key(w)] / dated[word_key(w)] >= _PLACE_SHARE for w in own]
        if any(mine):
            own = [
                w for i, w in enumerate(own)
                if mine[i] or (i > 0 and mine[i - 1]) or (i + 1 < len(own) and mine[i + 1])
            ]
        place = " ".join(trim_connectors(own)[:MAX_PLACE_WORDS])
        concerts.append(Concert(_date_text(k) + (f" · {place}" if place else ""), k, [videos[i] for i in idx]))
    concerts.sort(key=lambda c: (-c.count, c.key[0] or 0, c.key[1], c.key[2]))
    return Grouping(concerts, undecided)
