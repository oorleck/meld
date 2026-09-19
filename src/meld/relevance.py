"""Decide whether a search result is really about the concert the user asked for.

Rules, applied to the video title (and channel name for words):
- every meaningful word of the query must appear (prefix match, accent/case-insensitive);
- if the query names a date (16 august 2022), a title that names a *different* date is rejected.
  Titles that name no date are kept; the date is often only in the description.

`all_words` makes it strict: every word typed (bar connectors such as "the" and "at") must be a whole word of the title
itself, not of the channel name, and the date, month and year typed must be in the title too. Fewer videos, but each one
says outright that it is what was asked for.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_MONTHS = {
    1: "january januar janvier gennaio enero janeiro januari jan janv ene gen",
    2: "february februar fevrier febbraio febrero fevereiro februari feb fev febr",
    3: "march marz mars marzo marco maart mar mrz",
    4: "april avril aprile abril apr avr abr",
    5: "may mai maggio mayo maio mei mag",
    6: "june juni juin giugno junio junho jun giu",
    7: "july juli juillet luglio julio julho jul juil lug",
    8: "august augustus aout agosto ago aug",
    9: "september sept septembre settembre septiembre setiembre setembro sep set",
    10: "october oktober octobre ottobre octubre outubro oct okt ott out",
    11: "november novembre noviembre novembro nov",
    12: "december dezember decembre dicembre diciembre dezembro dec dez dic",
}
_MONTH_NUM = {w: m for m, names in _MONTHS.items() for w in names.split()}
# English months as typed in a query: a full name or an unmistakable abbreviation ("mar", "set" and "out" are also
# months in other languages, but as words in a query they are far more likely to be just words)
_ENGLISH_MONTHS = set(_MONTHS[m].split()[0] for m in _MONTHS) | {
    "jan", "feb", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_MONTH_RE = "|".join(sorted(_MONTH_NUM, key=len, reverse=True))

CONNECTORS = {"the", "a", "an", "at", "in", "of", "on", "and", "de", "la", "el", "los", "en"}  # never worth requiring
STOP = CONNECTORS | {"live", "concert", "show", "tour", "video", "fan", "hd", "full", "song"}  # ... nor, usually, these

_ISO = re.compile(r"(?<![\d./|-])(\d{4})[./|-](\d{1,2})[./|-](\d{1,2})(?!\d)")  # (a | as well: 21|6|2024 is a date in titles)
_NUM = re.compile(r"(?<![\d./|-])(\d{1,2})[./|-](\d{1,2})(?:[./|-](\d{4}|\d{2}))?(?![\d])")
_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th|er|o)?\.?\s*(?:of\s+|de\s+|del\s+)?({_MONTH_RE})\b\.?(?:\s*,?\s*(\d{{4}}))?"
)
_MDY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s*,?\s*(\d{{4}}))?")
_MONTH_YEAR = re.compile(rf"\b({_MONTH_RE})\.?\s+\d{{4}}\b")
# not part of a date (16/08/2022 is read by the date parser) and not a resolution like 1920x1080
_YEAR = re.compile(r"(?<![\dx./-])(?:19|20)\d\d(?![\dxp])")

_NEGATIVE = re.compile(
    r"\b(?:interviews?|commercials?|adverts?|advertisements?|trailers?|teasers?|auditions?|rehears\w*|soundchecks?|"
    r"studios?|day off|behind the scenes|documentar\w*|reactions?|react|tutorials?|lessons?|how to|covers?|karaoke|"
    r"lyrics?|official video|music video|making of|podcast|reviews?|unboxing|vlog|press conference|parody|remix|"
    r"tribute|talking about|jam session|backing track)\b"
)
_POSITIVE = re.compile(
    r"\b(?:live|concert\w*|konzert|concierto|festival|fest|tour|gig|show|stadium|stade|estadio|arena|amphitheat\w*|"
    r"theat(?:re|er)|teatro|gardens?|coliseum|colosseum|pavilion|ballroom|rock (?:am|im|in) \w+|fancam|fan cam|pov|"
    r"front row|soundboard|sbd|audience|bootleg|pro-?shot|setlist)\b"
)

# a date is a list of candidate (day, month, year|None) readings, because 08/12 can be either order
Date = list[tuple[int, int, "int | None"]]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _year(s: str | None) -> int | None:
    if not s:
        return None
    y = int(s)
    return y + 2000 if y < 100 else y


def _ok(d: int, m: int) -> bool:
    return 1 <= d <= 31 and 1 <= m <= 12


def find_dates(text: str, months: "set[int] | None" = None) -> tuple[list[Date], str]:
    """Dates mentioned in `text`, and the text with the date parts blanked out. If `months` is given, the months that
    are named without a day ("august 2025") are added to it, as numbers."""
    t = _norm(text)
    dates: list[Date] = []

    def blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())

    def sub(pattern: re.Pattern, reader) -> None:
        nonlocal t
        for m in pattern.finditer(t):
            cands = [c for c in reader(m) if _ok(c[0], c[1])]
            if cands:
                dates.append(cands)
        t = pattern.sub(lambda m: blank(m) if _read_ok(reader, m) else m.group(0), t)

    def _read_ok(reader, m) -> bool:
        return any(_ok(c[0], c[1]) for c in reader(m))

    sub(_ISO, lambda m: [(int(m[3]), int(m[2]), int(m[1]))])

    def num(m):
        a, b, y = int(m[1]), int(m[2]), _year(m[3])
        if y is None and max(a, b) <= 12:
            return []  # 1/2 without a year: could be "part 1/2", not a date
        return [(a, b, y), (b, a, y)]

    sub(_NUM, num)
    sub(_DMY, lambda m: [(int(m[1]), _MONTH_NUM[m[2]], _year(m[3]))])
    sub(_MDY, lambda m: [(int(m[2]), _MONTH_NUM[m[1]], _year(m[3]))])
    if months is not None:
        months.update(_MONTH_NUM[m[1]] for m in _MONTH_YEAR.finditer(t))
    t = _MONTH_YEAR.sub(lambda m: blank(m), t)
    return dates, t


def find_years(text: str) -> set[int]:
    """Every year a text states: 4-digit years, and the year inside any date it contains."""
    dates, _ = find_dates(text)
    years = {y for cands in dates for _, _, y in cands if y}
    return years | {int(m.group()) for m in _YEAR.finditer(_norm(text))}


def _same_day(a: Date, b: Date) -> bool:
    return any(d1 == d2 and m1 == m2 and (y1 is None or y2 is None or y1 == y2) for d1, m1, y1 in a for d2, m2, y2 in b)


def title_dates(title: str) -> list[Date]:
    """The days a title names, each as its list of readings (see `find_dates`); [] if it names none. A month or a year
    alone is not a day."""
    return find_dates(title)[0]


def dates_conflict(a: list[Date], b: list[Date]) -> bool:
    """Do the days two titles name differ? Only if both name some, and no day of the one is a day of the other."""
    return bool(a and b) and not any(_same_day(x, y) for x in a for y in b)


@dataclass
class Relevance:
    words: list[str] = field(default_factory=list)
    dates: list[Date] = field(default_factory=list)
    years: set[int] = field(default_factory=set)
    # Drop titles that state no date/year at all? None = decide by how specific the query is:
    # yes for a bare year ("metallica 2003"), no for a full date, where the title often omits it.
    require_date: bool | None = None
    concert_only: bool = True  # drop interviews, commercials, rehearsals, covers, ... and titles with no live signal
    months: set[int] = field(default_factory=set)  # months named without a day ("august 2025"); only checked when strict
    all_words: bool = False  # strict: see the top of this file

    @classmethod
    def from_query(
        cls, query: str, require_date: bool | None = None, concert_only: bool = True, all_words: bool = False,
    ) -> "Relevance":
        months: set[int] = set()
        dates, rest = find_dates(query, months)
        skip = CONNECTORS if all_words else STOP  # strict: "live" and "show" were typed, so they are asked for
        words = [
            w for w in re.findall(r"[a-z0-9]+", rest)
            if w not in skip and not re.fullmatch(r"(19|20)\d\d", w)
        ]
        if all_words:
            # a month typed anywhere ("coldplay august wembley 2025") is a month the title must say, in words or in a
            # date (16/08/2025), not a word it must contain
            months |= {_MONTH_NUM[w] for w in words if w in _ENGLISH_MONTHS}
            words = [w for w in words if w not in _ENGLISH_MONTHS]
            if require_date is None:
                require_date = True  # the date and year typed are part of what was asked for
        return cls(words, dates, find_years(query), require_date, concert_only, months if all_words else set(), all_words)

    def check(self, title: str, uploader: str = "") -> str | None:
        """None if the result looks relevant, otherwise the reason it was rejected."""
        if self.all_words:  # whole words, of the title alone (a plural is the same word, whichever was typed)
            hay = _norm(title)
            stems = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in self.words]
            missing = [
                w for w, stem in zip(self.words, stems)
                if not re.search(r"(?<![a-z0-9])" + re.escape(stem) + r"(?:e?s)?(?![a-z0-9])", hay)
            ]
        else:
            hay = _norm(f"{title} {uploader}")
            missing = [w for w in self.words if not re.search(r"(?<![a-z0-9])" + re.escape(w), hay)]
        if missing:
            return "missing " + ", ".join(missing)

        need = self.require_date if self.require_date is not None else (bool(self.years) and not self.dates)
        title_dates, _ = find_dates(title)
        title_years = find_years(title)
        if self.dates:
            if title_dates and not any(_same_day(td, qd) for td in title_dates for qd in self.dates):
                return "different date"
            if not title_dates and need:
                return "no date in title"
        if self.years:
            if title_years and not (title_years & self.years):
                return "different year"
            if not title_years and need:
                return "no year in title"
        if self.months and not self.dates:  # strict, and a month was typed without a day: the title must say that month
            if title_dates:
                if not any(m in self.months for cands in title_dates for _, m, _ in cands):
                    return "different month"
            elif not any(_MONTH_NUM.get(w) in self.months for w in re.findall(r"[a-z]+", _norm(title))):
                return "no month in title"
        if self.concert_only:
            t = _norm(title)
            bad = _NEGATIVE.search(t)
            if bad:
                return f"not a concert ({bad.group()})"
            if not _POSITIVE.search(t) and not title_dates:  # a dated performance is a concert signal too
                return "no live/concert signal"
        return None
