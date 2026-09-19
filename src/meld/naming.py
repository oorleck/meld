"""A short, fitting file name for a fused concert, taken from the titles of the videos it was made from.

Titles of clips of one concert share the band, venue and date and differ in everything else (song, uploader tags),
so the name is made of the words that most of the titles have in common, in the order they usually come.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .relevance import STOP

MAX_LEN = 60
MAX_WORDS = 8

# Small words that can be part of a name ("Queens of the Stone Age", "Coldplay at Wembley").
_CONNECTORS = {"the", "a", "an", "at", "in", "of", "on", "and", "de", "la", "el", "los", "en"}
# Words that say what kind of upload it is, not which concert. The search adds several of them to its queries itself.
_NOISE = (STOP - _CONNECTORS) | {
    "front", "row", "crowd", "4k", "8k", "hq", "uhd", "1080p", "720p", "2160p", "60fps", "official", "footage",
    "pov", "fancam", "cam", "audio", "clip", "part",
}

# a date such as 16/08/2022 or 2022-08-16 stays one word; otherwise a word, with an apostrophe inside it kept
_WORD = re.compile(r"\d{1,4}(?:[./-]\d{1,4}){1,2}|[^\W_]+(?:['’][^\W_]+)*")
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _key(word: str) -> str:
    """Case- and accent-insensitive form, so 'Hellström' and 'hellstrom' count as the same word."""
    return "".join(c for c in unicodedata.normalize("NFKD", word.casefold()) if not unicodedata.combining(c))


def _words(text: str) -> list[str]:
    return [re.sub(r"[./]", "-", m.group()) for m in _WORD.finditer(text)]  # 16/08/2022 -> 16-08-2022, valid in a file name


def _pretty(word: str) -> str:
    """SHOUTING and all-lowercase words become Capitalised; mixed case (deadmau5, McCartney) is left alone."""
    return word.capitalize() if (word.isupper() and len(word) > 4) or word.islower() else word


def content_keys(title: str) -> set[str]:
    """The words of a title that can say which concert or song it is: no noise words, connectors or numbers."""
    return {
        k for w in _words(title)
        if (k := _key(w)) not in _NOISE and k not in _CONNECTORS and len(k) >= 3 and not any(c.isdigit() for c in k)
    }


def safe_filename(text: str, max_len: int = MAX_LEN) -> str:
    """Drop what Windows does not allow in a file name and keep it short."""
    text = re.sub(r"\s+", " ", _BAD_CHARS.sub("", text)).strip(" .")
    text = text[:max_len].rstrip(" .")
    return text + "_" if text.upper() in _RESERVED else text


def shared_words(titles: list[str]) -> list[str]:
    """The words most of `titles` have in common, in the order they usually come. Empty when there are none."""
    per_title: list[dict[str, str]] = []  # key -> spelling used, in order of appearance
    for title in titles:
        seen: dict[str, str] = {}
        for w in _words(title):
            k = _key(w)
            if k not in _NOISE and k not in seen:
                seen[k] = w
        if seen:
            per_title.append(seen)

    chosen: list[str] = []
    if per_title:
        need = 1 if len(per_title) == 1 else max(2, math.ceil(len(per_title) / 2))
        docs = Counter(k for seen in per_title for k in seen)
        pos_sum: Counter = Counter()
        spelling: dict[str, Counter] = {}
        first_seen: dict[str, int] = {}
        for seen in per_title:
            for i, (k, w) in enumerate(seen.items()):
                pos_sum[k] += i
                spelling.setdefault(k, Counter())[w] += 1
                first_seen.setdefault(k, len(first_seen))
        keys = [k for k, n in docs.items() if n >= need]
        # Order the words as they come in the title that has the most of them, so 'Wembley Stadium, London' stays
        # together; the few it lacks follow, by where they usually come.
        best = max(per_title, key=lambda seen: (sum(k in seen for k in keys), -len(seen)))
        rank = {k: i for i, k in enumerate(best)}
        keys.sort(key=lambda k: (rank.get(k, len(rank)), pos_sum[k] / docs[k], first_seen[k]))
        chosen = trim_connectors([spelling[k].most_common(1)[0][0] for k in keys])[:MAX_WORDS]
    return chosen if any(c.isalpha() for w in chosen for c in w) else []  # only a year in common is not a name


def trim_connectors(words: list[str]) -> list[str]:
    """Drop small words from the ends: 'Coldplay at' -> 'Coldplay', 'in Paris' -> 'Paris' (but 'The Cure' stays)."""
    end = len(words)
    while end and _key(words[end - 1]) in _CONNECTORS:
        end -= 1
    start = 0
    while start < end and _key(words[start]) in _CONNECTORS and _key(words[start]) != "the":
        start += 1
    return words[start:end]


def concert_name(titles: list[str], fallback: str = "") -> str:
    """The words most of `titles` share, e.g. 'Coldplay at Wembley Stadium 2022'. `fallback` (what the user typed)
    is used, tidied up, when the titles give nothing to go on."""
    chosen = shared_words(titles) or _words(fallback)[:MAX_WORDS]
    # connectors stay lowercase inside a name ("Coldplay at Wembley"); a leading one is capitalised like any word
    words = [w if i and _key(w) in _CONNECTORS else _pretty(w) for i, w in enumerate(chosen)]
    return safe_filename(" ".join(words)) or "Concert"


def unique_stem(folder: Path, stem: str, exts=(".mp4", ".wav")) -> str:
    """`stem`, or 'stem (2)', 'stem (3)' ... so that none of the files it would create exists yet."""
    n = 1
    while True:
        candidate = stem if n == 1 else f"{stem} ({n})"
        if not any((Path(folder) / (candidate + ext)).exists() for ext in exts):
            return candidate
        n += 1
