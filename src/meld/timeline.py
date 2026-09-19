from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

OUT_FPS = 30


@dataclass
class ClipEntry:
    file: str
    offset: float  # seconds from the start of the shared timeline
    duration: float
    has_video: bool
    width: int
    height: int
    z: float | None = None  # sync confidence (None for the reference clip)
    anchor: str | None = None  # clip this one was aligned against

    @property
    def end(self) -> float:
        return self.offset + self.duration


@dataclass
class Timeline:
    """The clips that are fused (`clips`, one aligned group), plus what was left out: clips that matched nothing
    (`rejected`) and other groups of clips that line up with each other but not with `clips` (`others`, biggest first;
    another concert, or another part of the same one). Each group has its own time, starting at 0."""

    clips: list[ClipEntry] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    others: list[list[ClipEntry]] = field(default_factory=list)

    @property
    def left_out(self) -> int:
        """How many clips were found but are not used."""
        return len(self.rejected) + sum(len(g) for g in self.others)

    @property
    def end(self) -> float:
        return max((c.end for c in self.clips), default=0.0)

    def segments(self) -> list[tuple[float, float]]:
        """Time ranges covered by at least one clip; gaps between them are skipped in the output."""
        merged: list[list[float]] = []
        for a, b in sorted((c.offset, c.end) for c in self.clips):
            if merged and a <= merged[-1][1] + 0.05:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(a, b) for a, b in merged if b - a >= 0.5]

    def frame_segments(self) -> list[tuple[float, int]]:
        """(start_time, n_frames) per segment; audio and video are both cut to these exact lengths."""
        return [(a, round((b - a) * OUT_FPS)) for a, b in self.segments()]

    def save(self, path: Path) -> None:
        data = {
            "version": 2,
            "clips": [asdict(c) for c in self.clips],
            "rejected": self.rejected,
            "other_groups": [[asdict(c) for c in group] for group in self.others],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Timeline":
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # version 1 files have no other_groups
        others = [[ClipEntry(**c) for c in group] for group in data.get("other_groups", [])]
        return cls([ClipEntry(**c) for c in data["clips"]], data.get("rejected", []), others)
