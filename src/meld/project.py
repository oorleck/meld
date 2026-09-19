from __future__ import annotations

import re
from pathlib import Path

MEDIA_EXTS = {
    ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".3gp",
    ".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus",
}
_PARTIAL = re.compile(r"\.f\d+\.|\.part$|\.ytdl$")


def slugify(text: str, fallback: str = "concert") -> str:
    """A folder-name-safe version of a search query."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or fallback


class Project:
    """A working directory: clips/ (inputs), cache/ (derived data), out/ (results)."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.clips_dir = self.root / "clips"
        self.cache_dir = self.root / "cache"
        self.out_dir = self.root / "out"
        for d in (self.clips_dir, self.cache_dir, self.out_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.timeline_path = self.root / "timeline.json"
        self.sources_path = self.root / "sources.json"

    def preview(self) -> "Project":
        """A project of its own inside this one, for small audio-only copies of the candidates: they are lined up by
        their sound to see which belong together before any full video is downloaded."""
        return Project(self.root / "preview")

    def clip_files(self) -> list[Path]:
        return sorted(
            p for p in self.clips_dir.iterdir()
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS and not _PARTIAL.search(p.name)
        )

    def find_clip(self, stem: str) -> Path | None:
        """The finished media file for a downloaded id (`stem`), if there is one; half-finished downloads do not count."""
        for p in sorted(self.clips_dir.glob(f"{stem}.*")):
            if p.suffix.lower() in MEDIA_EXTS and not _PARTIAL.search(p.name):
                return p
        return None

    def clip_path(self, entry) -> Path:
        return self.clips_dir / entry.file
