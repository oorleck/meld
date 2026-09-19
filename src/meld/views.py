"""Which phones can see the same thing, and when?

A 3D model of a moment needs phones that look at the same scenery. Picking phones by how sharp their picture is (what
recon.py used to do) can hand the model a wide shot from the crowd, a close-up of a drummer on the big screen and a view
of the stage from the side, which have nothing in common: the model then reports its lowest confidence for every pixel.

Two frames see the same thing when features of one (SIFT, as COLMAP does them) can be matched with features of the
other in a way one camera move explains: those are the verified matches. Unrelated frames get none; two views of one
scene get hundreds. Phones are connected when they have enough of them, and a set of phones connected one to another is
a set that can be reconstructed together, although not all of them see the same part of the scene.
"""
from __future__ import annotations

import itertools
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .project import Project
from .timeline import Timeline

MIN_MATCHES = 25  # verified matches that make two views one scene (unrelated views get none; two of one scene hundreds)
MIN_VIEWS = 3  # fewest phones worth reconstructing from


def pair_matches(frames: list[Path], work: Path | None = None) -> dict[tuple[int, int], int]:
    """The verified matches of every pair of the image files `frames`: {(i, j): how many}, i < j."""
    import pycolmap

    own = work is None
    work = Path(tempfile.mkdtemp(prefix="meld-views-")) if own else Path(work)
    try:
        shutil.rmtree(work / "images", ignore_errors=True)
        (work / "images").mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(frames):
            shutil.copy(f, work / "images" / f"{k:03d}.jpg")
        db = work / "views.db"
        db.unlink(missing_ok=True)
        try:
            pycolmap.logging.minloglevel = 3  # COLMAP writes a line for everything it does
        except AttributeError:
            pass
        pycolmap.extract_features(db, work / "images", camera_mode=pycolmap.CameraMode.PER_IMAGE)
        pycolmap.match_exhaustive(db)
        d = pycolmap.Database.open(str(db))
        try:
            ids = {im.name: im.image_id for im in d.read_all_images()}
            out = {}
            for i, j in itertools.combinations(range(len(frames)), 2):
                a, b = ids.get(f"{i:03d}.jpg"), ids.get(f"{j:03d}.jpg")
                out[(i, j)] = len(d.read_two_view_geometry(a, b).inlier_matches) if a and b else 0
            return out
        finally:
            d.close()
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)


def connected_sets(n: int, pairs: dict[tuple[int, int], int], min_matches: int = MIN_MATCHES) -> list[list[int]]:
    """The sets of views (numbered 0 to n-1) that are connected one to another by pairs of at least `min_matches`,
    the biggest first. A view that no other is connected to is a set of one."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), m in pairs.items():
        if m >= min_matches:
            parent[find(i)] = find(j)
    sets: dict[int, list[int]] = {}
    for v in range(n):
        sets.setdefault(find(v), []).append(v)
    return sorted(sets.values(), key=lambda s: (-len(s), s))


@dataclass
class Moment:
    """What the phones filming at time `t` can see of each other."""

    t: float
    clips: list[int]  # the indices in the timeline of the clips that were filming, in the order of the frames
    pairs: dict[tuple[int, int], int] = field(default_factory=dict)  # verified matches between them (indices into `clips`)
    sets: list[list[int]] = field(default_factory=list)  # connected sets of them, as indices into the timeline, biggest first

    @property
    def best(self) -> list[int]:
        return self.sets[0] if self.sets else []

    @property
    def strength(self) -> int:
        """How much the biggest set has to hold it together: the matches of its weakest link, roughly. 0 if it is one view."""
        inside = [m for (i, j), m in self.pairs.items() if self.clips[i] in self.best and self.clips[j] in self.best]
        return int(np.median(sorted(inside)[-max(1, len(self.best) - 1):])) if len(self.best) > 1 and inside else 0


def live_clips(tl: Timeline, t: float, margin: float = 0.5) -> list[int]:
    return [i for i, c in enumerate(tl.clips) if c.has_video and c.offset + margin <= t <= c.end - margin]


def candidate_times(tl: Timeline, step: float = 10.0, at_least: int = MIN_VIEWS, margin: float = 0.5) -> list[float]:
    """The moments, every `step` seconds, at which at least `at_least` phones are filming."""
    return [float(t) for t in np.arange(0.0, tl.end, step) if len(live_clips(tl, float(t), margin)) >= at_least]


def look_at(project: Project, tl: Timeline, t: float, max_views: int = 24, min_matches: int = MIN_MATCHES,
            frames_dir: Path | None = None) -> Moment:
    """Which of the phones filming at `t` (at most `max_views`, the sharpest) can see the same thing."""
    from .recon import extract_frames, select_views

    clips = select_views(project, tl, t, max_views)
    if len(clips) < 2:
        return Moment(t, clips, {}, [[c] for c in clips])
    own = frames_dir is None
    frames_dir = Path(tempfile.mkdtemp(prefix="meld-frames-")) if own else Path(frames_dir)
    try:
        frames = extract_frames(project, tl, clips, t, frames_dir)
        pairs = pair_matches(frames, frames_dir / "work")
    finally:
        if own:
            shutil.rmtree(frames_dir, ignore_errors=True)
    sets = [[clips[k] for k in s] for s in connected_sets(len(clips), pairs, min_matches)]
    return Moment(t, clips, pairs, sets)


def scan(project: Project, tl: Timeline, times: list[float], max_views: int = 24, min_matches: int = MIN_MATCHES,
         log=lambda *_: None) -> list[Moment]:
    """`look_at` every moment in `times`, the most promising (the biggest connected set, then the strongest) first."""
    moments = []
    for n, t in enumerate(times, 1):
        m = look_at(project, tl, t, max_views, min_matches)
        moments.append(m)
        log(f"  {t:7.1f}s: {len(m.clips)} phones filming, {len(m.best)} of them see the same thing ({m.strength} matches)"
            f" [{n}/{len(times)}]")
    return sorted(moments, key=lambda m: (-len(m.best), -m.strength, m.t))
