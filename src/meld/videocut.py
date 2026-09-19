"""Cut a multicam video from the aligned clips.

Every video clip is scored for sharpness and exposure at 2 fps. A greedy editor then walks the
timeline in 0.5 s steps, staying on the current angle until a clearly better one is available
(never cutting faster than min_shot, never staying longer than max_shot) and avoiding angles
it just used. Shots are cut on exact frame boundaries so picture stays locked to the fused audio.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import uniform_filter1d

from .media import ffmpeg_exe, iter_gray_frames, run_ffmpeg
from .project import Project
from .timeline import OUT_FPS, ClipEntry, Timeline

GRID = 0.5  # seconds per planning step; equals 1 / ANALYSIS_FPS
ANALYSIS_FPS = 2
AN = 256


def _frame_score(frame: np.ndarray) -> float:
    f = frame.astype(np.float32)
    lap = 4 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]
    sharp = math.log1p(float(lap.var()))
    mean = float(f.mean())
    blown = float((f >= 250).mean() + (f <= 12).mean())
    exposure = math.exp(-(((mean - 115) / 95) ** 2)) * (1 - min(blown, 0.8))
    return sharp * exposure


def score_clip(project: Project, entry: ClipEntry) -> np.ndarray:
    path = project.clip_path(entry)
    cache = project.cache_dir / "vscore" / f"{entry.file}.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)
    res = min(1.0, (min(entry.width, entry.height) / 720) ** 0.5)
    scores = np.array(
        [_frame_score(fr) * res for fr in iter_gray_frames(path, ANALYSIS_FPS, AN, AN)], np.float32
    )
    np.save(cache, scores)
    return scores


def plan_segment(S, entries, vidx, t0, t1, min_shot, max_shot, margin, variety):
    """Greedy shot list for one segment: [(first_step, end_step, clip_index | None)]."""
    min_steps = max(1, round(min_shot / GRID))
    max_steps = max(min_steps + 1, round(max_shot / GRID))
    s_lo = int(math.floor(t0 / GRID + 1e-9))
    s_hi = int(math.ceil(t1 / GRID - 1e-9))
    shots: list[tuple[int, int, int | None]] = []
    recent: list[int] = []
    cur: int | None = None
    cur_start = s_lo

    for s in range(s_lo, s_hi):
        a, b = max(s * GRID, t0), min((s + 1) * GRID, t1)
        av = [c for c in vidx if entries[c].offset <= a + 0.03 and entries[c].end >= b - 0.03]

        def adj(c):
            return S[c, s] * variety ** recent[-3:].count(c)

        best = max(av, key=adj) if av else None
        choice = cur
        if not av:
            choice = None
        elif cur not in av:
            choice = best
        elif s - cur_start >= max_steps:
            others = [c for c in av if c != cur]
            choice = max(others, key=adj) if others else cur
        elif s - cur_start >= min_steps and best != cur and adj(best) > adj(cur) * (1 + margin):
            choice = best

        if choice != cur:
            if s > cur_start:
                shots.append((cur_start, s, cur))
            cur, cur_start = choice, s
            if cur is not None:
                recent.append(cur)
    if s_hi > cur_start:
        shots.append((cur_start, s_hi, cur))
    return shots


def _shot_filter(entry: ClipEntry, W: int, H: int) -> tuple[str, list]:
    if abs(entry.width / entry.height - W / H) < 0.02:
        return f"scale={W}:{H},setsar=1,fps={OUT_FPS},tpad=stop_mode=clone:stop_duration=1,format=yuv420p", ["-vf"]
    # Different aspect ratio (e.g. vertical phone video): fit inside a blurred copy of itself.
    fc = (
        f"[0:v]split=2[a][b];"
        f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:2[bg];"
        f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={OUT_FPS},"
        f"tpad=stop_mode=clone:stop_duration=1,format=yuv420p[v]"
    )
    return fc, ["-filter_complex"]


_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(OUT_FPS)]


def _render_shot(project, entries, size, job):
    idx, clip, t_clip, n, out = job
    W, H = size
    if clip is None:
        run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={OUT_FPS}", "-frames:v", n, *_ENC, out])
        return
    e = entries[clip]
    flt, flag = _shot_filter(e, W, H)
    args = ["-ss", f"{max(t_clip, 0):.3f}", "-i", project.clip_path(e), "-an", *flag, flt]
    if flag[0] == "-filter_complex":
        args += ["-map", "[v]"]
    run_ffmpeg([*args, "-frames:v", n, *_ENC, out])


def render_video(
    project: Project, size=(1920, 1080), min_shot=3.0, max_shot=12.0, margin=0.15, variety=0.8, log=print,
):
    tl = Timeline.load(project.timeline_path)
    audio_path = project.out_dir / "fused_audio.wav"
    if not audio_path.exists() or audio_path.stat().st_mtime < project.timeline_path.stat().st_mtime:
        from .audiofuse import fuse_audio

        log("Fused audio is missing or older than the timeline; fusing it first ...")
        audio_path = fuse_audio(project, log=log)
    entries = tl.clips
    vidx = [i for i, e in enumerate(entries) if e.has_video]
    if not vidx:
        raise SystemExit("None of the aligned clips has a video track.")

    log(f"Scoring video quality of {len(vidx)} clip(s) ...")
    nsteps = int(math.ceil(tl.end / GRID)) + 2
    S = np.zeros((len(entries), nsteps), np.float32)
    for c in vidx:
        s = score_clip(project, entries[c])
        start = int(round(entries[c].offset / GRID))
        n = min(len(s), nsteps - start)
        S[c, start:start + n] = s[:n]
    S = uniform_filter1d(S, 5, axis=1, mode="nearest")

    shots_dir = project.cache_dir / "shots"
    shots_dir.mkdir(exist_ok=True)
    for old in shots_dir.glob("shot_*.mp4"):
        old.unlink()

    jobs = []
    used = np.zeros(len(entries))
    for t0, nf in tl.frame_segments():
        t1 = t0 + nf / OUT_FPS
        for a, b, clip in plan_segment(S, entries, vidx, t0, t1, min_shot, max_shot, margin, variety):
            fa = round((min(max(a * GRID, t0), t1) - t0) * OUT_FPS)
            fb = round((min(max(b * GRID, t0), t1) - t0) * OUT_FPS)
            if fb <= fa:
                continue
            t_clip = t0 + fa / OUT_FPS - (entries[clip].offset if clip is not None else 0)
            jobs.append((len(jobs), clip, t_clip, fb - fa, shots_dir / f"shot_{len(jobs):05d}.mp4"))
            if clip is not None:
                used[clip] += (fb - fa) / OUT_FPS
    log(f"Rendering {len(jobs)} shots ...")

    pool = ThreadPoolExecutor(max_workers=3)
    try:
        for i, _ in enumerate(pool.map(lambda j: _render_shot(project, entries, size, j), jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)}")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # stop queued shots if log() raised (Cancel)

    list_path = project.cache_dir / "shots.txt"
    list_path.write_text(
        "".join("file '{}'\n".format(j[4].as_posix().replace("'", "'\\''")) for j in jobs), encoding="utf-8"
    )
    out_path = project.out_dir / "fused.mp4"
    run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", list_path, "-i", audio_path,
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
        "-movflags", "+faststart", out_path,
    ])
    log(f"Fused video -> {out_path}")
    for c in vidx:
        log(f"  {used[c]:7.1f}s shown from {entries[c].file}")
    return out_path
