"""Cut a multicam video from the aligned clips.

Every video clip is scored for sharpness and exposure at 2 fps. Then there are two ways to cut, both landing on exact
frame boundaries so that picture stays locked to the fused audio:

- To the music (the default, where there is a beat to find; see beats.py). Cuts go on beats, a couple of frames early,
  as editors do. How long a shot lasts follows the music: short shots (a beat or two) where it is loud and driving or
  building up, and longer ones (a bar or two) where it is quiet, and shorter at a fast tempo than a slow one; cuts prefer
  to end on the first beat of a bar. Which angle is shown is chosen per shot, by picture quality, never the same one twice
  running if there is another to show.
- Otherwise (no beat, or asked not to): a greedy editor walks the timeline in 0.5 s steps, staying on the current angle
  until a clearly better one is available (never cutting faster than min_shot, never staying longer than max_shot) and
  avoiding angles it just used.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import beats as beatfinder
from .media import ffmpeg_exe, iter_gray_frames, load_audio, run_ffmpeg
from .project import Project
from .timeline import OUT_FPS, ClipEntry, Timeline

GRID = 0.5  # seconds per planning step; equals 1 / ANALYSIS_FPS
ANALYSIS_FPS = 2
AN = 256

# cutting to the music
MIN_CONFIDENCE = 0.25  # below this there is no beat to speak of (talk, applause, noise): cut by picture quality instead
MUSIC_MIN_SHOT = 0.9  # the shortest shot when cutting to the music, seconds
LEAD = 0.06  # cut this long before the beat: a cut just ahead of the beat reads as on it (about two frames)
SHOT_INTENSE, SHOT_CALM = 1.5, 9.0  # the shot length aimed at, seconds, for the most intense and the calmest music
TEMPO_EXPONENT = 0.7  # how much a fast tempo shortens shots and a slow one lengthens them: shot length goes as tempo to this power
BEATS_PER_SHOT = {  # how many beats a shot may last, and how much less musical each is (a beat, half a bar, a bar, two
    # bars are; three, six and twelve not quite; the rest are awkward, but a shot of an odd number of beats is what puts
    # the cuts back on the strong beats if they have drifted onto the weak ones, since shots of 2, 4, 6 ... beats cannot)
    **{m: 0.0 for m in (1, 2, 4, 8, 16)}, **{m: 0.1 for m in (3, 6, 12)}, **{m: 0.15 for m in (5, 7, 9, 10, 11, 13, 14, 15)},
}
RHYTHM_WEIGHT = 0.5  # how much aiming at the wrong length costs, next to picture quality (0..1)


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


def target_shot(energy: float, rise: float, bpm: float, pace: float = 1.0) -> float:
    """How long a shot should be here, in seconds: from SHOT_CALM down to SHOT_INTENSE as the music gets more intense
    (louder than the rest of the concert, or building up), and shorter still at a faster tempo (120 BPM is neutral).
    `pace` above 1 cuts faster."""
    intensity = min(1.0, max(0.0, energy + 0.35 * rise))
    seconds = SHOT_CALM * (SHOT_INTENSE / SHOT_CALM) ** intensity
    return seconds * (120.0 / min(max(bpm, 60.0), 200.0)) ** TEMPO_EXPONENT / max(pace, 1e-3)


def plan_music(S, entries, vidx, t0, t1, bm, min_shot, max_shot, pace=1.0, lead=LEAD, variety=0.8):
    """Shot list for one stretch of music, like plan_segment: [(first_step, end_step, clip_index | None)] (steps may be
    fractions, as cuts are on beats, not on the 0.5 s grid). Cuts are on the beats of `bm` (a BeatMap, seconds from t0).

    From each cut it looks at ending the shot on any of the next few beats (a beat, two, three ... a bar, two bars), and
    with any clip that shows all of it, and takes the best: picture quality (against the best there is), less how far
    the length is from the one the music asks for, less something for not ending on a strong beat, and for showing the
    angle just shown or one shown lately. So the cuts follow the music, and the pictures follow what looks best."""
    times = [t0]
    meta = [(0, bm.energy[0], bm.rise[0], float(bm.bpm[0]))]  # (phase, energy, rise, bpm) at each cut
    for i, b in enumerate(bm.beats):
        cut = t0 + float(b) - lead
        if t0 + 0.25 < cut < t1 - 0.25:
            times.append(cut)
            meta.append((int(bm.phase[i]), float(bm.energy[i]), float(bm.rise[i]), float(bm.bpm[i])))
    times.append(t1)
    meta.append((0, meta[-1][1], meta[-1][2], meta[-1][3]))
    last = len(times) - 1
    steps = lambda a, b: (max(int(a / GRID), 0), max(int(math.ceil(b / GRID)), int(a / GRID) + 1))  # noqa: E731

    shots: list[tuple[float, float, int | None]] = []
    recent: list[int] = []
    lengths: list[int] = []  # the beats in the last shots, so that the same length is not used for ever
    pos = 0
    while pos < last:
        a = times[pos]
        _, energy, rise, bpm = meta[min(pos + 2, last)]  # the music a moment ahead: what this shot is played over
        want = target_shot(energy, rise, bpm, pace)
        options = []
        for m, unmusical in BEATS_PER_SHOT.items():
            j = pos + m
            if j > last:
                continue
            b = times[j]
            d = b - a
            if d > max_shot or (d < min_shot and j < last):
                continue
            if j < last and t1 - b < 0.7 * min_shot:  # would leave a sliver at the end
                continue
            av = [c for c in vidx if entries[c].offset <= a + 0.03 and entries[c].end >= b - 0.03]
            if not av:
                continue
            s0, s1 = steps(a, b)
            quality = {c: float(S[c, s0:s1].mean()) for c in av}
            phase = meta[j][0]
            options.append((j, m, d, unmusical + (0.0 if phase == 0 or j == last else 0.05 if phase == 2 else 0.35), quality))
        if not options:
            # nothing shows a whole beat-length: take the clip that lasts longest and cut where it ends
            here = [c for c in vidx if entries[c].offset <= a + 0.03 and entries[c].end > a + 0.25]
            if not here:
                nxt = times[pos + 1]
                shots.append((a / GRID, nxt / GRID, None))
                pos += 1
                continue
            c = max(here, key=lambda k: entries[k].end)
            end = min(entries[c].end, t1)
            j = next((k for k in range(pos + 1, last + 1) if times[k] >= end - 0.03), last)
            if times[j] > end + 0.03:  # the clip ends between beats: cut there
                times.insert(j, end)
                meta.insert(j, meta[j])
                last += 1
            shots.append((a / GRID, times[j] / GRID, c))
            recent.append(c)
            pos = j
            continue
        best_q = max(q for *_, quals in options for q in quals.values()) or 1.0
        # never the same angle twice running when another could be shown, however much better this one looks
        others = bool(recent) and any(c != recent[-1] for *_, quals in options for c in quals)
        run = 0  # how many shots in a row have had the length of the last one
        while run < len(lengths) and lengths[-1 - run] == lengths[-1]:
            run += 1
        pick = None
        for j, m, d, off, quals in options:
            rhythm = RHYTHM_WEIGHT * math.log2(d / want) ** 2 + off
            if run >= 2 and m == lengths[-1]:
                rhythm += 0.1 * run  # the third time running, and each one after: change it
            for c, q in quals.items():
                if others and c == recent[-1]:
                    continue
                score = (q / best_q) * variety ** recent[-3:].count(c) - rhythm
                if pick is None or score > pick[0]:
                    pick = (score, j, m, c)
        _, j, m, c = pick
        shots.append((a / GRID, times[j] / GRID, c))
        recent.append(c)
        lengths.append(m)
        pos = j
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


def find_beats(project: Project, tl: Timeline, audio_path, log=print) -> list:
    """A BeatMap (or None) for each stretch of the timeline that is played through, taken from the fused audio, which
    has those stretches one after another. Kept for next time, until the audio changes."""
    segs = tl.frame_segments()
    starts, at = [], 0.0
    for _, nf in segs:
        starts.append(at)
        at += nf / OUT_FPS
    stretches = [(a, nf / OUT_FPS) for a, (_, nf) in zip(starts, segs)]
    key = f"{audio_path.stat().st_size}:{audio_path.stat().st_mtime_ns}:{[(round(a, 3), round(n, 3)) for a, n in stretches]}"
    cache = project.cache_dir / "beatmaps.npz"
    try:
        with np.load(cache, allow_pickle=False) as z:
            if str(z["key"]) == key:
                out = []
                for i in range(len(stretches)):
                    if f"beats{i}" not in z:
                        out.append(None)
                        continue
                    out.append(beatfinder.BeatMap(
                        z[f"beats{i}"], z[f"period{i}"], z[f"phase{i}"], z[f"energy{i}"], z[f"rise{i}"], float(z[f"conf{i}"]),
                    ))
                log("Beats: using what was found last time.")
                return out
    except (OSError, ValueError, KeyError):
        pass
    log("Finding the beats ...")
    x = load_audio(audio_path, project.cache_dir, beatfinder.SR)
    maps = beatfinder.beat_maps(x, stretches, log=lambda *_: None)
    arrays = {"key": np.array(key)}
    for i, bm in enumerate(maps):
        if bm is not None:
            arrays.update({
                f"beats{i}": bm.beats, f"period{i}": bm.period, f"phase{i}": bm.phase, f"energy{i}": bm.energy,
                f"rise{i}": bm.rise, f"conf{i}": np.array(bm.confidence),
            })
    try:
        np.savez(cache, **arrays)
    except OSError:
        pass
    return maps


def render_video(
    project: Project, size=(1920, 1080), min_shot=None, max_shot=12.0, margin=0.15, variety=0.8, log=print,
    beats=True, pace=1.0,
):
    """`beats`: cut to the music where there is a beat (else by picture quality only). `pace` above 1 cuts faster, below
    1 slower. `min_shot` is the shortest shot: 0.9 s when cutting to the music, 3 s otherwise, unless given."""
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

    maps = find_beats(project, tl, audio_path, log) if beats else []
    jobs = []
    used = np.zeros(len(entries))
    music = plain = 0
    for k, (t0, nf) in enumerate(tl.frame_segments()):
        t1 = t0 + nf / OUT_FPS
        bm = maps[k] if k < len(maps) else None
        if bm is not None and bm.confidence >= MIN_CONFIDENCE and len(bm.beats) >= 4:
            plan = plan_music(S, entries, vidx, t0, t1, bm, MUSIC_MIN_SHOT if min_shot is None else min_shot, max_shot, pace, LEAD, variety)
            music += 1
        else:
            plan = plan_segment(S, entries, vidx, t0, t1, 3.0 if min_shot is None else min_shot, max_shot, margin, variety)
            plain += 1
        for a, b, clip in plan:
            fa = round((min(max(a * GRID, t0), t1) - t0) * OUT_FPS)
            fb = round((min(max(b * GRID, t0), t1) - t0) * OUT_FPS)
            if fb <= fa:
                continue
            t_clip = t0 + fa / OUT_FPS - (entries[clip].offset if clip is not None else 0)
            jobs.append((len(jobs), clip, t_clip, fb - fa, shots_dir / f"shot_{len(jobs):05d}.mp4"))
            if clip is not None:
                used[clip] += (fb - fa) / OUT_FPS
    if beats:
        log(f"Cut to the music in {music} stretch(es); {plain} without a beat cut by picture quality.")
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
