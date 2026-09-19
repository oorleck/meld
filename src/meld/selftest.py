"""Check that an installed copy works end to end, without the internet.

Generates two short "phone recordings" of the same synthetic music, then runs sync, audio and video on them. Used
to verify the packaged Windows app (`Meld.exe --selftest`).
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 44100


def _music(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(int(seconds * SR), np.float32)
    notes = [110, 146.8, 196, 220, 261.6, 329.6, 440, 523.3]
    for k in range(int(seconds * 4)):
        start = int((k * 0.25 + rng.uniform(0, 0.12)) * SR)
        m = min(int(0.3 * SR), len(x) - start)
        if m <= 0:
            break
        t = np.arange(m) / SR
        f = notes[int(rng.integers(len(notes)))] * (1 + 0.01 * rng.standard_normal())
        phase = rng.uniform(0, 2 * np.pi, 3)
        x[start:start + m] += 0.3 * np.exp(-t * 8) * sum(np.sin(2 * np.pi * f * h * t + phase[h - 1]) / h for h in (1, 2, 3))
        if k % 2 == 0:
            x[start:start + m] += 0.5 * rng.standard_normal(m) * np.exp(-t * 40)
    return x


def _clip(path: Path, audio: np.ndarray, video: str) -> None:
    from .media import run_ffmpeg

    wav = path.with_suffix(".tmp.wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    run_ffmpeg([
        "-f", "lavfi", "-i", f"{video}:duration={len(audio) / SR}", "-i", wav,
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path,
    ])
    wav.unlink()


def run(report=print, online: bool = False) -> bool:
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        report(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}".rstrip())

    from . import ytdlp
    from .audiofuse import fuse_audio
    from .media import ffmpeg_exe, probe
    from .project import Project
    from .sync import sync_project
    from .videocut import render_video

    try:
        yt = ytdlp.import_yt_dlp()
        check("yt-dlp loads", True, f"version {yt.version.__version__}")
        if online:
            with yt.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
                found = ydl.extract_info("ytsearch3:live concert", download=False).get("entries") or []
            check("YouTube search works", len(found) > 0, f"{len(found)} results")
    except Exception as e:  # noqa: BLE001
        check("yt-dlp works", False, repr(e))
    check("ffmpeg found", Path(ffmpeg_exe()).exists(), ffmpeg_exe())

    with tempfile.TemporaryDirectory(prefix="meld-selftest-") as tmp:
        project = Project(tmp)
        music = _music(40)
        rng = np.random.default_rng(1)
        for name, (a, b), video in (
            ("a.mp4", (0, 28), "testsrc2=size=640x360:rate=30"),
            ("b.mp4", (12, 40), "rgbtestsrc=size=640x360:rate=30"),
        ):
            seg = music[int(a * SR): int(b * SR)]
            seg = seg / seg.std() * 0.15 + 0.03 * rng.standard_normal(len(seg)).astype(np.float32)
            _clip(project.clips_dir / name, seg.astype(np.float32), video)

        tl = sync_project(project, log=lambda *_: None)
        offsets = {c.file: c.offset for c in tl.clips}
        gap = offsets.get("b.mp4", 0) - offsets.get("a.mp4", 0)
        check("sync aligns the clips", len(tl.clips) == 2 and abs(gap - 12) < 0.05, f"offset {gap:.3f}s (expected 12)")

        # the same matching with the comparisons in separate processes (in the installed app these are Meld.exe
        # started again, which only works if the entry point handles it): same answer, and no fall-back to one at a time
        lines: list[str] = []
        tl2 = sync_project(project, log=lines.append, workers=2)
        same = [(c.file, round(c.offset, 6)) for c in tl2.clips] == [(c.file, round(c.offset, 6)) for c in tl.clips]
        used = any("at a time." in line and "Comparing" in line for line in lines)
        fell_back = any("carrying on" in line for line in lines)
        check(
            "matching in worker processes", same and used and not fell_back and len(tl2.clips) == 2,
            "same result as one at a time" if same and used and not fell_back
            else f"same={same} started={used} fell back={fell_back}",
        )

        wav = fuse_audio(project, log=lambda *_: None)
        check("audio fused", Path(wav).exists() and Path(wav).stat().st_size > 1_000_000)

        video = render_video(project, size=(640, 360), log=lambda *_: None)
        info = probe(Path(video))
        check(
            "video cut", info.has_video and info.has_audio and abs(info.duration - 40) < 0.5,
            f"{info.duration:.1f}s, {info.width}x{info.height}",
        )
    report("ALL OK" if ok else "SOMETHING FAILED")
    return ok
