"""ffmpeg helpers. Uses the static ffmpeg binary bundled with imageio-ffmpeg, so nothing else needs installing."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

_FFMPEG: str | None = None


def ffmpeg_exe() -> str:
    global _FFMPEG
    if _FFMPEG is None:
        import imageio_ffmpeg

        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def run_ffmpeg(args: list) -> None:
    p = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-v", "error", "-y", *map(str, args)],
        capture_output=True,
    )
    if p.returncode:
        # a negative or huge code with no message means ffmpeg crashed rather than reporting an error
        raise RuntimeError(f"ffmpeg failed (exit code {p.returncode}): {p.stderr.decode(errors='replace')[-800:]}")


@dataclass
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool
    width: int
    height: int


def probe(path: Path) -> MediaInfo:
    p = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    text = p.stderr
    duration = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])

    has_audio = has_video = False
    width = height = 0
    for line in text.splitlines():
        if "Stream #" not in line:
            continue
        if ": Audio:" in line:
            has_audio = True
        elif ": Video:" in line and "attached pic" not in line and not has_video:
            vm = re.search(r",\s*(\d{2,5})x(\d{2,5})", line)
            if vm:
                has_video = True
                width, height = int(vm[1]), int(vm[2])

    rot = re.search(r"rotation of (-?[\d.]+) degrees", text)
    if has_video and rot and round(abs(float(rot[1]))) % 180 == 90:
        width, height = height, width
    return MediaInfo(duration, has_audio, has_video, width, height)


def audio_cache_path(src: Path, cache_dir: Path, sr: int) -> Path:
    """Where the decoded audio of `src` is kept (raw little-endian float32, mono, at `sr`)."""
    return cache_dir / "audio" / f"{src.name}.{sr}.f32"


def load_audio(src: Path, cache_dir: Path, sr: int) -> np.ndarray:
    """Mono float32 audio at `sr`, decoded once and memory-mapped from the cache."""
    dst = audio_cache_path(src, cache_dir, sr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        run_ffmpeg(["-i", src, "-vn", "-ac", "1", "-ar", sr, "-f", "f32le", dst])
    if dst.stat().st_size == 0:
        return np.zeros(0, np.float32)
    return np.memmap(dst, dtype="<f4", mode="r")


def iter_gray_frames(path: Path, fps: float, width: int, height: int) -> Iterator[np.ndarray]:
    """Frames sampled at `fps`, scaled to width x height, as uint8 grayscale. Frame k is at clip time k/fps."""
    cmd = [
        ffmpeg_exe(), "-v", "error", "-i", str(path), "-an",
        "-vf", f"fps={fps}:start_time=0,scale={width}:{height},format=gray",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
