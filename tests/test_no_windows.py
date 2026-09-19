"""The installed app has no console, so every program it starts must ask for no window of its own. Where Windows
Terminal is the default terminal, one that does not gets a Terminal window, once per call (hundreds in a run)."""
import ast
import subprocess
from pathlib import Path

import pytest
from synth import phone, synth_music, write_wav

from meld import media
from meld.media import NO_WINDOW

SRC = Path(__file__).resolve().parents[1] / "src" / "meld"
LAUNCHERS = {"run", "Popen", "call", "check_call", "check_output"}
# recon.py is the optional 3D reconstruction, only ever run from a terminal by the command line (never in the app).
NOT_IN_THE_APP = {"recon.py"}


def test_the_flag_is_create_no_window_on_windows_and_nothing_elsewhere():
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        assert NO_WINDOW == subprocess.CREATE_NO_WINDOW != 0
    else:
        assert NO_WINDOW == 0  # creationflags must be 0 outside Windows


def test_every_ffmpeg_call_is_started_with_no_window(tmp_path, monkeypatch):
    seen = []
    real_run, real_popen = subprocess.run, subprocess.Popen

    def run(*a, **kw):
        seen.append(("run", kw.get("creationflags")))
        return real_run(*a, **kw)

    class Popen(real_popen):
        def __init__(self, *a, **kw):
            seen.append(("Popen", kw.get("creationflags")))
            super().__init__(*a, **kw)

    monkeypatch.setattr(media.subprocess, "run", run)
    monkeypatch.setattr(media.subprocess, "Popen", Popen)

    clip = tmp_path / "c.mp4"
    wav = tmp_path / "c.wav"
    write_wav(wav, phone(synth_music(6, 16000), 16000, 0, 5, seed=1), 16000)
    media.run_ffmpeg([  # a call of the first kind
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=2", "-i", wav,
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip,
    ])
    assert media.probe(clip).has_video  # the second kind
    assert sum(1 for _ in media.iter_gray_frames(clip, 5, 32, 24)) > 0  # the third, a long-running one
    kinds = [k for k, _ in seen]
    assert kinds.count("run") == 2 and kinds.count("Popen") >= 1
    assert all(flags == NO_WINDOW for _, flags in seen), seen


def _launch_calls(path: Path):
    """(line, has creationflags) for every subprocess launch written in a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        owner = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else None
        if name in LAUNCHERS and (owner == "subprocess" or name == "Popen"):
            yield node.lineno, any(k.arg == "creationflags" for k in node.keywords)
        if owner == "os" and name in {"system", "popen"}:
            yield node.lineno, False  # these cannot ask for no window at all


@pytest.mark.parametrize("path", sorted(p for p in SRC.glob("*.py") if p.name not in NOT_IN_THE_APP), ids=lambda p: p.name)
def test_nothing_in_the_app_starts_a_program_that_could_open_a_window(path):
    missing = [line for line, ok in _launch_calls(path) if not ok]
    assert not missing, f"{path.name} starts a program without creationflags=NO_WINDOW at line(s) {missing}"


def test_the_source_check_would_notice_a_launch_without_the_flag(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import subprocess, os\nsubprocess.run(['x'])\nsubprocess.Popen(['y'], creationflags=0)\nos.system('z')\n")
    assert list(_launch_calls(bad)) == [(2, False), (3, True), (4, False)]
