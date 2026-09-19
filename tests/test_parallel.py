"""Running comparisons in worker processes must change how long matching takes and nothing else."""
import numpy as np
import pytest
from synth import phone, synth_music, write_wav
from test_groups import A, B, SR, TWO_CONCERTS, project_with

from meld import pool as pool_mod
from meld.pool import Comparer, default_workers
from meld.project import Project
from meld.sync import correlate, sync_project


def run(project, workers, **kw):
    lines = []
    tl = sync_project(project, log=lines.append, workers=workers, **kw)
    return tl, lines


def summary(tl):
    """Everything a run decided, with the scores, so that 'the same' means exactly the same."""
    def group(clips):
        return [(c.file, c.offset, c.z, c.anchor) for c in clips]

    return group(tl.clips), [group(g) for g in tl.others], sorted((r["file"], r["reason"]) for r in tl.rejected)


def test_parallel_gives_exactly_the_same_result_as_one_at_a_time(tmp_path):
    serial, _ = run(project_with(tmp_path / "s", TWO_CONCERTS), workers=1)
    parallel, lines = run(project_with(tmp_path / "p", TWO_CONCERTS), workers=3)
    assert any("3 clips at a time" in line for line in lines)  # it really did use worker processes
    assert summary(parallel) == summary(serial)


def test_the_same_when_clips_have_to_wait_for_the_second_look(tmp_path):
    spec = {f"a{i}": (A, i * 30, i * 30 + 60) for i in range(5)}
    spec |= {"b1": (B, 0, 20), "b2": (B, 12, 32), "lone": (synth_music(60, SR, seed=9), 0, 30)}
    serial, _ = run(project_with(tmp_path / "s", spec), workers=1, max_compare=3)
    parallel, _ = run(project_with(tmp_path / "p", spec), workers=4, max_compare=3)
    assert {"b1.wav", "b2.wav"} <= {c.file for g in parallel.others for c in g}  # the second look did its work
    assert summary(parallel) == summary(serial)


def test_the_same_with_a_bridge_between_groups(tmp_path):
    spec = {"x": (A, 0, 70), "y": (A, 100, 170), "z": (A, 60, 110)}
    serial, _ = run(project_with(tmp_path / "s", spec), workers=1)
    parallel, _ = run(project_with(tmp_path / "p", spec), workers=2)
    assert summary(parallel) == summary(serial) and {c.file for c in parallel.clips} == {"x.wav", "y.wav", "z.wav"}


def test_a_small_job_is_left_on_one_process(tmp_path):
    _, lines = run(project_with(tmp_path, {"a1": (A, 0, 40), "a2": (A, 20, 60)}), workers=None)
    assert not any("at a time" in line for line in lines)  # starting workers would cost more than it saves


def test_workers_can_be_switched_off(tmp_path):
    _, lines = run(project_with(tmp_path, TWO_CONCERTS), workers=1)
    assert not any("at a time" in line for line in lines)


def test_default_workers_is_a_sensible_number(monkeypatch):
    for cores, want in [(None, 1), (1, 1), (2, 1), (4, 2), (8, 4), (16, 4), (64, 4)]:
        monkeypatch.setattr(pool_mod.os, "cpu_count", lambda cores=cores: cores)
        assert default_workers() == want


# ---- the Comparer on its own


def _two_files(tmp_path):
    a = phone(A, SR, 0, 40, seed=1)
    b = phone(A, SR, 15, 55, seed=2)
    paths = {}
    for name, x in (("a", a), ("b", b)):
        p = tmp_path / f"{name}.f32"
        p.write_bytes(x.astype("<f4").tobytes())
        paths[name] = str(p)
    mm = {k: np.memmap(v, dtype="<f4", mode="r") for k, v in paths.items()}
    return paths, mm


def test_comparer_answers_from_a_worker_with_the_serial_answer(tmp_path):
    paths, mm = _two_files(tmp_path)
    want = correlate(mm["a"], mm["b"], SR)
    comparer = Comparer(paths, lambda a, b: correlate(mm[a], mm[b], SR), workers=2, fs=SR)
    try:
        comparer.top_up(iter([("a", "b")]))
        assert comparer.parallel and ("a", "b") in comparer._flying  # it was started in the background
        assert comparer.result("a", "b") == want
        assert not comparer._flying
    finally:
        comparer.close()


def test_comparer_works_out_a_pair_nobody_started(tmp_path):
    paths, mm = _two_files(tmp_path)
    calls = []

    def compute(a, b):
        calls.append((a, b))
        return correlate(mm[a], mm[b], SR)

    comparer = Comparer(paths, compute, workers=2, fs=SR)
    try:
        assert comparer.result("a", "b") == correlate(mm["a"], mm["b"], SR)
        assert calls == [("a", "b")]
    finally:
        comparer.close()


def test_comparer_with_one_worker_never_starts_a_process(tmp_path, monkeypatch):
    paths, mm = _two_files(tmp_path)
    monkeypatch.setattr(pool_mod, "ProcessPoolExecutor", lambda *a, **k: pytest.fail("no process wanted"))
    comparer = Comparer(paths, lambda a, b: correlate(mm[a], mm[b], SR), workers=1, fs=SR)
    comparer.top_up(iter([("a", "b")]))
    assert not comparer.parallel and comparer.result("a", "b") == correlate(mm["a"], mm["b"], SR)


def test_comparer_carries_on_one_at_a_time_if_processes_cannot_be_started(tmp_path, monkeypatch):
    paths, mm = _two_files(tmp_path)

    def cannot(*a, **k):
        raise OSError("no more processes")

    monkeypatch.setattr(pool_mod, "ProcessPoolExecutor", cannot)
    said = []
    comparer = Comparer(paths, lambda a, b: correlate(mm[a], mm[b], SR), workers=4, fs=SR, on_fallback=said.append)
    comparer.top_up(iter([("a", "b")]))
    assert not comparer.parallel and len(said) == 1 and "could not start" in said[0]
    assert comparer.result("a", "b") == correlate(mm["a"], mm["b"], SR)  # still answered


def test_a_whole_run_still_finishes_when_worker_processes_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_mod, "ProcessPoolExecutor", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    serial, _ = run(project_with(tmp_path / "s", TWO_CONCERTS), workers=1)
    fallback, lines = run(project_with(tmp_path / "p", TWO_CONCERTS), workers=4)
    assert any("one comparison at a time" in line for line in lines)
    assert summary(fallback) == summary(serial)


def test_cancel_forgets_what_was_started(tmp_path):
    paths, mm = _two_files(tmp_path)
    comparer = Comparer(paths, lambda a, b: correlate(mm[a], mm[b], SR), workers=2, fs=SR)
    try:
        comparer.top_up(iter([("a", "b"), ("a", "b")]))
        comparer.cancel()  # whether it was dropped (not begun) or kept (already running), asking later still works
        assert comparer.result("a", "b") == correlate(mm["a"], mm["b"], SR)
        assert not comparer._flying
    finally:
        comparer.close()


def test_the_audio_cache_path_is_what_load_audio_uses(tmp_path):
    from meld.media import audio_cache_path, load_audio

    project = Project(tmp_path)
    write_wav(project.clips_dir / "c.wav", phone(A, SR, 0, 10, seed=1), SR)
    (clip,) = project.clip_files()
    load_audio(clip, project.cache_dir, 16000)
    assert audio_cache_path(clip, project.cache_dir, 16000).exists()



def test_one_failed_comparison_in_a_worker_is_worked_out_here_and_the_pool_stays_on(tmp_path):
    paths, mm = _two_files(tmp_path)
    good = dict(paths)
    paths_bad = {**paths, "gone": str(tmp_path / "missing.f32")}
    mm["gone"] = mm["b"]  # what the local computation would use for the same name
    comparer = Comparer(paths_bad, lambda a, b: correlate(mm[a], mm[b], SR), workers=2, fs=SR)
    try:
        comparer.top_up(iter([("a", "gone")]))  # the worker cannot open the file and raises
        assert comparer.result("a", "gone") == correlate(mm["a"], mm["b"], SR)  # answered here instead
        assert comparer.parallel  # a bad file is not a dead pool
    finally:
        comparer.close()
    assert good


def test_finished_comparisons_are_kept_when_a_guess_is_cancelled(tmp_path):
    paths, mm = _two_files(tmp_path)
    comparer = Comparer(paths, lambda a, b: correlate(mm[a], mm[b], SR), workers=2, fs=SR)
    try:
        comparer.top_up(iter([("a", "b")]))
        future = comparer._flying[("a", "b")]
        future.result()  # it has finished
        comparer.cancel()
        assert ("a", "b") in comparer._flying  # a finished answer is not thrown away
        assert comparer.result("a", "b") == correlate(mm["a"], mm["b"], SR)
    finally:
        comparer.close()
