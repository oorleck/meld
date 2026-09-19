from synth import phone, synth_music

from meld.sync import correlate

SR = 16000


def test_recovers_positive_lag():
    music = synth_music(90, SR)
    a = phone(music, SR, 0, 60, seed=1)
    b = phone(music, SR, 20.3, 80, seed=2, gain=3.0)  # clipped, louder, starts 20.3 s later
    lag, z = correlate(a, b, SR)
    assert abs(lag - 20.3) < 0.002
    assert z > 10


def test_recovers_negative_lag():
    music = synth_music(90, SR)
    a = phone(music, SR, 30, 80, seed=3)
    b = phone(music, SR, 5.55, 45, seed=4)
    lag, z = correlate(a, b, SR)
    assert abs(lag - (5.55 - 30)) < 0.002
    assert z > 10


def test_short_overlap_still_found():
    music = synth_music(90, SR)
    a = phone(music, SR, 0, 40, seed=5)
    b = phone(music, SR, 32, 90, seed=6)  # 8 s overlap
    lag, z = correlate(a, b, SR)
    assert abs(lag - 32) < 0.002
    assert z > 8


def test_unrelated_audio_is_not_a_match():
    a = phone(synth_music(60, SR, seed=1), SR, 0, 60, seed=1)
    b = phone(synth_music(60, SR, seed=2), SR, 0, 60, seed=2)
    _, z = correlate(a, b, SR)
    assert z < 8


def test_format_duration():
    from meld.sync import format_duration

    assert format_duration(4.4) == "4s"
    assert format_duration(59.6) == "1m 00s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3 * 3600 + 7 * 60 + 12) == "3h 07m"


def test_estimate_remaining():
    from meld.sync import estimate_remaining

    # nothing placed yet: only the worst case is known (3 clips, each allowed 5 comparisons, 1 already used)
    expected, worst = estimate_remaining([1, 0, 0], 5, 2.0, None)
    assert expected is None and worst == (4 + 5 + 5) * 2.0

    # placed clips needed 1.5 comparisons on average: 3 pending clips -> 4.5 comparisons of 2 s
    expected, worst = estimate_remaining([1, 0, 0], 5, 2.0, 1.5)
    assert expected == 3 * 1.5 * 2.0 and worst == 28.0

    # the expected time can never exceed the worst case
    expected, worst = estimate_remaining([4, 4], 5, 1.0, 10.0)
    assert expected == worst == 2.0

    # nothing pending: nothing left
    assert estimate_remaining([], 5, 2.0, 1.5) == (0.0, 0.0)
