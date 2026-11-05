"""CPU tests for the duet renderer's pure audio helpers."""

from __future__ import annotations

import numpy as np
import pytest

from macchina_ricordi.instrument.duet import (
    crossfade_append,
    equal_power_pan,
    mix_tails,
    pick_least_silent,
    silence_fraction,
    stereo_master,
)

SR = 16_000


def _sine(seconds: float, amplitude: float = 0.5, hz: float = 220.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * SR), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_silence_fraction_extremes() -> None:
    assert silence_fraction(np.zeros(SR, dtype=np.float32)) == 1.0
    assert silence_fraction(_sine(1.0)) < 0.05


def test_silence_fraction_mixed_signal() -> None:
    half_silent = np.concatenate([np.zeros(SR, dtype=np.float32), _sine(1.0)])
    fraction = silence_fraction(half_silent)
    assert 0.35 < fraction < 0.65


def test_pick_least_silent() -> None:
    batch = np.stack([np.zeros(SR, dtype=np.float32), _sine(1.0)])
    winner, fractions = pick_least_silent(batch)
    assert winner == 1
    assert fractions[0] == 1.0


def test_mix_tails_weights_and_alignment() -> None:
    own = np.full(SR, 0.4, dtype=np.float32)
    partner = np.full(SR // 2, -0.4, dtype=np.float32)  # shorter -> front-padded
    blend = mix_tails(own, partner, listen=0.5)
    assert blend.size == own.size
    # front half: only own at half weight; back half: cancels to ~0
    assert blend[: SR // 4].mean() == pytest.approx(0.2, abs=1e-6)
    assert np.abs(blend[-SR // 4 :]).max() == pytest.approx(0.0, abs=1e-6)


def test_mix_tails_validates_listen() -> None:
    own = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="listen"):
        mix_tails(own, own, listen=1.5)


def test_crossfade_append_preserves_correlated_signal() -> None:
    a = np.full(SR, 0.5, dtype=np.float32)
    b = np.full(SR, 0.5, dtype=np.float32)
    joined = crossfade_append(a, b, crossfade_samples=400)
    assert joined.size == a.size + b.size - 400
    # linear crossfade of identical constants stays constant
    assert np.abs(joined - 0.5).max() < 1e-6


def test_crossfade_append_empty_track_is_plain_concat() -> None:
    b = _sine(0.5)
    joined = crossfade_append(np.zeros(0, dtype=np.float32), b, crossfade_samples=400)
    assert joined.size == b.size
    np.testing.assert_allclose(joined, b)


def test_equal_power_pan_positions() -> None:
    mono = np.ones(64, dtype=np.float32)
    center = equal_power_pan(mono, 0.0)
    assert center.shape == (2, 64)
    np.testing.assert_allclose(center[0], center[1])
    np.testing.assert_allclose(center[0], np.full(64, np.sqrt(0.5)), rtol=1e-6)
    left = equal_power_pan(mono, -1.0)
    assert left[0].max() == pytest.approx(1.0)
    assert np.abs(left[1]).max() == pytest.approx(0.0, abs=1e-7)


def test_stereo_master_pads_gains_and_respects_ceiling() -> None:
    a = np.full(SR, 0.9, dtype=np.float32)
    b = np.full(SR // 2, 0.9, dtype=np.float32)  # shorter stem -> padded
    master = stereo_master(a, b, gain_a=2.0, gain_b=2.0, ceiling=0.89)
    assert master.shape == (2, SR)
    assert np.abs(master).max() == pytest.approx(0.89, rel=1e-5)
