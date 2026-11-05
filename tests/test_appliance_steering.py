"""Steering the past: the transforms, the live memory and stride, pins, and what reaches the model.

A fake backend that can encode records what it is fed, so each test can check
the one thing that matters about steering: the model continues from a changed
past, and nobody hears that changed past again.
"""

from __future__ import annotations

import numpy as np
import pytest

from appliance.engine import Engine
from appliance.player import Knobs, Player
from appliance.spec import ModelSpec
from appliance.steering import Tilt, apply_tilt, band_gain_curve, blend, end_aligned, resample_to

SR = 3200  # small, so tests are quick: 64 samples a frame at 50 Hz


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(
        dim=16,
        num_heads=2,
        num_layers=2,
        n_q=4,
        card=32,
        hidden_scale=2.0,
        layer_scale=1e-4,
        delays=[0, 1, 2, 3],
        frame_rate=50.0,
        sample_rate=SR,
        channels=1,
        segment_duration=20.0,
        name="stub",
    )


class Recorder:
    """Random logits, a decoder that tells codes apart, and an encoder that leaves a mark."""

    MARK = 7

    def __init__(self, spec, *, can_encode=True):
        self.spec = spec
        self.hop = int(spec.sample_rate / spec.frame_rate)
        self.can_encode = can_encode
        self.primes: list[np.ndarray] = []
        self.decoded: list[np.ndarray] = []
        self.encoded: list[np.ndarray] = []

    def start(self, batch, horizon):
        return 0

    def step(self, tokens, offset, state):
        if tokens.shape[-1] > 1:
            self.primes.append(tokens.copy())
        logits = np.random.default_rng(offset + 1000 * len(self.primes)).standard_normal(
            tokens.shape + (self.spec.card,)
        )
        logits[..., self.MARK] = -50.0  # the model never writes the mark itself
        return logits.astype(np.float32), state

    def decode(self, codes):
        self.decoded.append(codes.copy())
        level = 0.1 + 0.5 * codes[0].mean(axis=0) / self.spec.card
        return np.repeat(level, self.hop).astype(np.float32)

    def encode(self, audio):
        self.encoded.append(audio.copy())
        frames = audio.size // self.hop
        return np.full((1, self.spec.n_q, frames), self.MARK, dtype=np.int64)


def _player(spec, **kw):
    backend = kw.pop("backend", None) or Recorder(spec)
    return Player(backend, context_seconds=1.0, round_seconds=0.8, seed=0, **kw)


# ---- the transforms ----------------------------------------------------------------


@pytest.mark.parametrize("tilt", [(6, 0, 0), (-12, 3, 9), (4.5, 4.5, 4.5)])
def test_tilt_curve_is_the_streams(tilt):
    from macchina_ricordi.instrument.prompt_transform import PromptTilt
    from macchina_ricordi.instrument.prompt_transform import band_gain_curve as stream_curve

    f = np.linspace(0, 16000, 1025)
    assert np.array_equal(band_gain_curve(f, Tilt(*tilt)), stream_curve(f, PromptTilt(*tilt)))


def test_tilt_moves_each_band_by_its_setting():
    sr = 32000
    t = np.arange(sr * 2) / sr
    x = sum(0.2 * np.sin(2 * np.pi * hz * t) for hz in (100, 1000, 8000)).astype(np.float32)

    def level(sig, hz):
        spectrum = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
        k = int(round(hz * sig.size / sr))
        return 20 * np.log10(spectrum[k - 2 : k + 3].max())

    y = apply_tilt(x, sr, Tilt(6, 0, -6))
    assert [round(level(y, hz) - level(x, hz), 1) for hz in (100, 1000, 8000)] == [6.0, 0.0, -6.0]
    assert np.array_equal(apply_tilt(x, sr, Tilt()), x)
    assert np.abs(apply_tilt(x, sr, Tilt(3, 3, 3))).max() == pytest.approx(np.abs(x).max() * 10 ** (3 / 20), rel=1e-3)


@pytest.mark.parametrize("rate", [0.9, 1.0, 1.1])
def test_tape_speed_lands_exactly_and_shifts_pitch(rate):
    sr = 32000
    source = np.sin(2 * np.pi * 1000 * np.arange(int(rate * sr)) / sr).astype(np.float32)
    out = resample_to(source, sr)
    assert out.size == sr
    peak = np.argmax(np.abs(np.fft.rfft(out * np.hanning(out.size)))) * sr / out.size
    assert peak == pytest.approx(1000 * rate, abs=2)


def test_end_alignment_and_blend_keep_the_end():
    assert end_aligned(np.arange(5.0), 3).tolist() == [2, 3, 4]
    assert end_aligned(np.arange(2.0), 4).tolist() == [0, 0, 0, 1]
    mixed = blend(np.full(6, 0.9, np.float32), np.ones(2, np.float32), 0.6)
    assert mixed[-1] == pytest.approx(0.96) and mixed[0] == pytest.approx(0.36)
    assert np.abs(blend(np.ones(4, np.float32), np.ones(4, np.float32) * 3, 0.5)).max() <= 0.99 + 1e-6


# ---- knobs and the horizon ----------------------------------------------------------------


def test_knobs_clamp_the_new_controls():
    k = Knobs(tau_seconds=9, round_seconds=0.2, tilt_low_db=40, tilt_high_db=-40, prompt_rate=2, recap_every=99)
    k = k.clamped()
    assert (k.tau_seconds, k.round_seconds, k.tilt_low_db, k.tilt_high_db, k.prompt_rate, k.recap_every) == (
        5.0,
        1.0,
        12.0,
        -12.0,
        1.1,
        16,
    )
    assert Knobs().clamped().tau_seconds is None and not Knobs().steers
    assert Knobs(tilt_mid_db=1.5).steers and Knobs(prompt_rate=0.95).steers


def test_memory_and_stride_stay_inside_the_models_horizon(spec):
    from dataclasses import replace

    p = _player(spec)
    assert p.frames_for(Knobs(tau_seconds=5, round_seconds=10)) == (250, 500)
    short = _player(replace(spec, segment_duration=10.0), backend=Recorder(replace(spec, segment_duration=10.0)))
    context, fresh = short.frames_for(Knobs(tau_seconds=5, round_seconds=10))
    assert context == 250 and context + fresh == short.horizon_frames


# ---- the loop ----------------------------------------------------------------------------------


def test_memory_and_stride_are_live(spec):
    p = _player(spec)
    p.round(Knobs(top_k=8))
    chunk, info = p.round(Knobs(top_k=8, round_seconds=2.0))
    assert info["round_s"] == 2.0 and len(chunk) == 100 * p.samples_per_frame
    p.round(Knobs(top_k=8, tau_seconds=1.5, round_seconds=2.0))
    p.round(Knobs(top_k=8, tau_seconds=0.5, round_seconds=2.0))
    assert p._history.shape[-1] == 25


def test_nothing_is_encoded_while_every_control_is_neutral(spec):
    p = _player(spec)
    for _ in range(4):
        _, info = p.round(Knobs(top_k=8))
        assert not info["steered"]
    assert p.backend.encoded == []


def test_a_steered_past_is_what_the_model_hears_and_not_what_is_played(spec):
    p = _player(spec)
    p.round(Knobs(top_k=8))
    played = p._history.copy()
    _, info = p.round(Knobs(top_k=8, tilt_low_db=6.0))
    assert info["steered"] and len(p.backend.encoded[-1]) == played.shape[-1] * p.samples_per_frame
    prime = p.backend.primes[-1]
    assert (prime[0, 0, 1 : 1 + played.shape[-1]] == Recorder.MARK).all(), "the model was not fed the steered past"
    decoded = p.backend.decoded[-1]
    assert np.array_equal(decoded[:, :, : played.shape[-1]], played), "the steered past was played again"


def test_tape_speed_reports_what_it_could_actually_do(spec):
    p = _player(spec)
    p.round(Knobs(top_k=8))
    _, info = p.round(Knobs(top_k=8, prompt_rate=1.1))
    # the first round's past is only as long as the first round, so a speed-up
    # cannot reach 1.1x further back than exists
    assert info["rate_applied"] <= 1.1 and info["steered"]
    _, info = p.round(Knobs(top_k=8, prompt_rate=0.9))
    assert info["rate_applied"] == pytest.approx(0.9, abs=0.01)


def test_bringing_a_pin_back_happens_once(spec):
    p = _player(spec)
    p.round(Knobs(top_k=8))
    p.pin(np.full(3200, 0.5, np.float32))
    p.bring_back("heavy")
    _, info = p.round(Knobs(top_k=8))
    assert info["motif"] == "bring back" and info["motif_mix"] == 0.6
    _, info = p.round(Knobs(top_k=8))
    assert not info["steered"]


def test_recall_every_n_rounds(spec):
    p = _player(spec)
    p.pin(np.full(3200, 0.5, np.float32))
    motifs = [p.round(Knobs(top_k=8, recap_every=3))[1].get("motif") for _ in range(7)]
    assert motifs == [None, None, "recap", None, None, "recap", None]


def test_without_an_encoder_steering_is_skipped_not_fatal(spec):
    p = _player(spec, backend=Recorder(spec, can_encode=False))
    p.round(Knobs(top_k=8))
    _, info = p.round(Knobs(top_k=8, tilt_high_db=-6))
    assert not info["steered"] and "no encoder" in info["steer_skipped"]


def test_pins_survive_a_reset(spec):
    p = _player(spec)
    p.pin(np.ones(10, np.float32))
    p.reset()
    assert len(p.motifs) == 1


# ---- pinning what is audible ----------------------------------------------------------------------


def test_pin_takes_what_is_playing_not_what_was_generated_last(spec):
    engine = Engine(_player(spec), knobs=Knobs(top_k=8), prefill_rounds=3)
    engine.prefill()
    first_len = len(engine._produced[0][1])
    engine.pull(first_len // 2)  # halfway through the first round
    engine.pin(playing=True, at_least_seconds=0.1)
    assert np.array_equal(engine.player.motifs[-1], engine._produced[0][1])


def test_pin_with_nothing_played_says_so(spec):
    engine = Engine(_player(spec))
    with pytest.raises(ValueError, match="nothing has played"):
        engine.pin()
