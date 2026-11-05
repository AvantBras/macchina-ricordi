from __future__ import annotations

import numpy as np
import pytest

from macchina_ricordi.scoring.structure import (
    DEFAULT_STRUCTURE_WEIGHTS,
    StructureComponents,
    _checkerboard_kernel,
    _pick_peaks,
    _resolve_window_count,
    _window_slices,
    active_structure_weights,
    clear_structure_weights,
    compute_structure,
    compute_structure_waveform,
    load_structure_weights,
    silence_gate,
)

SR = 16_000


def _tone(frequency: float, seconds: float, *, sr: int = SR, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _noise(seconds: float, *, sr: int = SR, amplitude: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(sr * seconds))).astype(np.float32)


def _drone(seconds: float = 12.0) -> np.ndarray:
    return _tone(220.0, seconds)


# Distinct pitch classes: octave-related tones share a chroma vector and would make an
# "unrelated" fixture accidentally recurrent.
_A2, _E3, _C4 = 110.0, 164.8, 261.6


def _sectional(seconds: float = 12.0) -> np.ndarray:
    """A B A — returns to its opening material, so it should score on ssm_return."""
    third = seconds / 3.0
    return np.concatenate([_tone(_A2, third), _tone(_E3, third), _tone(_A2, third)])


def _non_returning(seconds: float = 12.0) -> np.ndarray:
    """A B C — the same amount of change as ``_sectional``, but nothing comes back."""
    third = seconds / 3.0
    return np.concatenate([_tone(_A2, third), _tone(_E3, third), _tone(_C4, third)])


def _lurching(seconds: float = 12.0) -> np.ndarray:
    """Six materials, none of which ever recur: change without structure.

    Every chunk is a distinct timbre — repeating *noise* here would be a mistake, since
    two different noise seeds are near-identical to any spectral feature and would make
    this fixture genuinely recurrent.
    """
    chunk = seconds / 6.0
    parts = [
        _tone(110.0, chunk),
        _square(180.0, chunk),
        _tone(880.0, chunk),
        _square(300.0, chunk, duty=0.15),
        _noise(chunk, seed=1),
        _tone(440.0, chunk, amplitude=0.1),
    ]
    return np.concatenate(parts)


def _square(frequency: float, seconds: float, *, sr: int = SR, duty: float = 0.5, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    phase = (t * frequency) % 1.0
    return (amplitude * np.where(phase < duty, 1.0, -1.0)).astype(np.float32)


# ── windowing ─────────────────────────────────────────────────────────


def test_window_slices_overlap_and_cover():
    slices = _window_slices(100, 8)
    assert len(slices) >= 4
    assert slices[0][0] == 0
    assert slices[-1][1] == 100
    # 50 % overlap: each window starts before the previous one ends.
    for (start_a, stop_a), (start_b, _stop_b) in zip(slices, slices[1:], strict=False):
        assert start_b < stop_a
        assert start_b > start_a


def test_window_slices_degenerate_input():
    assert _window_slices(0, 8) == []
    assert _window_slices(3, 8) == [(0, 3)]


def test_resolve_window_count_shrinks_for_short_audio():
    assert _resolve_window_count(20.0, 8) == 8
    assert _resolve_window_count(2.0, 8) == 3
    assert _resolve_window_count(0.4, 8) == 0


# ── silence gate ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.0, 1.0), (0.30, 1.0), (0.60, 0.0), (0.9, 0.0), (float("nan"), 0.0)],
)
def test_silence_gate_endpoints(fraction, expected):
    assert silence_gate(fraction) == pytest.approx(expected)


def test_silence_gate_ramps_between_thresholds():
    assert 0.0 < silence_gate(0.45) < 1.0
    assert silence_gate(0.35) > silence_gate(0.55)


def test_mostly_silent_audio_scores_zero():
    waveform = np.concatenate([_tone(220.0, 2.0), np.zeros(int(SR * 8.0), dtype=np.float32)])
    components = compute_structure_waveform(waveform, SR)
    assert components.silence_gate == pytest.approx(0.0)
    assert components.structure_score == pytest.approx(0.0)


# ── development axes ──────────────────────────────────────────────────


def test_drone_has_low_development():
    components = compute_structure_waveform(_drone(), SR)
    assert components.dev_top2 < 0.05
    assert components.dev_harmonic < 0.05


def test_pitch_movement_shows_on_the_harmonic_axis():
    moving = np.concatenate([_tone(freq, 2.0) for freq in (220.0, 277.0, 330.0, 415.0, 494.0, 587.0)])
    components = compute_structure_waveform(moving, SR)
    drone = compute_structure_waveform(_drone(), SR)
    assert components.dev_harmonic > drone.dev_harmonic


def test_dynamics_axis_sees_a_fade_that_chroma_cannot():
    steady = _tone(220.0, 12.0)
    envelope = np.linspace(1.0, 0.05, steady.size).astype(np.float32)
    fading = steady * envelope
    faded = compute_structure_waveform(fading, SR)
    flat = compute_structure_waveform(steady, SR)
    assert faded.dev_dynamics > flat.dev_dynamics
    # The whole point: the harmonic axis is blind to this, the composite is not.
    assert faded.dev_harmonic == pytest.approx(flat.dev_harmonic, abs=0.05)
    assert faded.dev_top2 > flat.dev_top2


def test_timbre_axis_separates_tone_from_noise():
    alternating = np.concatenate([_tone(220.0, 2.0), _noise(2.0, seed=4), _tone(220.0, 2.0), _noise(2.0, seed=5)])
    components = compute_structure_waveform(alternating, SR)
    assert components.dev_timbre > compute_structure_waveform(_drone(8.0), SR).dev_timbre


def test_dev_top2_is_the_mean_of_the_two_strongest_axes():
    components = compute_structure_waveform(_lurching(), SR)
    axes = sorted(
        [
            components.dev_harmonic,
            components.dev_timbre,
            components.dev_dynamics,
            components.dev_rhythm,
        ],
        reverse=True,
    )
    assert components.dev_top2 == pytest.approx(float(np.mean(axes[:2])), abs=1e-6)


# ── structure ─────────────────────────────────────────────────────────


def test_returning_material_scores_above_non_returning():
    """A-B-A against A-B-C: identical amount of change, only recurrence differs."""
    returning = compute_structure_waveform(_sectional(), SR)
    onward = compute_structure_waveform(_non_returning(), SR)
    assert returning.ssm_return > 0.2
    assert onward.ssm_return < 0.1
    assert returning.ssm_return > onward.ssm_return


def test_return_metric_is_not_just_inverse_similarity():
    """The regression that motivated the lag profile: A-B-A has the *higher* mean."""
    returning = compute_structure_waveform(_sectional(), SR)
    onward = compute_structure_waveform(_non_returning(), SR)
    assert returning.ssm_mean > onward.ssm_mean
    assert returning.ssm_return > onward.ssm_return


def test_return_from_lag_profile_detrends_monotone_decay():
    from macchina_ricordi.scoring.structure import _return_from_lag_profile

    decaying = np.linspace(0.9, 0.2, 20)
    assert _return_from_lag_profile(decaying) < 0.05
    recovering = np.concatenate([np.linspace(0.9, 0.2, 10), np.linspace(0.2, 0.8, 10)])
    assert _return_from_lag_profile(recovering) > 0.4
    assert _return_from_lag_profile(np.full(20, 0.7)) < 0.05
    assert _return_from_lag_profile(np.asarray([0.5, 0.4])) == 0.0


def test_drone_has_no_return_bonus():
    """A drone matches everything equally; that is coherence, not structure."""
    components = compute_structure_waveform(_drone(), SR)
    assert components.ssm_mean > 0.9
    assert components.ssm_return < 0.1


def test_sectional_audio_is_more_novel_than_a_drone():
    sectional = compute_structure_waveform(_sectional(), SR)
    drone = compute_structure_waveform(_drone(), SR)
    assert sectional.ssm_novelty > 0.2
    assert drone.ssm_novelty < 0.05
    assert sectional.boundaries_per_minute > drone.boundaries_per_minute


def test_drone_novelty_does_not_saturate_on_numerical_noise():
    """A relative shape statistic over the novelty curve reads ~1.0 here. Absolute does not."""
    assert compute_structure_waveform(_drone(), SR).ssm_novelty == pytest.approx(0.0, abs=0.02)


def test_lurching_audio_is_incoherent():
    """Change without a coherent voice: high development, low global similarity."""
    lurching = compute_structure_waveform(_lurching(), SR)
    drone = compute_structure_waveform(_drone(), SR)
    assert lurching.ssm_mean < 0.5
    assert drone.ssm_mean > 0.9
    assert lurching.dev_top2 > drone.dev_top2


def test_checkerboard_kernel_is_antisymmetric():
    kernel = _checkerboard_kernel(4)
    assert kernel.shape == (9, 9)
    assert float(np.sum(kernel)) == pytest.approx(0.0, abs=1e-9)
    assert kernel[0, 0] > 0 and kernel[0, -1] < 0


def test_pick_peaks_respects_minimum_distance():
    curve = np.zeros(40)
    curve[10] = 5.0
    curve[12] = 4.0
    curve[30] = 6.0
    peaks = _pick_peaks(curve, min_distance=5)
    assert peaks == [10, 30]


def test_flat_curve_has_no_peaks():
    assert _pick_peaks(np.full(30, 0.5), min_distance=3) == []


# ── plumbing ──────────────────────────────────────────────────────────


def test_short_audio_returns_zeroed_components():
    components = compute_structure_waveform(np.zeros(100, dtype=np.float32), SR)
    assert components == StructureComponents(structure_seconds=components.structure_seconds)


def test_rejects_stereo():
    with pytest.raises(ValueError):
        compute_structure_waveform(np.zeros((2, SR), dtype=np.float32), SR)


def test_compute_structure_trims_prompt_echo(tmp_path):
    import soundfile as sf

    path = tmp_path / "clip.wav"
    sf.write(path, np.concatenate([_noise(4.0, seed=7), _tone(220.0, 8.0)]), SR)
    full = compute_structure(path)
    trimmed = compute_structure(path, prompt_duration_seconds=4.0)
    assert trimmed.structure_seconds == pytest.approx(8.0, abs=0.1)
    assert full.structure_seconds == pytest.approx(12.0, abs=0.1)
    assert trimmed.dev_timbre < full.dev_timbre


def test_compute_structure_missing_file_raises():
    """Degenerate *audio* returns zeros; a missing file is a caller bug and must surface."""
    with pytest.raises(FileNotFoundError):
        compute_structure("/nonexistent/clip.wav")


def test_fieldnames_match_dataclass():
    names = StructureComponents.fieldnames()
    assert "structure_score" in names
    assert set(names) == set(StructureComponents().to_dict())


def test_weights_are_loadable_and_clearable(tmp_path):
    import json

    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"structure_weights": {"development": 0.7, "novelty": 0.2, "return": 0.1}}))
    try:
        assert active_structure_weights() == DEFAULT_STRUCTURE_WEIGHTS
        loaded = load_structure_weights(path)
        assert loaded["development"] == pytest.approx(0.7)
        assert active_structure_weights()["development"] == pytest.approx(0.7)
    finally:
        clear_structure_weights()
    assert active_structure_weights() == DEFAULT_STRUCTURE_WEIGHTS


def test_load_weights_rejects_unrelated_json(tmp_path):
    import json

    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"unrelated": 1}))
    with pytest.raises(ValueError):
        load_structure_weights(path)


def test_explicit_weights_change_the_composite():
    waveform = _sectional()
    novelty_heavy = compute_structure_waveform(
        waveform, SR, weights={"development": 0.0, "novelty": 1.0, "return": 0.0}
    )
    development_heavy = compute_structure_waveform(
        waveform, SR, weights={"development": 1.0, "novelty": 0.0, "return": 0.0}
    )
    assert novelty_heavy.structure_score != pytest.approx(development_heavy.structure_score)
