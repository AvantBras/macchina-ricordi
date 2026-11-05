from __future__ import annotations

import numpy as np
import pytest

from macchina_ricordi.scoring.reward import (
    score_arrhythmic_percussive_waveform,
)


def _make_click_track(
    onset_times: np.ndarray,
    *,
    duration: float = 8.0,
    sr: int = 32_000,
) -> np.ndarray:
    y = np.zeros(int(duration * sr), dtype=np.float32)
    click = np.hanning(64).astype(np.float32)
    for onset in onset_times:
        idx = int(onset * sr)
        if 0 <= idx < len(y) - len(click):
            y[idx : idx + len(click)] += click
    return y


def test_arrhythmic_clicks_score_higher_than_regular_grid() -> None:
    rng = np.random.default_rng(42)
    base = np.arange(0.5, 7.5, 0.4)
    jittered = np.sort(base + rng.uniform(-0.13, 0.13, size=base.shape[0]))
    jittered = np.clip(jittered, 0.2, 7.8)

    regular = _make_click_track(base)
    arrhythmic = _make_click_track(jittered)

    regular_score = score_arrhythmic_percussive_waveform(regular)
    arrhythmic_score = score_arrhythmic_percussive_waveform(arrhythmic)

    assert arrhythmic_score.tempo_instability > regular_score.tempo_instability
    assert arrhythmic_score.score > regular_score.score
    assert regular_score.dominant_tempo_consistency >= arrhythmic_score.dominant_tempo_consistency
    assert regular_score.tempogram_peak_ratio >= arrhythmic_score.tempogram_peak_ratio
    assert arrhythmic_score.percussive_energy_ratio > 0.0
    assert regular_score.percussive_onset_density > 0.0


def test_silence_scores_zero() -> None:
    silence = np.zeros(32_000, dtype=np.float32)
    result = score_arrhythmic_percussive_waveform(silence)
    assert result.score == 0.0
    assert result.tempo_instability == 0.0
    assert result.percussive_energy_ratio == 0.0
    assert result.percussive_onset_density == 0.0
    assert result.silence_fraction == 1.0
    assert result.spectral_change_mean == 0.0


def test_score_waveform_rejects_non_mono_input() -> None:
    with pytest.raises(ValueError):
        score_arrhythmic_percussive_waveform(np.zeros((2, 16), dtype=np.float32))
