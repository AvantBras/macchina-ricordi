from __future__ import annotations

import numpy as np
import pytest

from macchina_ricordi.scoring.features import (
    analyze_onsets,
    validate_mono_waveform,
)


def _make_click_track(
    onset_times: np.ndarray,
    *,
    duration: float = 3.0,
    sr: int = 16_000,
) -> np.ndarray:
    waveform = np.zeros(int(duration * sr), dtype=np.float32)
    click = np.hanning(64).astype(np.float32)
    for onset in onset_times:
        idx = int(onset * sr)
        if 0 <= idx < len(waveform) - len(click):
            waveform[idx : idx + len(click)] += click
    return waveform


def test_analyze_onsets_reports_click_track_features() -> None:
    waveform = _make_click_track(np.arange(0.3, 2.6, 0.35))
    features = analyze_onsets(waveform, sr=16_000)

    assert features.count > 0
    assert features.density > 0.0
    assert features.instability >= 0.0
    assert features.count == len(features.frames) == len(features.times)
    assert features.strength_p90 >= features.strength_mean


def test_validate_mono_waveform_rejects_non_mono_input() -> None:
    with pytest.raises(ValueError):
        validate_mono_waveform(np.zeros((2, 1_000), dtype=np.float32))
