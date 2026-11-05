from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from macchina_ricordi.scoring.arc import compute_arc_coherence
from macchina_ricordi.scoring.live import LiveRoundComponents, score_live_round
from macchina_ricordi.scoring.reward import score_arrhythmic_percussive
from macchina_ricordi.scoring.similarity import clear_calibration

SR = 16_000


@pytest.fixture(autouse=True)
def _no_calibration() -> None:
    clear_calibration()


def _round_wav(tmp_path: Path, *, seconds: float = 8.0, name: str = "round.wav") -> Path:
    """A clip with onsets, harmonic movement and a quiet patch — every field non-trivial."""
    rng = np.random.default_rng(7)
    quarter = seconds / 4
    parts = []
    for freq in (261.63, 329.63, 392.00, 440.00):
        t = np.linspace(0.0, quarter, int(quarter * SR), endpoint=False)
        tone = 0.4 * np.sin(2 * np.pi * freq * t)
        clicks = np.zeros_like(tone)
        clicks[:: int(SR * 0.17)] = 1.0
        parts.append(tone + 0.3 * clicks + 0.01 * rng.standard_normal(tone.size))
    waveform = np.concatenate(parts).astype(np.float32)
    waveform[int(2.0 * SR) : int(2.6 * SR)] = 0.0  # a gap, so silence is non-zero
    path = tmp_path / name
    sf.write(str(path), waveform, SR)
    return path


# ── the point of the module: same numbers, fewer of them ──────────────


@pytest.mark.parametrize("echo", [None, 3.0])
def test_live_fields_match_the_full_scorers_exactly(tmp_path: Path, echo: float | None) -> None:
    path = _round_wav(tmp_path)
    lean = score_live_round(path, prompt_duration_seconds=echo).to_dict()
    full = {
        **score_arrhythmic_percussive(path).to_dict(),
        **compute_arc_coherence(path, prompt_duration_seconds=echo).to_dict(),
    }
    shared = set(lean) & set(full)
    # If this ever shrinks, the lean path has started measuring something else.
    assert shared == set(lean)
    for field in sorted(shared):
        assert lean[field] == pytest.approx(full[field], abs=0.0, rel=0.0), field


def test_live_drops_the_redundant_fields(tmp_path: Path) -> None:
    payload = score_live_round(_round_wav(tmp_path)).to_dict()
    for dropped in ("chroma_drift", "score", "tempo_instability", "arc_coherence_gated"):
        assert dropped not in payload
    # What the panel draws and what candidate selection ranks by must be there.
    for kept in (
        "arc_coherence_score",
        "timbral_coherence",
        "silence_fraction",
        "onset_density",
        "percussive_energy_ratio",
    ):
        assert kept in payload


def test_prompt_echo_is_trimmed_from_the_arc_half_only(tmp_path: Path) -> None:
    path = _round_wav(tmp_path)
    whole = score_live_round(path)
    trimmed = score_live_round(path, prompt_duration_seconds=3.0)
    assert trimmed.arc_tail_seconds == pytest.approx(whole.arc_tail_seconds - 3.0, abs=0.05)
    # Rate, force and silence deliberately still measure the whole round.
    assert trimmed.onset_density == whole.onset_density
    assert trimmed.silence_fraction == whole.silence_fraction
    assert trimmed.duration_seconds == whole.duration_seconds


# ── degenerate audio must never take a performance down ───────────────


def test_empty_audio_returns_zeroed_components(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    sf.write(str(path), np.zeros(0, dtype=np.float32), SR)
    assert score_live_round(path) == LiveRoundComponents()


def test_very_short_audio_scores_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "blip.wav"
    sf.write(str(path), np.zeros(int(0.05 * SR), dtype=np.float32), SR)
    components = score_live_round(path)
    assert components.arc_coherence_score == 0.0
    assert components.duration_seconds > 0.0


def test_silent_round_reports_full_silence(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros(int(4.0 * SR), dtype=np.float32), SR)
    components = score_live_round(path)
    assert components.silence_fraction == pytest.approx(1.0)
    assert components.arc_coherence_score == 0.0
