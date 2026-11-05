from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.listening.describe import (
    _DIMENSIONS,
    AudioDescription,
    _lexicalize,
    describe_audio,
    describe_waveform,
    diff_audio,
    diff_waveform,
)


def _make_noise(*, duration: float = 3.0, sr: int = 16_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, int(duration * sr)).astype(np.float32)


def _make_click_track(
    onset_times: np.ndarray,
    *,
    duration: float = 5.0,
    sr: int = 16_000,
) -> np.ndarray:
    y = np.zeros(int(duration * sr), dtype=np.float32)
    click = np.hanning(64).astype(np.float32)
    for onset in onset_times:
        idx = int(onset * sr)
        if 0 <= idx < len(y) - len(click):
            y[idx : idx + len(click)] += click
    return y


# ── lexicalize ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (-999.0, _DIMENSIONS[0].labels[0]),  # below the first threshold
        (999.0, _DIMENSIONS[0].labels[-1]),  # above the last threshold
        (0.07, "moderate"),  # loudness thresholds: [0.01, 0.05, 0.12, 0.25, 0.50]
    ],
)
def test_lexicalize_maps_a_value_onto_its_band(value: float, label: str) -> None:
    assert _lexicalize(value, _DIMENSIONS[0]) == label


# ── describe ───────────────────────────────────────────────────────────


class TestDescribeWaveform:
    def test_noise_returns_a_non_empty_label_per_field(self) -> None:
        wav = _make_noise(duration=2.0)
        desc = describe_waveform(wav, sr=16_000)
        assert isinstance(desc, AudioDescription)
        for field in AudioDescription.fieldnames():
            assert isinstance(getattr(desc, field), str)
            assert len(getattr(desc, field)) > 0
        assert set(desc.to_dict()) == set(AudioDescription.fieldnames())

    def test_silence_returns_minimal_labels(self) -> None:
        wav = np.zeros(8000, dtype=np.float32)
        desc = describe_waveform(wav, sr=16_000)
        assert desc.loudness == "whisper"
        assert desc.density == "sparse"
        assert desc.pulse == "still"

    def test_dense_clicks(self) -> None:
        wav = _make_click_track(np.arange(0.1, 3.0, 0.1), duration=3.0, sr=16_000)
        desc = describe_waveform(wav, sr=16_000)
        # Dense clicks should have high density, sharp attack
        assert desc.density in ("dense", "saturated")
        assert desc.attack in ("sharp", "needle-like")

    def test_rejects_non_mono(self) -> None:
        stereo = np.zeros((2, 1000), dtype=np.float32)
        with pytest.raises(ValueError):
            describe_waveform(stereo)


class TestDescribeAudio:
    def test_file_api(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        wav = _make_noise(duration=1.0)
        path = tmp_path / "test.wav"
        sf.write(str(path), wav, 16_000)
        desc = describe_audio(path, sr=16_000)
        assert isinstance(desc, AudioDescription)


# ── diff ───────────────────────────────────────────────────────────────


class TestDiffWaveform:
    def test_identical_returns_no_shift(self) -> None:
        wav = _make_noise(duration=2.0)
        result = diff_waveform(wav, wav, sr=16_000)
        assert "No perceptual shift" in result

    def test_silence_vs_noise_names_the_shift(self) -> None:
        silence = np.zeros(16000, dtype=np.float32)
        noise = _make_noise(duration=1.0, seed=99)
        result = diff_waveform(silence, noise, sr=16_000)
        assert "shifted from" in result


class TestDiffAudio:
    def test_identical_file(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        wav = _make_noise(duration=1.0)
        path = tmp_path / "test.wav"
        sf.write(str(path), wav, 16_000)
        result = diff_audio(path, path, sr=16_000)
        assert "No perceptual shift" in result

    def test_different_files(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        noise = _make_noise(duration=1.0, seed=1)
        clicks = _make_click_track(np.arange(0.5, 1.5, 0.2), duration=2.0)
        sf.write(str(tmp_path / "noise.wav"), noise, 16_000)
        sf.write(str(tmp_path / "clicks.wav"), clicks, 16_000)
        result = diff_audio(tmp_path / "noise.wav", tmp_path / "clicks.wav", sr=16_000)
        assert "shifted from" in result


# ── dataclass ──────────────────────────────────────────────────────────


def test_audio_description_to_dict() -> None:
    desc = AudioDescription(
        loudness="forceful",
        brightness="bright",
        attack="crisp",
        noisiness="textured",
        density="dense",
        pulse="limping",
        resonance="room",
    )
    d = desc.to_dict()
    assert d["loudness"] == "forceful"
    assert d["resonance"] == "room"


def test_audio_description_sentence_names_every_label() -> None:
    desc = AudioDescription(
        loudness="forceful",
        brightness="bright",
        attack="crisp",
        noisiness="textured",
        density="dense",
        pulse="limping",
        resonance="room",
    )
    sent = desc.to_sentence()
    assert sent.startswith("A ")
    assert sent.endswith(".")
    for label in ("forceful", "bright", "crisp"):
        assert label in sent
