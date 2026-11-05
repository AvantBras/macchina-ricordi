from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.instrument.prompt_transform import (
    FLAT_TILT,
    MAX_PROMPT_RATE,
    MAX_TILT_DB,
    MIN_PROMPT_RATE,
    PromptTilt,
    apply_rate,
    apply_tilt,
    band_gain_curve,
    clamp_prompt_rate,
    clamp_tilt_db,
    render_prompt_transform,
)

SR = 32_000


def _tone(freq: float, seconds: float, *, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _three_band_signal(seconds: float = 1.0, *, sr: int = SR) -> np.ndarray:
    return ((_tone(80, seconds, sr=sr) + _tone(1000, seconds, sr=sr) + _tone(9000, seconds, sr=sr)) / 3).astype(
        np.float32
    )


def _band_shares(waveform: np.ndarray, *, sr: int = SR) -> list[float]:
    spectrum = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(waveform.size, 1 / sr)
    total = float(np.sum(spectrum**2))
    return [
        float(np.sum(spectrum[(freqs >= lo) & (freqs < hi)] ** 2) / total)
        for lo, hi in ((0, 250), (250, 4000), (4000, sr / 2))
    ]


def _dominant_hz(waveform: np.ndarray, *, sr: int = SR) -> float:
    spectrum = np.abs(np.fft.rfft(waveform))
    return float(np.fft.rfftfreq(waveform.size, 1 / sr)[int(np.argmax(spectrum))])


def _write_wav(path: Path, waveform: np.ndarray, *, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes((np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return path


# ── the gain curve ────────────────────────────────────────────────────


def test_a_uniform_tilt_is_a_flat_gain_at_every_frequency() -> None:
    # Otherwise "boost everything by 6 dB" would quietly also reshape the
    # spectrum, and the crossover positions would leak into every reading.
    freqs = np.array([0.0, 50.0, 250.0, 1000.0, 4000.0, 16000.0])
    curve = band_gain_curve(freqs, PromptTilt(6.0, 6.0, 6.0))
    assert np.allclose(curve, 10.0 ** (6.0 / 20.0))


def test_the_gain_curve_reaches_each_bands_full_value_in_band() -> None:
    freqs = np.array([40.0, 1000.0, 12000.0])
    db = 20.0 * np.log10(band_gain_curve(freqs, PromptTilt(low_db=12.0, mid_db=0.0, high_db=-12.0)))
    assert db[0] == pytest.approx(12.0, abs=0.5)
    assert db[1] == pytest.approx(0.0, abs=0.5)
    assert db[2] == pytest.approx(-12.0, abs=0.5)


def test_the_gain_curve_crosses_over_smoothly_rather_than_as_a_brick_wall() -> None:
    # A hard band edge rings, and the ringing would be indistinguishable from
    # the model responding to the tilt.
    freqs = np.geomspace(20.0, 16000.0, 400)
    db = 20.0 * np.log10(band_gain_curve(freqs, PromptTilt(low_db=12.0, mid_db=0.0, high_db=0.0)))
    assert np.max(np.abs(np.diff(db))) < 1.0


# ── tilt ──────────────────────────────────────────────────────────────


def test_apply_tilt_moves_energy_into_the_boosted_band() -> None:
    signal = _three_band_signal()
    before = _band_shares(signal)
    after = _band_shares(apply_tilt(signal, sr=SR, tilt=PromptTilt(low_db=12.0)))
    assert after[0] > before[0] * 2
    assert after[2] < before[2]


def test_apply_tilt_preserves_length_and_is_a_noop_when_flat() -> None:
    signal = _three_band_signal(0.5)
    assert apply_tilt(signal, sr=SR, tilt=PromptTilt(high_db=-6.0)).size == signal.size
    assert apply_tilt(signal, sr=SR, tilt=FLAT_TILT) is signal


# ── rate ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rate", [0.9, 1.1, 1.5])
@pytest.mark.parametrize(("mode", "pitch_follows_rate"), [("tape", True), ("tempo", False)])
def test_tape_shifts_pitch_with_tempo_and_tempo_keeps_it(mode: str, pitch_follows_rate: bool, rate: float) -> None:
    source = _tone(440.0, rate)  # `rate` seconds, to become exactly 1.0
    shifted = apply_rate(source, sr=SR, target_samples=SR, mode=mode)
    assert shifted.size == SR
    expected_hz = 440.0 * rate if pitch_follows_rate else 440.0
    assert _dominant_hz(shifted) == pytest.approx(expected_hz, rel=0.02)


def test_apply_rate_hits_the_target_length_exactly() -> None:
    # The whole point of resampling *down to tau*: token count and the buffer
    # arithmetic must not move when the rate does.
    for rate in (0.5, 0.83, 1.0, 1.37, 2.0):
        source = _tone(300.0, rate)
        assert apply_rate(source, sr=SR, target_samples=SR, mode="tape").size == SR


def test_apply_rate_padding_keeps_the_end_of_the_audio() -> None:
    # The model continues from the end of the prompt, so a shortfall is paid at
    # the front where it is inaudible.
    source = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fitted = apply_rate(source, sr=SR, target_samples=5, mode="tape")
    assert fitted.size == 5
    assert float(np.max(np.abs(fitted[-2:]))) > 0.1  # the audio is still at the end


# ── render_prompt_transform ───────────────────────────────────────────


def test_render_always_produces_the_requested_duration(tmp_path: Path) -> None:
    source = _write_wav(tmp_path / "round.wav", _three_band_signal(15.0))
    for rate in (0.8, 1.0, 1.25):
        result = render_prompt_transform(
            source,
            tmp_path / f"out_{rate}.wav",
            start_seconds=10.0,
            duration_seconds=5.0,
            rate=rate,
        )
        assert result.duration_seconds == pytest.approx(5.0, abs=0.01)
        assert result.rate_applied == pytest.approx(rate, abs=0.01)
        assert not result.rate_was_clamped


def test_render_reports_a_rate_it_could_not_supply(tmp_path: Path) -> None:
    # rate 2.0 wants 10s of tail but the window starts at 0, so only 5s exists.
    source = _write_wav(tmp_path / "round.wav", _three_band_signal(15.0))
    result = render_prompt_transform(source, tmp_path / "out.wav", start_seconds=0.0, duration_seconds=5.0, rate=2.0)
    assert result.rate_was_clamped
    assert result.rate_applied == pytest.approx(1.0, abs=0.01)
    assert result.duration_seconds == pytest.approx(5.0, abs=0.01)  # still exactly tau


def test_render_applies_the_tilt_to_the_written_file(tmp_path: Path) -> None:
    import librosa

    source = _write_wav(tmp_path / "round.wav", _three_band_signal(15.0))
    out = tmp_path / "out.wav"
    render_prompt_transform(source, out, start_seconds=10.0, duration_seconds=5.0, tilt=PromptTilt(low_db=12.0))
    written, sr = librosa.load(out, sr=None, mono=True)
    assert _band_shares(written, sr=int(sr))[0] > 0.7


def test_render_keeps_a_boosted_prompt_from_clipping(tmp_path: Path) -> None:
    import librosa

    loud = (_three_band_signal(6.0) * 3.0).astype(np.float32)
    source = _write_wav(tmp_path / "round.wav", loud)
    out = tmp_path / "out.wav"
    render_prompt_transform(source, out, start_seconds=1.0, duration_seconds=5.0, tilt=PromptTilt(12.0, 12.0, 12.0))
    written, _ = librosa.load(out, sr=None, mono=True)
    assert float(np.max(np.abs(written))) <= 1.0


def test_render_rejects_an_empty_window(tmp_path: Path) -> None:
    source = _write_wav(tmp_path / "round.wav", _three_band_signal(2.0))
    with pytest.raises(ValueError):
        render_prompt_transform(source, tmp_path / "out.wav", start_seconds=0.0, duration_seconds=0.0)


# ── clamps ────────────────────────────────────────────────────────────


def test_clamp_prompt_rate_bounds_and_tolerates_junk() -> None:
    assert clamp_prompt_rate(1.05) == pytest.approx(1.05)
    assert clamp_prompt_rate(9.0) == MAX_PROMPT_RATE
    assert clamp_prompt_rate(0.0) == MIN_PROMPT_RATE
    assert clamp_prompt_rate("nonsense") == 1.0
    assert clamp_prompt_rate(float("nan")) == 1.0


def test_clamp_tilt_db_bounds_and_tolerates_junk() -> None:
    assert clamp_tilt_db(-3.5) == pytest.approx(-3.5)
    assert clamp_tilt_db(99.0) == MAX_TILT_DB
    assert clamp_tilt_db(-99.0) == -MAX_TILT_DB
    assert clamp_tilt_db(None) == 0.0


def test_is_flat_only_for_all_three_bands_at_zero() -> None:
    assert PromptTilt().is_flat
    assert not PromptTilt(mid_db=0.5).is_flat
