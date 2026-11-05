from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.scoring.similarity import (
    SimilarityComponents,
    _cosine_similarity,
    _mfcc_stats_vector,
    _rhythm_features_vector,
    _spectral_stats_vector,
    calibrate_from_dir,
    calibration_loaded,
    clear_calibration,
    compute_self_similarity,
    compute_similarity,
    compute_similarity_waveform,
    compute_similarity_with_baseline,
    load_calibration,
)


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


def _make_noise(*, duration: float = 3.0, sr: int = 16_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, int(duration * sr)).astype(np.float32)


def _make_chirp(*, duration: float = 3.0, sr: int = 16_000) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False, dtype=np.float32)
    f0, f1 = 200.0, 4000.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    return np.sin(phase).astype(np.float32) * 0.5


# ── helpers ──────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity(np.zeros(5), np.ones(5)) == 0.0
        assert _cosine_similarity(np.zeros(3), np.zeros(3)) == 0.0


class TestMfccStatsVector:
    def test_output_is_a_fixed_117_vector_whatever_the_clip_length(self) -> None:
        # Calibration dimensions depend on this: the width may not follow the clip.
        for duration in (0.5, 2.0, 10.0):
            vec = _mfcc_stats_vector(_make_noise(duration=duration), sr=16_000, n_mfcc=13)
            assert vec.shape == (117,)
            assert np.issubdtype(vec.dtype, np.floating)


class TestSpectralStatsVector:
    def test_output_is_a_fixed_39_vector_whatever_the_clip_length(self) -> None:
        for duration in (0.5, 2.0, 10.0):
            vec = _spectral_stats_vector(_make_noise(duration=duration), sr=16_000, n_mels=32)
            assert vec.shape == (39,)  # 32 mel means + 7 contrast means
            assert np.issubdtype(vec.dtype, np.floating)

    def test_chirp_vs_noise_differ_on_contrast(self) -> None:
        chirp = _make_chirp(duration=2.0)
        noise = _make_noise(duration=2.0, seed=99)
        v_chirp = _spectral_stats_vector(chirp, sr=16_000)
        v_noise = _spectral_stats_vector(noise, sr=16_000)
        # Contrast values (last 7 dims) should differ
        contrast_chirp = v_chirp[32:]
        contrast_noise = v_noise[32:]
        assert not np.allclose(contrast_chirp, contrast_noise, atol=2.0)


class TestRhythmFeaturesVector:
    def test_output_shape(self) -> None:
        wav = _make_noise(duration=2.0)
        vec = _rhythm_features_vector(wav, sr=16_000)
        assert vec.shape == (7,)
        assert np.issubdtype(vec.dtype, np.floating)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

    def test_empty_waveform_returns_zero_vector(self) -> None:
        vec = _rhythm_features_vector(np.array([], dtype=np.float32), sr=16_000)
        assert np.all(vec == 0.0)


# ── calibration ───────────────────────────────────────────────────────


class TestCalibration:
    def teardown_method(self) -> None:
        clear_calibration()

    def test_load_and_clear(self) -> None:
        assert not calibration_loaded()  # uncalibrated by default
        data = {
            "calibration": {
                "mfcc": {"means": [1.0] * 117, "stds": [0.5] * 117},
                "spectral": {"means": [2.0] * 39, "stds": [0.3] * 39},
                "rhythm": {"means": [0.5] * 8, "stds": [0.2] * 8},
            }
        }
        load_calibration(data)
        assert calibration_loaded()
        clear_calibration()
        assert not calibration_loaded()

    def test_load_calibration_skips_stale_blocks(self) -> None:
        """A calibration written against a different vector length must not install.

        `dominant_tempo_consistency` left the rhythm vector on 2026-08-06, so
        every JSON from before that carries an 8-wide rhythm block. Installing
        it would broadcast-error inside a cosine; skipping it degrades to raw.
        """
        from macchina_ricordi.scoring import similarity

        dims = similarity.expected_dims()
        data = {
            "calibration": {
                "mfcc": {"means": [0.0] * dims["mfcc"], "stds": [1.0] * dims["mfcc"]},
                "rhythm": {"means": [0.0] * (dims["rhythm"] + 1), "stds": [1.0] * (dims["rhythm"] + 1)},
            }
        }
        load_calibration(data)
        assert set(similarity._CALIBRATION) == {"mfcc"}
        clear_calibration()

    def test_calibrate_from_dir_on_tmp(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        for i in range(3):
            wav = _make_noise(duration=1.0, seed=i)
            sf.write(str(tmp_path / f"clip_{i}.wav"), wav, 16_000)

        result = calibrate_from_dir(tmp_path, sr=16_000)
        assert result["num_files"] == 3
        assert "mfcc" in result["calibration"]
        assert "spectral" in result["calibration"]
        assert "rhythm" in result["calibration"]
        assert len(result["calibration"]["mfcc"]["means"]) == 117
        assert len(result["calibration"]["spectral"]["means"]) == 39
        assert len(result["calibration"]["rhythm"]["means"]) == 7

        load_calibration(result)
        assert calibration_loaded()

        # Calibrated comparison should still be in valid range
        a = tmp_path / "clip_0.wav"
        b = tmp_path / "clip_1.wav"
        c = compute_similarity(a, b)
        assert -1.0 <= c.combined <= 1.0

        clear_calibration()


# ── waveform API ──────────────────────────────────────────────────────


class TestComputeSimilarityWaveform:
    def test_identical_waveform_cosine_near_one(self) -> None:
        clicks = _make_click_track(np.arange(0.5, 4.0, 0.3), duration=5.0)
        result = compute_similarity_waveform(clicks, clicks)
        assert result.mfcc_cosine == pytest.approx(1.0, abs=1e-6)
        assert result.spectral_cosine == pytest.approx(1.0, abs=1e-6)
        assert result.rhythm_cosine == pytest.approx(1.0, abs=1e-6)
        assert result.combined == pytest.approx(1.0, abs=1e-6)
        assert result.self_similarity_a is None

    def test_click_vs_noise_low_similarity(self) -> None:
        clicks = _make_click_track(np.arange(0.5, 4.0, 0.3), duration=5.0)
        noise = _make_noise(duration=5.0, seed=99)
        result = compute_similarity_waveform(clicks, noise)
        assert result.combined < 0.5

    def test_symmetry(self) -> None:
        a = _make_noise(duration=2.0, seed=3)
        b = _make_noise(duration=2.0, seed=7)
        ab = compute_similarity_waveform(a, b)
        ba = compute_similarity_waveform(b, a)
        assert ab.mfcc_cosine == pytest.approx(ba.mfcc_cosine)
        assert ab.combined == pytest.approx(ba.combined)

    def test_rejects_non_mono(self) -> None:
        stereo = np.zeros((2, 1000), dtype=np.float32)
        mono = np.zeros(1000, dtype=np.float32)
        with pytest.raises(ValueError):
            compute_similarity_waveform(stereo, mono)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (_make_noise(duration=1.0), _make_noise(duration=15.0, seed=7)),  # different lengths
            (np.array([], dtype=np.float32), _make_noise(duration=1.0)),  # empty
            (np.zeros(8000, dtype=np.float32), np.zeros(16000, dtype=np.float32)),  # silence vs silence
        ],
    )
    def test_edge_inputs_give_a_bounded_score(self, a: np.ndarray, b: np.ndarray) -> None:
        result = compute_similarity_waveform(a, b)
        assert -1.0 <= result.combined <= 1.0

    def test_return_type(self) -> None:
        a = _make_noise(duration=1.0)
        result = compute_similarity_waveform(a, a)
        assert isinstance(result, SimilarityComponents)
        assert isinstance(result.mfcc_cosine, float)


# ── file API ──────────────────────────────────────────────────────────


class TestComputeSimilarity:
    def test_identical_file(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        wav = _make_noise(duration=2.0)
        path = tmp_path / "test.wav"
        sf.write(str(path), wav, 16_000)
        result = compute_similarity(path, path)
        assert result.mfcc_cosine == pytest.approx(1.0, abs=1e-4)
        assert result.combined == pytest.approx(1.0, abs=1e-4)

    def test_two_different_files(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        noise = _make_noise(duration=2.0, seed=1)
        clicks = _make_click_track(np.arange(0.5, 1.5, 0.2), duration=2.0)
        path_a = tmp_path / "noise.wav"
        path_b = tmp_path / "clicks.wav"
        sf.write(str(path_a), noise, 16_000)
        sf.write(str(path_b), clicks, 16_000)
        result = compute_similarity(path_a, path_b)
        assert result.combined < 0.5


# ── self-similarity ───────────────────────────────────────────────────


class TestComputeSelfSimilarity:
    def test_noise_self_similarity_near_one(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        noise = _make_noise(duration=4.0, seed=42)
        path = tmp_path / "noise.wav"
        sf.write(str(path), noise, 16_000)
        result = compute_self_similarity(path, sr=16_000)
        assert result.combined > 0.9
        assert result.self_similarity_a is None

    def test_chirp_self_similarity_lower(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        chirp = _make_chirp(duration=4.0)
        path = tmp_path / "chirp.wav"
        sf.write(str(path), chirp, 16_000)
        result = compute_self_similarity(path, sr=16_000)
        assert result.combined < 0.95

    def test_short_file_fallback(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        short = _make_noise(duration=0.3, seed=1)
        path = tmp_path / "short.wav"
        sf.write(str(path), short, 16_000)
        result = compute_self_similarity(path, sr=16_000)
        assert result.combined == 1.0


# ── similarity with baseline ──────────────────────────────────────────


class TestComputeSimilarityWithBaseline:
    def test_fills_self_similarity_fields(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        noise_a = _make_noise(duration=4.0, seed=1)
        noise_b = _make_noise(duration=4.0, seed=2)
        path_a = tmp_path / "a.wav"
        path_b = tmp_path / "b.wav"
        sf.write(str(path_a), noise_a, 16_000)
        sf.write(str(path_b), noise_b, 16_000)
        result = compute_similarity_with_baseline(path_a, path_b, sr=16_000)
        assert isinstance(result.self_similarity_a, float)
        assert isinstance(result.self_similarity_b, float)
        assert 0.0 <= result.self_similarity_a <= 1.0

    def test_identical_files_baseline(self, tmp_path: Path) -> None:
        import soundfile as sf  # type: ignore[import-untyped]

        noise = _make_noise(duration=4.0, seed=1)
        path = tmp_path / "test.wav"
        sf.write(str(path), noise, 16_000)
        result = compute_similarity_with_baseline(path, path, sr=16_000)
        assert result.combined == pytest.approx(1.0, abs=1e-4)
        assert result.self_similarity_a == pytest.approx(result.self_similarity_b)


# ── dataclass ─────────────────────────────────────────────────────────


def test_similarity_components_to_dict() -> None:
    c = SimilarityComponents(mfcc_cosine=0.5, spectral_cosine=0.3, rhythm_cosine=0.7, combined=0.5)
    d = c.to_dict()
    assert d["mfcc_cosine"] == 0.5
    assert d["combined"] == 0.5
    assert set(d.keys()) == set(SimilarityComponents.fieldnames())


def test_archive_neighbours_and_score_sim_use_one_engine() -> None:
    from macchina_ricordi.listening.archive import ArchiveEntry, _entry_similarity, _vectors_for_waveform
    from macchina_ricordi.scoring.similarity import compute_similarity_waveform

    rng = np.random.default_rng(0)
    a = rng.standard_normal(16_000).astype(np.float32) * 0.1
    b = np.sin(np.linspace(0, 400 * np.pi, 16_000)).astype(np.float32) * 0.3
    direct = compute_similarity_waveform(a, b, sr=16_000)
    fields = {name: None for name in ArchiveEntry.__dataclass_fields__}
    entry_a = ArchiveEntry(**{**fields, "path": "a.wav", "vectors": _vectors_for_waveform(a, sr=16_000)})
    entry_b = ArchiveEntry(**{**fields, "path": "b.wav", "vectors": _vectors_for_waveform(b, sr=16_000)})
    neighbour = _entry_similarity(entry_a, entry_b)
    assert neighbour.similarity == pytest.approx(direct.combined, abs=1e-6)
    assert neighbour.rhythm_cosine == pytest.approx(direct.rhythm_cosine, abs=1e-6)
