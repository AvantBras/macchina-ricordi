from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.instrument.sweep import arc_score_sweep_rows
from macchina_ricordi.scoring import arc
from macchina_ricordi.scoring.arc import (
    ArcCoherenceComponents,
    compute_arc_coherence,
    compute_arc_coherence_waveform,
)
from macchina_ricordi.scoring.similarity import clear_calibration

SR = 16_000


@pytest.fixture(autouse=True)
def _no_calibration() -> None:
    # Keep similarity cosine raw/uncalibrated so assertions are deterministic.
    clear_calibration()


def _sine(freq: float, *, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# Pitch classes C/E/G/A — all distinct, so a clip that steps through them "develops".
_C4, _E4, _G4, _A4 = 261.63, 329.63, 392.00, 440.00


def _drone(seconds: float = 8.0) -> np.ndarray:
    return _sine(_C4, seconds=seconds)


def _developing(quarter: float = 2.0) -> np.ndarray:
    return np.concatenate([_sine(f, seconds=quarter) for f in (_C4, _E4, _G4, _A4)]).astype(np.float32)


def _incoherent_halves(half: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = (0.5 * rng.standard_normal(int(half * SR))).astype(np.float32)
    return np.concatenate([_sine(_C4, seconds=half), noise]).astype(np.float32)


def test_static_drone_has_low_arc_and_high_coherence() -> None:
    comp = compute_arc_coherence_waveform(_drone(), sr=SR)
    assert comp.arc_tv4 < 0.05
    assert comp.chroma_drift < 0.05
    assert comp.timbral_coherence > 0.9


def test_developing_clip_has_higher_arc_than_drone() -> None:
    drone = compute_arc_coherence_waveform(_drone(), sr=SR)
    dev = compute_arc_coherence_waveform(_developing(), sr=SR)
    assert dev.arc_tv4 > drone.arc_tv4
    assert dev.chroma_drift > drone.chroma_drift
    # development with a single (sine) timbre stays reasonably coherent
    assert dev.timbral_coherence > 0.5


def test_incoherent_halves_score_low_coherence() -> None:
    incoh = compute_arc_coherence_waveform(_incoherent_halves(), sr=SR)
    dev = compute_arc_coherence_waveform(_developing(), sr=SR)
    assert incoh.timbral_coherence < dev.timbral_coherence


def test_arc_coherence_score_gates_development_by_coherence() -> None:
    dev = compute_arc_coherence_waveform(_developing(), sr=SR)
    # composite == 10 * arc_tv4 * timbral_coherence
    assert dev.arc_coherence_score == pytest.approx(10.0 * dev.arc_tv4 * dev.timbral_coherence, rel=1e-6)
    assert dev.arc_coherence_score > 0.0


def test_compute_arc_coherence_trims_prompt_echo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    full = _developing(quarter=3.0)  # 12 s total
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"stub")

    def fake_load(path, *args, **kwargs):  # noqa: ANN001
        return full, SR

    monkeypatch.setattr(arc.librosa, "load", fake_load)
    comp = compute_arc_coherence(audio_path, prompt_duration_seconds=5.0)
    # 12 s - 5 s echo => ~7 s tail
    assert comp.arc_tail_seconds == pytest.approx(7.0, abs=0.1)


def test_compute_arc_coherence_graceful_on_short_tail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"stub")
    monkeypatch.setattr(arc.librosa, "load", lambda *a, **k: (np.zeros(64, dtype=np.float32), SR))
    comp = compute_arc_coherence(audio_path)
    assert comp == ArcCoherenceComponents(arc_tail_seconds=comp.arc_tail_seconds)
    assert comp.arc_tv4 == 0.0


# --- sweep integration ---------------------------------------------------------------


@dataclass(frozen=True)
class _FakeArc:
    arc_coherence_score: float = 9.9
    arc_tv4: float = 0.11
    timbral_coherence: float = 0.9

    def to_dict(self) -> dict[str, object]:
        return {
            "arc_coherence_score": self.arc_coherence_score,
            "arc_tv4": self.arc_tv4,
            "timbral_coherence": self.timbral_coherence,
        }


def test_arc_score_sweep_rows_adds_columns_and_passes_the_echo_length(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    seen: dict[str, object] = {}

    def scorer(path, **kwargs):  # noqa: ANN001
        seen.update(kwargs)
        return _FakeArc()

    rows = arc_score_sweep_rows(
        [
            {
                "path": str(audio_path),
                "error": "",
                "prompt_audio_path": "/tmp/ood.mp3",
                "prompt_start_seconds": "30.0",
                "prompt_duration_seconds": "5.0",
            }
        ],
        scorer=scorer,
        arc_components_cls=_FakeArc,
    )

    assert rows[0]["arc_tv4"] == "0.11"
    assert rows[0]["arc_coherence_score"] == "9.9"
    assert rows[0]["arc_score_error"] == ""
    # Only the echo length is passed through now: the arc metrics are computed
    # on the generated tail, and nothing measures adherence to the prompt.
    assert "prompt_audio_path" not in seen
    assert seen["prompt_duration_seconds"] == "5.0"


def test_arc_score_sweep_rows_skips_failed_and_empty_rows() -> None:
    calls: list[object] = []

    def scorer(path, **kwargs):  # noqa: ANN001
        calls.append(path)
        return _FakeArc()

    rows = arc_score_sweep_rows(
        [
            {"path": "", "error": ""},
            {"path": "x.wav", "error": "backend failed"},
        ],
        scorer=scorer,
        arc_components_cls=_FakeArc,
    )
    assert calls == []
    assert rows[0]["arc_tv4"] == "0.11"  # seeded default still present
    assert rows[1]["arc_score_error"] == ""


def test_arc_score_sweep_rows_records_exceptions(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    def scorer(path, **kwargs):  # noqa: ANN001
        raise RuntimeError("arc exploded")

    rows = arc_score_sweep_rows(
        [{"path": str(audio_path), "error": ""}],
        scorer=scorer,
        arc_components_cls=_FakeArc,
    )
    assert rows[0]["arc_score_error"] == "arc exploded"
