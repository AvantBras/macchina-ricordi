from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from macchina_ricordi.instrument.replay import (
    load_source_run,
    run_replay_audit,
)
from macchina_ricordi.instrument.saving import save_generation_run
from macchina_ricordi.musicgen.backend import GenerationRequest, GenerationResult


def _make_source_run(
    tmp_path: Path,
    *,
    backend: str = "micro",
    model_id: str = "test/model",
    device: str = "cpu",
    seed: int | None = 7,
    batch_size: int = 2,
    seconds: float = 0.1,
) -> tuple[Path, Path, np.ndarray]:
    sample_rate = 16000
    samples = int(sample_rate * seconds)
    audio = (np.linspace(-0.5, 0.5, samples, dtype=np.float32)[None, None, :]).repeat(batch_size, axis=0)
    result = GenerationResult(
        audio_values=audio,
        sample_rate=sample_rate,
        backend=backend,
        model_id=model_id,
        device=device,
        warnings=[],
    )
    request = GenerationRequest(
        prompts=[""] * batch_size,
        batch_size=batch_size,
        duration_seconds=seconds,
        max_new_tokens=64,
        seed=seed,
    )
    saved = save_generation_run(
        result,
        request,
        output_dir=tmp_path / "original",
        prefix="source",
    )
    return saved.metadata_path, saved.audio_paths[0].parent, audio


def test_run_replay_audit_reports_match_for_identical_bytes(tmp_path: Path) -> None:
    metadata_path, _, audio = _make_source_run(tmp_path, batch_size=2)
    source = load_source_run(metadata_path)

    class IdenticalBackend:
        def generate(self, request):  # type: ignore[no-untyped-def]
            return GenerationResult(
                audio_values=audio,
                sample_rate=source.sample_rate,
                backend=source.backend,
                model_id=source.model_id,
                device=source.device,
                warnings=[],
            )

    report = run_replay_audit(
        source,
        output_dir=tmp_path / "replays",
        backend_factory=lambda _: IdenticalBackend(),
    )

    assert report.all_match is True
    assert len(report.samples) == 2
    assert all(sample.matches for sample in report.samples)
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["all_match"] is True


def test_run_replay_audit_detects_byte_mismatch(tmp_path: Path) -> None:
    metadata_path, _, audio = _make_source_run(tmp_path, batch_size=2)
    source = load_source_run(metadata_path)

    altered = audio.copy()
    altered[0, 0, 0] = 0.99

    class AlteredBackend:
        def generate(self, request):  # type: ignore[no-untyped-def]
            return GenerationResult(
                audio_values=altered,
                sample_rate=source.sample_rate,
                backend=source.backend,
                model_id=source.model_id,
                device=source.device,
                warnings=[],
            )

    report = run_replay_audit(
        source,
        output_dir=tmp_path / "replays",
        backend_factory=lambda _: AlteredBackend(),
    )

    assert report.all_match is False
    matches = [sample.matches for sample in report.samples]
    assert matches[0] is False
    assert matches[1] is True
