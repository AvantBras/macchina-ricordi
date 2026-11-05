from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from macchina_ricordi.instrument.run_paths import OUTPUT_LAYOUT_FLAT
from macchina_ricordi.instrument.saving import save_generation_run
from macchina_ricordi.musicgen.backend import GenerationRequest, GenerationResult
from macchina_ricordi.musicgen.shared import write_text_atomic


def test_write_text_atomic_keeps_the_mode_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "models.toml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    write_text_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert target.stat().st_mode & 0o777 == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["models.toml"]


def _base_request() -> GenerationRequest:
    return GenerationRequest(
        prompts=["chaotic percussion, no steady beat"],
        batch_size=2,
        duration_seconds=5.0,
        max_new_tokens=128,
        seed=42,
    )


def _base_result(audio_values: np.ndarray, sample_rate: int = 32000) -> GenerationResult:
    return GenerationResult(
        audio_values=audio_values,
        sample_rate=sample_rate,
        backend="transformers",
        model_id="facebook/musicgen-small",
        device="cpu",
        warnings=[],
    )


def test_save_generation_run_writes_wavs_and_metadata(tmp_path: Path) -> None:
    # [B, T] mono batch
    audio = np.zeros((2, 1000), dtype=np.float32)
    saved = save_generation_run(
        _base_result(audio),
        _base_request(),
        output_dir=tmp_path,
        prefix="demo",
    )

    assert len(saved.audio_paths) == 2
    for wav_path in saved.audio_paths:
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 32000
            assert wav_file.getnframes() == 1000

    payload = json.loads(saved.metadata_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "transformers"
    assert payload["request"]["seed"] == 42
    assert len(payload["files"]) == 2
    assert payload["files"] == [path.name for path in saved.audio_paths]
    assert payload["output"]["layout"] == OUTPUT_LAYOUT_FLAT
