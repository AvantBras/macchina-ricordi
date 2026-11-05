from __future__ import annotations

from pathlib import Path

import numpy as np

from macchina_ricordi.cli import main
from macchina_ricordi.musicgen.backend import GenerationResult


def test_musicgen_generate_unconditional_subcommand_uses_audiocraft_backend(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    class DummyAudioCraftBackend:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            return GenerationResult(
                audio_values=np.zeros((2, 1, 32), dtype=np.float32),
                sample_rate=32_000,
                backend="audiocraft",
                model_id="dummy/micro",
                device="cpu",
                warnings=[],
                metadata={
                    "mode": "unconditional",
                    "source": {"kind": "xp", "checkpoint_path": "/tmp/checkpoint_5.th"},
                },
            )

    backend = DummyAudioCraftBackend()
    seen_kwargs = {}

    def fake_build_backend(**kwargs):  # type: ignore[no-untyped-def]
        seen_kwargs.update(kwargs)
        return backend

    monkeypatch.setattr("macchina_ricordi.musicgen.backend.build_audiocraft_backend", fake_build_backend)

    exit_code = main(
        [
            "play",
            "generate",
            "--profile",
            "micro-unconditional",
            "--xp-path",
            ".dora/xps/abcd1234",
            "--checkpoint",
            "latest",
            "--duration-seconds",
            "12",
            "--num-samples",
            "2",
            "--temperature",
            "1.2",
            "--no-save-output",
        ]
    )

    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert seen_kwargs["profile_name"] == "micro-unconditional"
    assert seen_kwargs["xp_path"] == ".dora/xps/abcd1234"
    assert seen_kwargs["checkpoint"] == "latest"
    assert backend.requests[0].mode == "unconditional"
    assert backend.requests[0].prompts == [""]
    assert backend.requests[0].batch_size == 2
    assert backend.requests[0].duration_seconds == 12.0
    assert backend.requests[0].temperature == 1.2
    assert backend.requests[0].extend_stride == 8.0
    assert "backend: audiocraft" in stdout
    assert "mode: unconditional" in stdout
    assert "checkpoint_path: /tmp/checkpoint_5.th" in stdout


def test_musicgen_generate_continue_subcommand_preserves_prompt_window(monkeypatch, tmp_path: Path) -> None:
    class DummyAudioCraftBackend:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            return GenerationResult(
                audio_values=np.zeros((1, 1, 32), dtype=np.float32),
                sample_rate=32_000,
                backend="audiocraft",
                model_id="dummy/micro",
                device="cpu",
                warnings=[],
                metadata={"mode": "continuation", "source": {"kind": "model_id"}},
            )

    backend = DummyAudioCraftBackend()
    monkeypatch.setattr("macchina_ricordi.musicgen.backend.build_audiocraft_backend", lambda **kwargs: backend)

    exit_code = main(
        [
            "play",
            "continue",
            "--prompt-audio",
            str(tmp_path / "seed.wav"),
            "--prompt-start-seconds",
            "1.5",
            "--prompt-duration-seconds",
            "2.5",
            "--duration-seconds",
            "10",
            "--no-save-output",
        ]
    )

    assert exit_code == 0
    request = backend.requests[0]
    assert request.mode == "continuation"
    assert request.prompt_audio_path == str(tmp_path / "seed.wav")
    assert request.prompt_start_seconds == 1.5
    assert request.prompt_duration_seconds == 2.5
