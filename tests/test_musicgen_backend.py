import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from macchina_ricordi.musicgen.backend import (
    DEFAULT_EXTEND_STRIDE_SECONDS,
    GenerationRequest,
    build_audiocraft_backend,
    resolve_generation_checkpoint,
)
from macchina_ricordi.musicgen.runtime import MusicGenProjectConfig


def _write_wav(path: Path, audio_ch_t: np.ndarray, sample_rate: int) -> None:
    pcm = (audio_ch_t * 32767.0).astype("<i2", copy=False)
    interleaved = np.ascontiguousarray(pcm.T)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(audio_ch_t.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(interleaved.tobytes())


def test_resolve_generation_checkpoint_selects_latest_epoch(tmp_path: Path) -> None:
    xp_path = tmp_path / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_2.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_10.th").write_text("", encoding="utf-8")

    checkpoint = resolve_generation_checkpoint(xp_path=xp_path, checkpoint="latest")

    assert checkpoint == (xp_path / "checkpoint_10.th").resolve()


def test_audiocraft_backend_runs_generation_helper(monkeypatch, tmp_path: Path) -> None:
    helper_script = tmp_path / "scripts" / "run_musicgen_generate.py"
    helper_script.parent.mkdir()
    helper_script.write_text("", encoding="utf-8")
    musicgen_python = tmp_path / "python"
    musicgen_python.write_text("", encoding="utf-8")
    config = MusicGenProjectConfig(
        repo_root=tmp_path,
        project_dir=tmp_path / "musicgen",
        env_dir=tmp_path / ".venv-musicgen",
        dora_dir=tmp_path / ".dora",
        musicgen_python=musicgen_python,
        audiocraft_dir=tmp_path / ".audiocraft",
        python_request="3.11",
        audiocraft_repo_url="https://github.com/aaronabebe/audiocraft",
        audiocraft_commit_sha="1ca32f28d2d6c223b4e2335042ac558c308f2ea2",
        default_recipe="micro-unconditional",
        default_valid_fraction=0.1,
        default_split_seed=42,
        recipes={},
        codecs={},
    )
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    checkpoint = xp_path / "checkpoint_5.th"
    checkpoint.write_text("", encoding="utf-8")
    prompt_audio = tmp_path / "seed.wav"
    prompt_audio.write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, check, capture_output, text, env):  # type: ignore[no-untyped-def]
        commands.append(list(command))
        output_dir = Path(command[command.index("--output-dir") + 1])
        sample_rate = 32_000
        sample = np.zeros((1, 32), dtype=np.float32)
        audio_path = output_dir / "sample_000.wav"
        _write_wav(audio_path, sample, sample_rate)
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "backend": "audiocraft",
                    "mode": "continuation",
                    "model_id": "dummy/micro",
                    "device": "cpu",
                    "sample_rate": sample_rate,
                    "files": [str(audio_path)],
                    "warnings": [],
                    "source": {
                        "kind": "xp",
                        "profile": "micro-unconditional",
                        "xp_path": str(xp_path),
                        "checkpoint_path": str(checkpoint),
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("macchina_ricordi.musicgen.backend.load_musicgen_project_config", lambda: config)
    monkeypatch.setattr("macchina_ricordi.musicgen.runtime.load_musicgen_project_config", lambda: config)
    monkeypatch.setattr("macchina_ricordi.musicgen.backend.musicgen_script_env", lambda: {})
    monkeypatch.setattr("macchina_ricordi.musicgen.backend.subprocess.run", fake_run)

    backend = build_audiocraft_backend(
        model_id="dummy/micro",
        profile_name="micro-unconditional",
        xp_path=xp_path,
        checkpoint="latest",
    )
    result = backend.generate(
        GenerationRequest(
            prompts=[""],
            batch_size=2,
            mode="continuation",
            duration_seconds=8.0,
            prompt_audio_path=str(prompt_audio),
            prompt_duration_seconds=2.0,
            seed=7,
            temperature=1.2,
        )
    )

    assert commands[0][0] == str(musicgen_python)
    assert commands[0][1] == str(helper_script)
    assert commands[0][commands[0].index("--mode") + 1] == "continuation"
    assert commands[0][commands[0].index("--num-samples") + 1] == "2"
    assert commands[0][commands[0].index("--checkpoint-path") + 1] == str(checkpoint.resolve())
    assert commands[0][commands[0].index("--extend-stride") + 1] == str(DEFAULT_EXTEND_STRIDE_SECONDS)
    assert "--prompt-audio" in commands[0]
    assert result.backend == "audiocraft"
    assert result.metadata["mode"] == "continuation"
    assert result.metadata["source"]["kind"] == "xp"
