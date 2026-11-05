from __future__ import annotations

import pytest

from macchina_ricordi import cli


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "segment",
                "file",
                "audio.mp3",
                "segments",
                "--min-silence-seconds",
                "2.5",
                "--silence-threshold-db",
                "-25.0",
                "--min-track-seconds",
                "15.0",
                "--output-sr",
                "48000",
                "--no-mono",
            ],
            {
                "segment_command": "file",
                "audio_path": "audio.mp3",
                "output_dir": "segments",
                "min_silence_seconds": 2.5,
                "silence_threshold_db": -25.0,
                "min_track_seconds": 15.0,
                "output_sr": 48000,
                "mono": False,
            },
        ),
        (
            ["segment", "dir", "recordings", "segments", "--recursive", "--extensions", "wav,flac"],
            {
                "segment_command": "dir",
                "input_dir": "recordings",
                "output_dir": "segments",
                "recursive": True,
                "extensions": "wav,flac",
            },
        ),
        (  # the documented defaults
            ["segment", "file", "audio.mp3", "segments"],
            {
                "min_silence_seconds": 1.5,
                "silence_threshold_db": -32.0,
                "min_track_seconds": 30.0,
                "output_sr": 32_000,
                "mono": True,
            },
        ),
        (["runtime", "setup"], {"command": "runtime", "runtime_command": "setup"}),
        (
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
                "30",
                "--num-samples",
                "8",
            ],
            {
                "command": "play",
                "play_command": "generate",
                "generation_mode": "unconditional",
                "profile": "micro-unconditional",
                "xp_path": ".dora/xps/abcd1234",
                "num_samples": 8,
                "duration_seconds": 30.0,
                "output_dir": "artifacts/sessions",
                "output_layout": "session",
                "extend_stride": 8.0,
            },
        ),
        (
            ["play", "continue", "--prompt-audio", "seed.wav", "--prompt-duration-seconds", "2.5"],
            {"generation_mode": "continue", "prompt_audio": "seed.wav", "prompt_duration_seconds": 2.5},
        ),
        (
            ["data", "prepare", "clips", "--workspace-dir", ".ws"],
            {"command": "data", "data_command": "prepare", "profile": None, "workspace_dir": ".ws"},
        ),
        (["runtime", "check", "--audiocraft-dir", "C:/audiocraft"], {"runtime_command": "check"}),
        (
            ["train", "run", "--workspace-dir", ".ws", "--audiocraft-dir", "C:/audiocraft"],
            {"train_command": "run", "profile": "reshape-nano"},  # the default recipe
        ),
        (
            ["train", "run", "--workspace-dir", ".ws", "--run-label", "midwife", "--no-auto-dataset-label"],
            {"train_command": "run", "run_label": "midwife", "no_auto_dataset_label": True},
        ),
        (
            [
                "model",
                "export",
                "midwife",
                "--bundle",
                "--out-dir",
                "models",
                "--weights-state",
                "best",
                "--force",
                "--dry-run",
            ],
            {
                "model_command": "export",
                "bundle": True,
                "box": False,
                "weights_state": "best",
                "out_dir": "models",
                "force": True,
                "dry_run": True,
            },
        ),
        (
            ["model", "export", "midwife@best", "--box"],
            {"model_command": "export", "bundle": False, "box": True, "weights_state": "model"},
        ),
        (
            ["model", "export", "midwife", "--bundle", "--weights-state", "ema"],
            {"model_command": "export", "bundle": True, "weights_state": "ema"},
        ),
        (["model", "perturb", "midwife@best", "out.th", "--sigma", "0.01"], {"model_command": "perturb"}),
        (["train", "resume", "abcd1234", "--epochs", "5"], {"train_command": "resume"}),
        (
            [
                "listen",
                "file",
                "sample.wav",
                "--output",
                "sample_listening_report.json",
                "--markdown",
                "sample_listening.md",
                "--sidecars",
                "--json",
                "--sr",
                "16000",
                "--hop-length",
                "256",
                "--res-type",
                "soxr_qq",
                "--tempo-method",
                "autocorr",
                "--window-seconds",
                "1.5",
                "--window-hop-seconds",
                "0.5",
                "--semantic-probes",
                "--semantic-backend",
                "msclap",
                "--semantic-model-version",
                "2023",
                "--semantic-device",
                "cpu",
                "--semantic-model-path",
                "clap.pt",
                "--captioner-witness",
                "--captioner-backend",
                "qwen2-audio",
                "--captioner-model-id",
                "Qwen/Qwen2-Audio-7B-Instruct",
                "--captioner-device",
                "cpu",
                "--captioner-dtype",
                "float32",
                "--captioner-quantization",
                "bnb-4bit",
                "--captioner-max-new-tokens",
                "64",
                "--captioner-max-audio-seconds",
                "30",
                "--captioner-prompt-profile",
                "experimental_texture_v1",
                "--no-captioner-trust-remote-code",
            ],
            {
                "command": "listen",
                "listen_command": "file",
                "audio_path": "sample.wav",
                "output": "sample_listening_report.json",
                "markdown": "sample_listening.md",
                "sidecars": True,
                "json": True,
                "sr": 16_000,
                "hop_length": 256,
                "res_type": "soxr_qq",
                "tempo_method": "autocorr",
                "window_seconds": 1.5,
                "window_hop_seconds": 0.5,
                "semantic_probes": True,
                "semantic_backend": "msclap",
                "semantic_model_version": "2023",
                "semantic_device": "cpu",
                "semantic_model_path": "clap.pt",
                "captioner_witness": True,
                "captioner_backend": "qwen2-audio",
                "captioner_model_id": "Qwen/Qwen2-Audio-7B-Instruct",
                "captioner_device": "cpu",
                "captioner_dtype": "float32",
                "captioner_quantization": "bnb-4bit",
                "captioner_max_new_tokens": 64,
                "captioner_max_audio_seconds": 30.0,
                "captioner_prompt_profile": "experimental_texture_v1",
                "captioner_trust_remote_code": False,
            },
        ),
        (
            ["listen", "file", "sample.wav"],
            {"captioner_quantization": "auto"},  # the default, when nothing is asked for
        ),
    ],
)
def test_each_command_parses_its_flags_into_the_names_the_code_reads(argv: list[str], expected: dict) -> None:
    parsed = vars(cli.build_parser().parse_args(argv))
    assert {name: parsed[name] for name in expected} == expected


def test_stream_completion_duration_cli_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MACCHINA_RICORDI_COMPLETION_DURATION_SECONDS", "7.5")
    parser = cli.build_parser()

    args = parser.parse_args(["play", "stream"])
    assert args.completion_duration == pytest.approx(7.5)

    explicit = parser.parse_args(["play", "stream", "--completion-duration", "3.25"])
    assert explicit.completion_duration == pytest.approx(3.25)

    with pytest.raises(SystemExit):
        parser.parse_args(["play", "stream", "--round-duration", "15"])

    monkeypatch.delenv("MACCHINA_RICORDI_COMPLETION_DURATION_SECONDS")
    default_args = cli.build_parser().parse_args(["play", "stream"])
    assert default_args.completion_duration == pytest.approx(10.0)
