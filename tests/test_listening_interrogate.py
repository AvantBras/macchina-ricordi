from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from macchina_ricordi.listening import cli as listening_cli
from macchina_ricordi.listening import interrogate as interrogate_module
from macchina_ricordi.listening.captioner import DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID
from macchina_ricordi.listening.features import analyze_waveform
from macchina_ricordi.listening.interrogate import (
    INTERROGATION_SCHEMA,
    append_interrogation_report,
    default_interrogation_report_path,
    interrogate_audio,
    render_interrogation_markdown,
)
from macchina_ricordi.listening.schema import WholeClip


class EchoBackend:
    name = "fake-interrogator"
    model_id = "fake/model"
    device = "cpu"
    dtype = "float32"
    quantization = "none"
    trust_remote_code = False

    def __init__(self, answers: dict[str, str] | None = None) -> None:
        self.answers = answers or {}
        self.prompts: list[str] = []

    def generate_caption(
        self,
        audio_path: str | Path,
        *,
        prompt: str,
        max_new_tokens: int,
        max_audio_seconds: float,
    ) -> str:
        self.prompts.append(prompt)
        return self.answers.get(prompt, f"Echo: {prompt}")


def _unstable_dossier(tmp_path: Path):
    dossier = analyze_waveform(np.zeros(16_000, dtype=np.float32), path=tmp_path / "clip.wav", sr=16_000)
    return replace(
        dossier,
        whole_clip=WholeClip(
            reward=dossier.whole_clip.reward,
            description={**dossier.whole_clip.description, "pulse": "unstable"},
            features={**dossier.whole_clip.features, "pulse_instability": 0.9},
        ),
    )


def test_interrogate_audio_asks_each_question_in_order(tmp_path: Path) -> None:
    backend = EchoBackend()
    report = interrogate_audio(
        tmp_path / "clip.wav",
        ["Is there a pulse?", "How dense is it?"],
        dossier=_unstable_dossier(tmp_path),
        captioner_backend=backend,
    )

    assert report.schema == INTERROGATION_SCHEMA
    assert backend.prompts == ["Is there a pulse?", "How dense is it?"]
    assert [exchange.question for exchange in report.exchanges] == ["Is there a pulse?", "How dense is it?"]
    assert report.exchanges[0].answer == "Echo: Is there a pulse?"
    assert report.backend.name == "fake-interrogator"
    assert all(exchange.asked_at for exchange in report.exchanges)


def test_interrogate_audio_cross_examines_answers(tmp_path: Path) -> None:
    backend = EchoBackend({"Q": "There is a steady beat and a gentle melody throughout."})
    report = interrogate_audio(
        tmp_path / "clip.wav",
        ["Q"],
        dossier=_unstable_dossier(tmp_path),
        captioner_backend=backend,
    )

    exchange = report.exchanges[0]
    assert exchange.claims["steady_beat"] is True
    assert exchange.claims["melody"] is True
    assert any("steady beat" in warning for warning in exchange.warnings)
    assert any("melody/harmony/chords" in warning for warning in exchange.warnings)


def test_interrogate_audio_requires_questions(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        interrogate_audio(tmp_path / "clip.wav", [], captioner_backend=EchoBackend())


def test_interrogation_report_round_trips_to_json(tmp_path: Path) -> None:
    report = interrogate_audio(
        tmp_path / "clip.wav",
        ["Anything?"],
        dossier=_unstable_dossier(tmp_path),
        captioner_backend=EchoBackend(),
    )

    round_trip = json.loads(json.dumps(report.to_dict()))

    assert round_trip["schema"] == INTERROGATION_SCHEMA
    assert round_trip["exchanges"][0]["question"] == "Anything?"


def test_append_interrogation_report_accumulates_exchanges(tmp_path: Path) -> None:
    dossier = _unstable_dossier(tmp_path)
    audio = tmp_path / "clip.wav"
    first = interrogate_audio(audio, ["First?"], dossier=dossier, captioner_backend=EchoBackend())
    second = interrogate_audio(audio, ["Second?"], dossier=dossier, captioner_backend=EchoBackend())

    path = append_interrogation_report(first)
    assert path == default_interrogation_report_path(audio).resolve()
    append_interrogation_report(second)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [exchange["question"] for exchange in payload["exchanges"]] == ["First?", "Second?"]


def test_append_interrogation_report_overwrites_corrupt_sidecar(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    report_path = default_interrogation_report_path(audio)
    report_path.write_text("not json", encoding="utf-8")

    report = interrogate_audio(audio, ["Q?"], dossier=_unstable_dossier(tmp_path), captioner_backend=EchoBackend())
    append_interrogation_report(report)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert [exchange["question"] for exchange in payload["exchanges"]] == ["Q?"]


def test_render_interrogation_markdown_lists_questions_and_warnings(tmp_path: Path) -> None:
    backend = EchoBackend({"Q": "A steady beat under sparse chords."})
    report = interrogate_audio(
        tmp_path / "clip.wav", ["Q"], dossier=_unstable_dossier(tmp_path), captioner_backend=backend
    )

    markdown = render_interrogation_markdown(report)

    assert "# Interrogation" in markdown
    assert "## Q: Q" in markdown
    assert "A steady beat under sparse chords." in markdown
    assert "Warnings:" in markdown
    assert "model testimony" in markdown


def _ask_args(audio_path: Path, questions: list[str], **overrides: object) -> argparse.Namespace:
    data = {
        "audio_path": str(audio_path),
        "questions": questions,
        "output": "",
        "sidecar": True,
        "json": False,
        "sr": 16_000,
        "hop_length": 512,
        "res_type": "soxr_hq",
        "tempo_method": "tempogram",
        "window_seconds": 2.0,
        "window_hop_seconds": 1.0,
        "captioner_backend": "qwen2-audio",
        "captioner_model_id": "",
        "captioner_device": "auto",
        "captioner_dtype": "auto",
        "captioner_quantization": "auto",
        "captioner_max_new_tokens": 64,
        "captioner_max_audio_seconds": 30.0,
        "captioner_trust_remote_code": True,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_cmd_listen_ask_writes_sidecar_and_prints_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16_000, dtype=np.float32), 16_000)
    monkeypatch.setattr(interrogate_module, "create_captioner_backend", lambda **kwargs: EchoBackend())

    exit_code = listening_cli.cmd_listen_ask(_ask_args(audio, ["What is this?"]))

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Echo: What is this?" in captured.out
    payload = json.loads(default_interrogation_report_path(audio).read_text(encoding="utf-8"))
    assert payload["exchanges"][0]["question"] == "What is this?"


def test_cmd_listen_ask_no_sidecar_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16_000, dtype=np.float32), 16_000)
    monkeypatch.setattr(interrogate_module, "create_captioner_backend", lambda **kwargs: EchoBackend())

    exit_code = listening_cli.cmd_listen_ask(_ask_args(audio, ["Q?"], sidecar=False, json=True))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exchanges"][0]["answer"] == "Echo: Q?"
    assert not default_interrogation_report_path(audio).exists()


def test_listen_ask_parser_accepts_af3_backend(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    listening_cli.register_listening_group(subparsers)

    args = parser.parse_args(
        ["listen", "ask", str(tmp_path / "a.wav"), "Q1", "Q2", "--captioner-backend", "audio-flamingo-3"]
    )

    assert args.questions == ["Q1", "Q2"]
    assert args.captioner_backend == "audio-flamingo-3"
    assert args.captioner_model_id == ""
    assert args.func is listening_cli.cmd_listen_ask
    assert DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID == "nvidia/music-flamingo-hf"
