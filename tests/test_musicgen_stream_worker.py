from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.instrument.stream.worker import PersistentStreamGenerator, StreamModelSource, _request_payload
from macchina_ricordi.musicgen.backend import GenerationRequest


class _FakePipe:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines or []
        self.writes: list[str] = []
        self.flushed = False
        self.closed = False

    def write(self, value: str) -> int:
        self.writes.append(value)
        return len(value)

    def flush(self) -> None:
        self.flushed = True

    def readline(self) -> str:
        return self.lines.pop(0) if self.lines else ""

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, stdout_lines: list[str] | None = None, *, poll_value: int | None = None) -> None:
        self.stdin = _FakePipe()
        self.stdout = _FakePipe(stdout_lines)
        self.poll_value = poll_value
        self.wait_calls = 0
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.poll_value

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        self.poll_value = 0
        return 0

    def terminate(self) -> None:
        self.terminated = True
        self.poll_value = 143

    def kill(self) -> None:
        self.killed = True
        self.poll_value = 137


def test_request_payload_contains_stream_worker_contract(tmp_path: Path) -> None:
    request = GenerationRequest(
        prompts=["dry metal"],
        mode="continuation",
        duration_seconds=12.0,
        seed=11,
        prompt_audio_path="seed.wav",
        prompt_start_seconds=3.0,
        prompt_duration_seconds=2.0,
        top_k=80,
        min_p=0.07,
        temperature=1.4,
    )
    source = StreamModelSource(
        model_id="dummy/model",
        device="cuda",
        profile_name="micro-unconditional",
        xp_path="/tmp/xp",
        checkpoint_path="/tmp/xp/checkpoint.th",
    )

    payload = _request_payload(
        request_id="req-1",
        request=request,
        output_path=tmp_path / "round.wav",
        source=source,
    )

    assert payload["request_id"] == "req-1"
    assert payload["output_path"] == str((tmp_path / "round.wav").resolve())
    assert payload["mode"] == "continuation"
    assert payload["duration_seconds"] == 12.0
    assert payload["seed"] == 11
    assert payload["prompt_audio_path"] == "seed.wav"
    assert payload["prompt_start_seconds"] == 3.0
    assert payload["prompt_duration_seconds"] == 2.0
    assert payload["top_k"] == 80
    assert payload["min_p"] == 0.07
    assert payload["temperature"] == 1.4
    assert payload["model_id"] == "dummy/model"
    assert payload["device"] == "cuda"
    assert payload["profile"] == "micro-unconditional"
    assert payload["xp_path"] == "/tmp/xp"
    assert payload["checkpoint_path"] == "/tmp/xp/checkpoint.th"


def test_round_trip_surfaces_worker_error_response() -> None:
    response = json.dumps({"request_id": "req-1", "ok": False, "error": "boom"}) + "\n"
    process = _FakeProcess([response])
    generator = PersistentStreamGenerator()

    with pytest.raises(RuntimeError, match="boom"):
        generator._round_trip(process, {"request_id": "req-1"}, request_id="req-1")
