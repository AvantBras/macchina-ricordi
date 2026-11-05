from __future__ import annotations

import csv
import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.instrument.run_paths import OUTPUT_LAYOUT_SESSION
from macchina_ricordi.instrument.saving import save_generation_run
from macchina_ricordi.musicgen.backend import GenerationRequest, GenerationResult
from macchina_ricordi.sessions import store
from macchina_ricordi.sessions.read import load_session
from macchina_ricordi.sessions.record import SESSION_FILENAME, TAKES_FILENAME, SessionNotFound
from macchina_ricordi.sessions.store import (
    find_session,
    list_sessions,
    new_session_dir,
)

XP = "/repo/.dora/xps/aaaa1111"


def _wav(path: Path, frames: int = 320) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(32000)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def _json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _legacy_run(repo: Path) -> Path:
    folder = repo / "artifacts" / "runs" / "2026-06-01" / "musicgen_20260601T100000Z"
    for index in range(2):
        _wav(folder / "samples" / f"{index:03d}.wav")
    _json(
        folder / "run.json",
        {
            "run_id": "20260601T100000Z",
            "backend": "audiocraft",
            "model_id": "facebook/musicgen-small",
            "device": "cuda",
            "sample_rate": 32000,
            "request": {"mode": "unconditional", "seed": 3, "temperature": 1.2, "duration_seconds": 5.0},
            "generation": {"source": {"kind": "xp", "xp_path": XP}},
            "files": ["samples/000.wav", "samples/001.wav"],
            "output": {"layout": "dated-run-dir", "prefix": "musicgen"},
        },
    )
    return folder


def _legacy_sweep(repo: Path) -> Path:
    folder = repo / "artifacts" / "sweeps" / "demo_20260602T100000Z"
    _json(
        folder / "sweep_plan.json",
        {
            "sweep_id": "demo_20260602T100000Z",
            "created_at": "2026-06-02T10:00:00+00:00",
            "mode": "unconditional",
            "model_id": "facebook/musicgen-small",
            "source": {"xp_path": XP},
            "combinations": [
                {"temperature": 1.0, "seed": 0, "combo_index": 0, "combo_id": "combo_000"},
                {"temperature": 1.5, "seed": 0, "combo_index": 1, "combo_id": "combo_001"},
            ],
        },
    )
    _json(folder / "sweep_report.json", {"status": "completed"})
    rows = []
    for index in range(2):
        audio = _wav(folder / f"combo_{index:03d}" / "sample_000.wav")
        _json(folder / f"combo_{index:03d}" / "metadata.json", {"run_id": f"x_combo_{index:03d}", "files": []})
        rows.append(
            {
                "sweep_id": "demo_20260602T100000Z",
                "combo_id": f"combo_{index:03d}",
                "sample_index": "0",
                "num_samples": "1",
                "path": str(audio),
                "error": "",
                "xp_path": XP,
                "seed": "0",
                "temperature": str(1.0 + index / 2),
                "score": str(5.0 + index),
            }
        )
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return folder


def _stream(folder: Path, run_id: str, *, rounds: int = 3, manifest: bool = True) -> Path:
    lines = []
    for index in range(rounds):
        _wav(folder / "rounds" / f"round_{index:03d}.wav")
        lines.append(
            json.dumps(
                {
                    "round_idx": index,
                    "round_wav": f"rounds/round_{index:03d}.wav",
                    "model_xp": XP,
                    "active_control": {"temperature": 1.3, "model_xp": XP},
                    "score": {"novelty_raw": 0.1 * index, "onset_density": 2.0, "candidates": [{"x": 1}]},
                }
            )
        )
    (folder / "performance.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if manifest:
        _json(
            folder / "manifest.json",
            {
                "schema": "macchina_ricordi.stream_session.v1",
                "run_id": run_id,
                "started_at": f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}T{run_id[9:11]}:00:00Z",
                "finished_at": f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}T{run_id[9:11]}:30:00Z",
                "initial_control": {"temperature": 1.3, "model_xp": XP},
                "backend": {"xp_path": XP},
            },
        )
    return folder


def _duet(repo: Path) -> Path:
    folder = repo / "artifacts" / "duets" / "20260711_smoke"
    for name in ("mix.wav", "voice_a.wav", "voice_b.wav"):
        _wav(folder / name)
    _json(
        folder / "manifest.json",
        {
            "schema": "macchina_ricordi.duet_session.v0",
            "rounds": 2,
            "seed": 11,
            "mix": "mix.wav",
            "voices": {
                "a": {"stem": "voice_a.wav", "source": {"xp_path": XP}, "temperature": 1.1},
                "b": {"stem": "voice_b.wav", "source": {"xp_path": "/repo/.dora/xps/bbbb2222"}, "listen": 0.5},
            },
        },
    )
    return folder


def _variations(repo: Path, parent: Path) -> Path:
    folder = repo / "artifacts" / "variations" / "20260712_smoke"
    for name in ("variation_00.wav", "variation_01.wav", "reference_roundtrip.wav"):
        _wav(folder / name)
    _json(
        folder / "manifest.json",
        {
            "schema": "macchina_ricordi.variations.v0",
            "input": str(parent),
            "source": {"xp_path": XP},
            "freeze_books": 1,
            "temperature": 1.3,
            "seed": 7,
            "reference_roundtrip": "reference_roundtrip.wav",
            "files": ["variation_00.wav", "variation_01.wav"],
        },
    )
    return folder


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(store, "get_repo_root", lambda: tmp_path)
    _legacy_run(tmp_path)
    _legacy_sweep(tmp_path)
    old = _stream(tmp_path / "artifacts" / "streams" / "20260603T100000Z", "20260603T100000Z")
    _duet(tmp_path)
    _variations(tmp_path, old / "rounds" / "round_001.wav")
    # The hand-kept notebook lives beside the dated folders and is not a session.
    _wav(tmp_path / "artifacts" / "sessions" / "listening_odyssey" / "keeper.wav")
    _json(tmp_path / "artifacts" / "sessions" / "listening_odyssey" / "run.json", {"note": "not a run"})
    return tmp_path


def test_sessions_are_listed_newest_first_and_filtered(repo: Path) -> None:
    newest = list_sessions(repo)[0]
    assert newest.kind in ("duet", "variations")
    assert [session.kind for session in list_sessions(repo, kind="stream")] == ["stream"]
    assert {session.kind for session in list_sessions(repo, sig="bbbb2222")} == {"duet"}


def test_find_session_by_id_run_id_and_unique_part(repo: Path) -> None:
    assert find_session("demo_20260602T100000Z", repo_root=repo).kind == "sweep"
    assert find_session("20260603T100000Z", repo_root=repo).kind == "stream"
    assert find_session("artifacts/duets/20260711_smoke", repo_root=repo).kind == "duet"
    with pytest.raises(SessionNotFound, match="several sessions"):
        find_session("2026060", repo_root=repo)
    with pytest.raises(SessionNotFound, match="Did you mean"):
        find_session("demo_20260602T100000X", repo_root=repo)


def test_generate_writes_a_sealed_session(repo: Path) -> None:
    request = GenerationRequest(prompts=[""], batch_size=2, duration_seconds=1.0, seed=5, mode="unconditional")
    result = GenerationResult(
        audio_values=np.zeros((2, 320), dtype=np.float32),
        sample_rate=32000,
        backend="audiocraft",
        model_id="facebook/musicgen-small",
        device="cpu",
        warnings=[],
        metadata={"source": {"kind": "xp", "xp_path": XP}},
    )
    saved = save_generation_run(
        result,
        request,
        output_dir=repo / "artifacts" / "sessions",
        output_layout=OUTPUT_LAYOUT_SESSION,
        now=datetime(2026, 10, 6, 9, 0, tzinfo=UTC),
    )
    folder = repo / "artifacts" / "sessions" / "2026-10-06" / "generate_20261006T090000Z"
    assert saved.output_dir == folder
    assert [path.relative_to(folder).as_posix() for path in saved.audio_paths] == ["takes/000.wav", "takes/001.wav"]
    record = json.loads((folder / SESSION_FILENAME).read_text())
    assert (record["kind"], record["take_count"], record["takes_log"], record["native"]) == (
        "generate",
        2,
        TAKES_FILENAME,
        "run.json",
    )
    session = find_session("generate_20261006T090000Z", repo_root=repo)
    assert session.sealed and session.sigs == ("aaaa1111",)
    takes = session.takes()
    assert [take.seed for take in takes] == [5, 5]
    assert takes[0].audio == folder / "takes" / "000.wav"
    assert takes[0].replay == folder / "run.json"
    assert list_sessions(repo)[0].id == "generate_20261006T090000Z"


def test_a_rerolled_round_is_another_take(repo: Path) -> None:
    folder = _stream(new_session_dir("stream", run_id="20261006T120000Z", repo_root=repo), "20261006T120000Z")
    reroll = {"round_idx": 1, "take_idx": 1, "round_wav": "rounds/round_001_take_1.wav", "score": {}}
    _wav(folder / "rounds" / "round_001_take_1.wav")
    with (folder / "performance.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(reroll) + "\n")
    takes = load_session(folder).takes()
    assert [take.label for take in takes] == ["round 0", "round 1", "round 2", "round 1 · take 1"]
    assert [take.index for take in takes] == [0, 1, 2, 3]
