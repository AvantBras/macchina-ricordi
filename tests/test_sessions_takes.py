from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.scoring.batch import iter_sweep_items, sweep_base_metrics
from macchina_ricordi.sessions import store
from macchina_ricordi.sessions.takes import playable_takes


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(store, "get_repo_root", lambda: tmp_path)
    stream = tmp_path / "artifacts" / "streams" / "20261005T141908Z"
    (stream / "rounds").mkdir(parents=True)
    (stream / "manifest.json").write_text(json.dumps({"run_id": "20261005T141908Z"}))
    rows = []
    for index in range(3):
        (stream / "rounds" / f"{index}.wav").write_bytes(b"RIFF")
        rows.append(
            {
                "round_idx": index,
                "round_wav": f"rounds/{index}.wav",
                "model_xp": "/r/.dora/xps/abcd1234",
                "prompt_duration_seconds": 2.5,
                "score": {"timbral_coherence": 0.9} if index != 1 else {},
            }
        )
    (stream / "performance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    run = tmp_path / "artifacts" / "runs" / "2026-10-05" / "night_20261005T150000Z"
    (run / "samples").mkdir(parents=True)
    (run / "samples" / "000.wav").write_bytes(b"RIFF")
    (run / "samples" / "000_listening_report.json").write_text(json.dumps({"whole_clip": {"reward": {"score": 0.4}}}))
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": "20261005T150000Z",
                "request": {"mode": "unconditional", "duration_seconds": 8.0},
                "generation": {"source": {"xp_path": "/r/.dora/xps/abcd1234"}},
                "files": ["samples/000.wav"],
            }
        )
    )
    return tmp_path


def test_stream_rounds_keep_the_spelling_the_preference_store_recorded(repo: Path) -> None:
    by_id = playable_takes(["20261005T141908Z"])
    by_folder = playable_takes(["artifacts/streams/20261005T141908Z"])
    assert [take.path for take in by_id] == [take.path for take in by_folder]
    assert by_folder[0].path == "artifacts/streams/20261005T141908Z/rounds/0.wav"
    assert [(take.position, take.round_idx) for take in by_folder] == [(0, 0), (1, 1), (2, 2)]
    assert by_folder[0].prompt_duration_seconds == 2.5 and by_folder[0].group == "abcd1234"
    assert [take.position for take in playable_takes(["20261005T141908Z"], scored_only=True)] == [0, 2]


def test_every_kind_comes_from_one_listing(repo: Path) -> None:
    sources = ["20261005T141908Z", "night_20261005T150000Z"]
    assert {take.kind for take in playable_takes(sources)} == {"stream", "generate"}
    items = dict(iter_sweep_items(sources))
    assert len(items) == 4
    metrics = sweep_base_metrics(sources)
    generated = next(path for path in metrics if path.endswith("000.wav"))
    assert metrics[generated] == {"listening_score": 0.4}
