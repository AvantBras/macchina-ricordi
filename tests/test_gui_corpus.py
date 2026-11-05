from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from macchina_ricordi.data.weights import load_weights_overlay
from macchina_ricordi.gui.corpus import (
    CorpusManager,
    CorpusRequest,
    CorpusRequestError,
)
from macchina_ricordi.musicgen.templates import DATASET_CONFIG_NAME

PROFILES = [
    SimpleNamespace(name="nano", overrides=("dataset.segment_duration=10",)),
    SimpleNamespace(name="long", overrides=("dataset.segment_duration=15",)),
]
# Stands in for `python -m macchina_ricordi.cli`: prints a progress line and exits 0, whatever it is asked.
FAKE_CLI = (sys.executable, "-c", "print('progress: 1/1')")


def _wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), np.zeros(int(seconds * 8000), dtype=np.float32), 8000)


def _manifest(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for clip, duration in rows:
            handle.write(json.dumps({"path": clip, "duration": duration, "weight": None}) + "\n")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _wav(tmp_path / "examples" / "sample_data" / "drones" / "a.wav", 12.0)
    _wav(tmp_path / "examples" / "sample_data" / "drones" / "b.wav", 8.0)
    _wav(tmp_path / "examples" / "sample_data" / "noise" / "c.wav", 30.0)
    workspace = tmp_path / ".workspace_old"
    train = workspace / "data" / "train"
    rows = [(str(train / f"{'loud' if i < 9 else 'quiet'}__{i}.wav"), 10.0 if i else 9.0) for i in range(12)]
    _manifest(workspace / "egs" / "ds" / "train" / "data.jsonl.gz", rows)
    _manifest(
        workspace / "egs" / "ds" / "valid" / "data.jsonl.gz", [(str(workspace / "data" / "valid" / "v.wav"), 10.0)]
    )
    (workspace / "logs").mkdir()
    (workspace / "logs" / "prepare_dataset.json").write_text(
        json.dumps({"input_dir": str(tmp_path / "gone"), "valid_fraction": 0.1, "seed": 42})
    )
    return tmp_path


def _manager(repo: Path, **kwargs) -> CorpusManager:
    kwargs.setdefault("profiles_loader", lambda: PROFILES)
    kwargs.setdefault("cli_prefix", FAKE_CLI)
    return CorpusManager(repo, **kwargs)


def _request(**payload) -> CorpusRequest:
    base = {
        "name": "fresh",
        "sources": ["examples/sample_data/drones"],
        "slice_seconds": 5,
        "dataset": DATASET_CONFIG_NAME,
    }
    return CorpusRequest.from_payload({**base, **payload})


def test_preview_blocks_existing_workspaces_and_names(repo: Path) -> None:
    manager = _manager(repo)
    preview = manager.corpus_preview(_request())
    assert preview["clips"] == 3  # 12 s -> 2 slices of 5 s, 8 s -> 1
    assert preview["blockers"] == [] and preview["key"]
    assert preview["workspace"] == "fresh"  # a dataset is named after its corpus
    assert preview["output_dir"] == "datasets/fresh"
    assert "10 s" in preview["warnings"][0]  # 5 s clips under a 10 s segment get padded

    (repo / "datasets" / "fresh").mkdir(parents=True)
    (repo / "datasets" / "fresh" / "dataset.json").write_text("{}")
    blocked = manager.corpus_preview(_request())
    assert blocked["key"] is None and any("never rebuilt" in b for b in blocked["blockers"])
    with pytest.raises(CorpusRequestError, match="library"):
        manager.corpus_preview(_request(sources=["../outside"]))


def test_mix_save_keeps_each_source_at_its_share(repo: Path) -> None:
    manager = _manager(repo)
    with pytest.raises(CorpusRequestError, match="not in the mix: quiet"):
        manager.mix_save(".workspace_old", {"dataset": "ds", "mix": {"loud": 1}, "name": "loud_only"})
    saved = manager.mix_save(".workspace_old", {"dataset": "ds", "mix": {"loud": 1, "quiet": 1}, "name": "half"})
    assert saved["name"] == "half"
    overlay = load_weights_overlay(repo, ".workspace_old", "half")
    assert (overlay.mode, overlay.mix) == ("folder-mix", {"loud": 1, "quiet": 1})
