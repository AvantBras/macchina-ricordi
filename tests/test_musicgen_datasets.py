from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile

from macchina_ricordi.data import datasets
from macchina_ricordi.data.datasets import (
    BUILD_STATUS_FILENAME,
    DATASET_FILENAME,
    DatasetError,
    build_dataset,
)
from macchina_ricordi.data.prepare import prepare_musicgen_dataset
from macchina_ricordi.models import record as models
from macchina_ricordi.musicgen.templates import DATASET_CONFIG_NAME

_real_run = subprocess.run


def _fake_manifest_run(command, check=False, cwd=None, env=None, **kwargs):  # type: ignore[no-untyped-def]
    """Stands in for `python -m audiocraft.data.audio_dataset <folder> <manifest>`."""
    if "audiocraft.data.audio_dataset" not in command:
        return _real_run(command, check=check, cwd=cwd, env=env, **kwargs)
    source, manifest = Path(command[-2]), Path(command[-1])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(manifest, "wt", encoding="utf-8") as handle:
        for path in sorted(source.rglob("*.wav")):
            handle.write(json.dumps({"path": str(path), "duration": soundfile.info(path).duration}) + "\n")
    return subprocess.CompletedProcess(command, 0)


def _tone(path: Path, seconds: float = 3.0, sr: int = 8_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    soundfile.write(path, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for module in (datasets, models):
        monkeypatch.setattr(module, "get_repo_root", lambda explicit_root=None: tmp_path)
    monkeypatch.setenv("AUDIOCRAFT_DORA_DIR", str(tmp_path / ".dora"))
    monkeypatch.setattr("macchina_ricordi.data.prepare.subprocess.run", _fake_manifest_run)
    monkeypatch.chdir(tmp_path)
    for index in range(5):
        _tone(tmp_path / "library" / "tones" / f"tone_{index}.wav")
    (tmp_path / "python").write_text("")
    return tmp_path


def _build(repo: Path, name: str = "tones"):  # type: ignore[no-untyped-def]
    return build_dataset(name, [repo / "library" / "tones"], valid_fraction=0.2, musicgen_python=str(repo / "python"))


def test_build_makes_one_folder_with_linked_clips_split_and_manifests(repo: Path) -> None:
    dataset = _build(repo)

    folder = repo / "datasets" / "tones"
    assert dataset.path == folder.resolve() and dataset.name == "tones" and not dataset.legacy
    assert len(list((folder / "clips").glob("*.wav"))) == 5
    split = sorted((folder / "data").rglob("*.wav"))
    assert len(split) == 5
    clip_inodes = {path.stat().st_ino for path in (folder / "clips").glob("*.wav")}
    assert {path.stat().st_ino for path in split} <= clip_inodes  # hard links, not copies
    assert (folder / "egs" / DATASET_CONFIG_NAME / "train" / "data.jsonl.gz").exists()
    assert (dataset.train_clips, dataset.valid_clips, dataset.valid_fraction) == (4, 1, 0.2)
    assert dataset.train_seconds == pytest.approx(12.0)
    assert dataset.sources == (str((repo / "library" / "tones").resolve()),)
    assert dataset.fingerprint_digest
    record = json.loads((folder / DATASET_FILENAME).read_text())
    assert record["schema"] == "macchina_ricordi.dataset.v1" and record["egs_name"] == DATASET_CONFIG_NAME
    assert json.loads((folder / BUILD_STATUS_FILENAME).read_text())["status"] == "completed"


def test_a_dataset_is_never_rebuilt(repo: Path) -> None:
    _build(repo)
    with pytest.raises(DatasetError, match="never rebuilt"):
        _build(repo)
    (repo / "datasets" / "half").mkdir(parents=True)
    with pytest.raises(DatasetError, match="unfinished build"):
        _build(repo, "half")
    with pytest.raises(DatasetError, match="invalid dataset name"):
        _build(repo, ".hidden")


def test_prepare_refuses_to_wipe_a_prepared_workspace(repo: Path) -> None:
    dataset = _build(repo)
    with pytest.raises(FileExistsError, match="already prepared"):
        prepare_musicgen_dataset(
            input_dir=dataset.path / "clips", workspace_dir=dataset.path, musicgen_python=str(repo / "python")
        )
