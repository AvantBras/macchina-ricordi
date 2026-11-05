from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from macchina_ricordi.gui.housekeeping import (
    HousekeepingError,
    HousekeepingManager,
    HouseRequest,
)

OLD = time.time() - 10 * 86400


def _xp(repo: Path, sig: str, epochs: list[int], *, valid_ce: list[float] | None = None, init: Path | None = None):
    folder = repo / ".dora" / "xps" / sig
    folder.mkdir(parents=True)
    metadata = {"effective_label": sig, "profile": "p"}
    if init is not None:
        metadata["init_from_checkpoint"] = {"resolved_local_path": str(init)}
    (folder / "macchina_ricordi_train_metadata.json").write_text(json.dumps(metadata))
    (folder / "history.json").write_text(json.dumps([{"valid": {"ce": ce}} for ce in (valid_ce or [5.0] * 20)]))
    for epoch in epochs:
        (folder / f"checkpoint_{epoch}.th").write_bytes(b"e" * (100 + epoch))
    (folder / "checkpoint.th").write_bytes(b"latest" * 50)
    for path in [folder, *folder.rglob("*")]:
        os.utime(path, (OLD, OLD))
    return folder


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    parent = _xp(tmp_path, "aaaa1111", [5, 10, 15, 20], valid_ce=[5, 5, 5, 5, 5, 5, 5, 5, 5, 3.0] + [4.0] * 10)
    _xp(tmp_path, "bbbb2222", [5, 10], init=parent / "checkpoint_15.th")
    _xp(tmp_path, "cccc3333", [])
    husk = tmp_path / ".dora" / "xps" / "dddd4444"
    husk.mkdir()
    (husk / "checkpoint_cleanup_report.json").write_text("{}")
    return tmp_path


def _manager(repo: Path, **kwargs) -> HousekeepingManager:
    return HousekeepingManager(repo, **kwargs)


def _request(**payload) -> HouseRequest:
    return HouseRequest.from_payload(payload)


def test_commit_requires_the_previewed_plan(repo: Path) -> None:
    manager = _manager(repo)
    request = _request(action="checkpoints", keep_last=1, keep_best=True)
    key = manager.plan(request)["key"]
    with pytest.raises(HousekeepingError, match="preview again"):
        manager.commit(request, "stale")
    # something changed on disk: the plan no longer matches the key
    extra = repo / ".dora" / "xps" / "bbbb2222" / "checkpoint_1.th"
    extra.write_bytes(b"new")
    os.utime(extra, (OLD, OLD))
    with pytest.raises(HousekeepingError, match="preview again"):
        manager.commit(request, key)

    key = manager.plan(request)["key"]
    result = manager.commit(request, key)
    assert result["removed_count"] == 3
    xps = repo / ".dora" / "xps"
    assert not (xps / "aaaa1111" / "checkpoint_5.th").exists()
    assert (xps / "aaaa1111" / "checkpoint_15.th").exists()
    assert (xps / "aaaa1111" / "checkpoint.th").exists()
    # only XPs that lost a file carry a cleanup report
    assert (xps / "aaaa1111" / "checkpoint_cleanup_report.json").exists()
    assert not (xps / "cccc3333" / "checkpoint_cleanup_report.json").exists()


def test_archive_refuses_the_same_disk(repo: Path) -> None:
    plan = _manager(repo).plan(_request(action="archive", destination=str(repo / "archive")))
    assert "same filesystem" in plan["destination_error"]
    assert plan["key"] is None
    with pytest.raises(HousekeepingError, match="absolute"):
        _manager(repo).plan(_request(action="archive", destination="relative/dir"))
