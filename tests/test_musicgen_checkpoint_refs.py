from __future__ import annotations

import json
from pathlib import Path

from macchina_ricordi.tidy.refs import scan_checkpoint_references


def _xp(repo: Path, sig: str) -> Path:
    folder = repo / ".dora" / "xps" / sig
    folder.mkdir(parents=True)
    (folder / "checkpoint_10.th").write_bytes(b"x")
    return folder


def test_references_from_records_children_and_notes(tmp_path: Path) -> None:
    parent = _xp(tmp_path, "aaaa1111")
    child = _xp(tmp_path, "bbbb2222")
    (child / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps({"init_from_checkpoint": {"resolved_local_path": str(parent / "checkpoint_10.th")}})
    )
    run = tmp_path / "artifacts" / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"checkpoint_path": ".dora/xps/bbbb2222/checkpoint.th"}))
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "note.md").write_text("aaaa1111 sounds best; deadbeef is not an XP\n")

    refs = scan_checkpoint_references(tmp_path)

    assert refs.for_checkpoint("aaaa1111", "checkpoint_10.th") == (
        ".dora/xps/bbbb2222/macchina_ricordi_train_metadata.json",
    )
    assert refs.for_checkpoint("bbbb2222", "checkpoint.th") == ("artifacts/runs/r1/metadata.json",)
    assert set(refs.for_xp("aaaa1111")) == {
        ".dora/xps/bbbb2222/macchina_ricordi_train_metadata.json",
        "research/note.md",
    }
    assert refs.for_xp("deadbeef") == ()
