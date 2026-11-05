from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from macchina_ricordi.models.ledger import (
    LEDGER_SCHEMA,
    LedgerEditError,
    _header_comments,
    load_experiment_ledger,
    render_ledger,
    save_experiment,
)


def test_load_ledger_resolves_paths_and_annotations(tmp_path: Path) -> None:
    ledger_path = tmp_path / "research" / "training" / "experiment_ledger.toml"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        f'''schema = "{LEDGER_SCHEMA}"

[[experiments]]
id = "quiet-basin"
title = "Escape the quiet basin"
status = "completed"
tags = ["distribution", "arc"]
xp_sigs = ["abc12345"]
checkpoint_paths = [".dora/xps/abc12345/checkpoint.th"]
sweep_ids = ["sweep_a"]
stream_ids = ["20260801T120000Z"]
artifact_paths = ["artifacts/sweeps/sweep_a/index.csv"]
notes_paths = ["research/training/next-experiments.md"]
verdict = "It opened."
next_action = "Listen again."
updated_at = "2026-08-01"

[experiments.annotations]
favorite = true
score = 4.2
''',
        encoding="utf-8",
    )

    ledger = load_experiment_ledger(tmp_path)

    assert ledger.schema == LEDGER_SCHEMA
    assert ledger.warnings == ()
    assert len(ledger.entries) == 1
    entry = ledger.entries[0]
    assert entry.experiment_id == "quiet-basin"
    assert entry.tags == ("distribution", "arc")
    assert entry.checkpoint_paths == (str((tmp_path / ".dora/xps/abc12345/checkpoint.th").resolve()),)
    assert entry.annotations == {"favorite": True, "score": 4.2}


def test_load_ledger_reports_validation_warnings_without_raising(tmp_path: Path) -> None:
    ledger_path = tmp_path / "research" / "training" / "experiment_ledger.toml"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        """schema = "wrong"

[[experiments]]
id = "Bad ID"
status = "mystery"

[[experiments]]
id = "good"

[[experiments]]
id = "good"
""",
        encoding="utf-8",
    )

    ledger = load_experiment_ledger(tmp_path)

    assert len(ledger.entries) == 1
    assert ledger.entries[0].experiment_id == "good"
    assert any("unknown status" in warning for warning in ledger.warnings)
    assert any("duplicate experiment id" in warning for warning in ledger.warnings)
    assert any("schema" in warning for warning in ledger.warnings)


# ── editing ───────────────────────────────────────────────────────────

_HAND_WRITTEN = f'''# Git-tracked human annotations.
# Second header line.

schema = "{LEDGER_SCHEMA}"

[[experiments]]
id = "long-lineage"
title = "A lineage with – an en dash"
status = "completed"
tags = ["a", "b"]
xp_sigs = [
  "11111111",
  "22222222",
  "33333333",
  "44444444",
  "55555555",
  "66666666",
  "77777777",
]
notes_paths = ["research/notes/x.md"]
verdict = "It \\"worked\\"."
updated_at = "2026-08-02"
custom_key = 3

[experiments.annotations]
favorite = true
best = ".dora/xps/11111111/checkpoint_30.th"
'''


def _ledger(tmp_path: Path, text: str = _HAND_WRITTEN) -> Path:
    path = tmp_path / "research" / "training" / "experiment_ledger.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_render_reproduces_a_hand_written_ledger(tmp_path: Path) -> None:
    assert render_ledger(tomllib.loads(_HAND_WRITTEN), header=_header_comments(_HAND_WRITTEN)) == _HAND_WRITTEN


def test_update_keeps_unknown_fields_and_header(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    result = save_experiment(tmp_path, "long-lineage", {"verdict": "Changed.", "tags": "a, c, c"}, today="2026-09-29")
    assert result.changed_fields == ("verdict", "tags")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Git-tracked human annotations.\n# Second header line.\n\nschema =")
    entry = tomllib.loads(text)["experiments"][0]
    assert entry["verdict"] == "Changed."
    assert entry["tags"] == ["a", "c"]
    assert entry["custom_key"] == 3
    assert entry["annotations"] == {"favorite": True, "best": ".dora/xps/11111111/checkpoint_30.th"}
    assert entry["updated_at"] == "2026-09-29"


def test_empty_values_remove_keys_and_noop_does_not_write(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    save_experiment(tmp_path, "long-lineage", {"notes_paths": [], "verdict": ""}, today="2026-09-29")
    entry = tomllib.loads(path.read_text(encoding="utf-8"))["experiments"][0]
    assert "notes_paths" not in entry and "verdict" not in entry
    before = path.read_text(encoding="utf-8")
    result = save_experiment(tmp_path, "long-lineage", {"title": entry["title"]}, today="2030-01-01")
    assert result.changed_fields == ()
    assert path.read_text(encoding="utf-8") == before


def test_create_appends_and_reads_back(tmp_path: Path) -> None:
    _ledger(tmp_path)
    save_experiment(tmp_path, "new-one", {"title": "New", "xp_sigs": "abc\ndef"}, create=True, today="2026-09-29")
    ledger = load_experiment_ledger(tmp_path)
    assert [e.experiment_id for e in ledger.entries] == ["long-lineage", "new-one"]
    created = ledger.entries[1]
    assert created.status == "planned"
    assert created.xp_sigs == ("abc", "def")
    assert created.updated_at == "2026-09-29"


def test_create_without_a_ledger_file(tmp_path: Path) -> None:
    save_experiment(tmp_path, "first", {"title": "First"}, create=True, today="2026-09-29")
    ledger = load_experiment_ledger(tmp_path)
    assert ledger.schema == LEDGER_SCHEMA
    assert ledger.entries[0].title == "First"


@pytest.mark.parametrize(
    ("experiment_id", "fields", "create", "message"),
    [
        ("Bad Id", {"title": "x"}, True, "invalid experiment id"),
        ("long-lineage", {"title": "x"}, True, "already exists"),
        ("missing", {"title": "x"}, False, "no experiment"),
        ("long-lineage", {"status": "done"}, False, "status must be one of"),
        ("long-lineage", {"annotations": {}}, False, "cannot edit"),
        ("long-lineage", {"tags": [1]}, False, "list of strings"),
    ],
)
def test_invalid_edits_are_refused_without_writing(
    tmp_path: Path, experiment_id: str, fields: dict, create: bool, message: str
) -> None:
    path = _ledger(tmp_path)
    with pytest.raises(LedgerEditError, match=message):
        save_experiment(tmp_path, experiment_id, fields, create=create)
    assert path.read_text(encoding="utf-8") == _HAND_WRITTEN
