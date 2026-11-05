from __future__ import annotations

import json
from pathlib import Path

from macchina_ricordi.instrument.provenance import (
    infer_checkpoint_xp_path,
    infer_xp_profile,
    resolve_generation_profile,
)


def test_infer_xp_profile_from_macchina_ricordi_metadata(tmp_path: Path) -> None:
    xp = tmp_path / ".dora" / "xps" / "abc123"
    xp.mkdir(parents=True)
    (xp / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps({"profile": "micro-unconditional"}),
        encoding="utf-8",
    )

    assert infer_xp_profile(xp) == "micro-unconditional"


def test_infer_checkpoint_xp_path_from_merge_sidecar(tmp_path: Path) -> None:
    xp = tmp_path / ".dora" / "xps" / "abc123"
    xp.mkdir(parents=True)
    source_checkpoint = xp / "checkpoint_30.th"
    source_checkpoint.write_bytes(b"source")
    (xp / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps({"profile": "micro-unconditional"}),
        encoding="utf-8",
    )

    merged = tmp_path / "artifacts" / "checkpoints" / "merged" / "blend"
    merged.mkdir(parents=True)
    checkpoint = merged / "checkpoint.th"
    checkpoint.write_bytes(b"merged")
    (merged / "merged_from.json").write_text(
        json.dumps({"sources": {"a": {"path": str(source_checkpoint)}}}),
        encoding="utf-8",
    )

    assert infer_checkpoint_xp_path(checkpoint) == xp.resolve()
    assert resolve_generation_profile(profile_name=None, checkpoint_path=checkpoint) == "micro-unconditional"
