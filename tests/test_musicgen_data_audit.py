from __future__ import annotations

import gzip
import json
from pathlib import Path

from macchina_ricordi.data.prepare import audit_workspace_slice_lengths


def _make_workspace(
    root: Path,
    name: str,
    *,
    dataset_name: str = "ds",
    train_durations: list[float] | None = None,
) -> Path:
    workspace = root / name
    if train_durations is not None:
        manifest = workspace / "egs" / dataset_name / "train" / "data.jsonl.gz"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(manifest, "wt", encoding="utf-8") as handle:
            for index, duration in enumerate(train_durations):
                handle.write(json.dumps({"path": f"c{index}.wav", "duration": duration}) + "\n")
    else:
        workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def test_audit_scans_and_sorts_by_slice_length(tmp_path: Path) -> None:
    _make_workspace(tmp_path, ".workspace_20s", train_durations=[20.0] * 30)
    _make_workspace(tmp_path, ".workspace_5s", train_durations=[5.0] * 30)
    # 10s clips plus a short-fragment tail (2 of 22 are < 5s).
    _make_workspace(tmp_path, ".macchina_ricordi_10s", train_durations=[10.0] * 20 + [2.0, 0.0])
    _make_workspace(tmp_path, ".workspace_empty")  # no manifest

    rows = audit_workspace_slice_lengths(root=tmp_path)
    by_name = {row["workspace"]: row for row in rows}

    assert by_name[".workspace_5s"]["safe_segment_duration"] == 5
    assert by_name[".macchina_ricordi_10s"]["safe_segment_duration"] == 10
    assert by_name[".macchina_ricordi_10s"]["stats"]["n_shorter_than_half"] == 2
    assert by_name[".workspace_20s"]["safe_segment_duration"] == 20
    assert by_name[".workspace_empty"]["stats"] is None

    # Sorted ascending by slice length, with the manifest-less workspace last.
    safe_order = [r["safe_segment_duration"] for r in rows]
    assert safe_order == [5, 10, 20, None]
