from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from macchina_ricordi.scoring.batch import (
    iter_sweep_items,
    read_metric_lookup,
    rescore_paths,
    summarize_by_group,
    write_rows_csv,
)

SR = 16_000


def _write_sweep(root, sweep_id: str, *, samples: int = 2, broken: bool = False):
    sweep_dir = root / sweep_id
    sweep_dir.mkdir(parents=True)
    rows = []
    for index in range(samples):
        combo = sweep_dir / f"combo_{index:03d}"
        combo.mkdir()
        path = combo / "sample_000.wav"
        t = np.linspace(0.0, 3.0, int(SR * 3.0), endpoint=False)
        sf.write(path, (0.3 * np.sin(2 * np.pi * (220.0 + 40 * index) * t)).astype(np.float32), SR)
        rows.append(
            {
                "sweep_id": sweep_id,
                "combo_id": f"combo_{index:03d}",
                "path": str(path),
                "error": "",
                "temperature": str(1.0 + 0.2 * index),
                "score": str(3.0 + index),
            }
        )
    if broken:
        rows.append(
            {
                "sweep_id": sweep_id,
                "combo_id": "combo_bad",
                "path": str(sweep_dir / "missing.wav"),
                "error": "generation failed",
                "temperature": "1.0",
                "score": "",
            }
        )
    with (sweep_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sweep_dir


# ── rescoring ─────────────────────────────────────────────────────────


def test_rescore_paths_inline_produces_metrics(tmp_path):
    _write_sweep(tmp_path, "sweep_a")
    rows = rescore_paths(iter_sweep_items([tmp_path / "sweep_a"]), jobs=1)
    assert len(rows) == 2
    assert all(not row.error for row in rows)
    assert all("structure_score" in row.metrics and "arc_tv4" in row.metrics for row in rows)


# ── summarising ───────────────────────────────────────────────────────


def _rows(gate_values, metric_values, group="g"):
    return [
        {"group": group, "silence_gate": gate, "m": value}
        for gate, value in zip(gate_values, metric_values, strict=True)
    ]


def test_summarize_splits_groups():
    rows = _rows([1.0], [2.0], group="a") + _rows([1.0], [8.0], group="b")
    summary = summarize_by_group(rows, metrics=["m"])
    assert summary["a"]["m"] == pytest.approx(2.0)
    assert summary["b"]["m"] == pytest.approx(8.0)


# ── io ────────────────────────────────────────────────────────────────


def test_read_metric_lookup_roundtrip(tmp_path):
    path = write_rows_csv(
        tmp_path / "out.csv",
        [{"path": "x.wav", "m": 1.5, "note": "text", "empty": ""}],
    )
    lookup = read_metric_lookup(path)
    assert lookup["x.wav"]["m"] == pytest.approx(1.5)
    assert "note" not in lookup["x.wav"]
    assert "empty" not in lookup["x.wav"]
