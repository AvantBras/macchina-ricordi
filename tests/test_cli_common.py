from __future__ import annotations

import argparse
import csv
from pathlib import Path

from macchina_ricordi.cli.common import (
    add_shared_scoring_args,
    bool_env,
    format_metric_value,
    write_csv,
)


def test_format_metric_value_formats_scalar_types() -> None:
    expected_float = f"{3.14159265:.6f}"

    assert format_metric_value(7) == "7"
    assert format_metric_value(3.14159265) == expected_float
    assert format_metric_value("clip.wav") == "clip.wav"


def test_write_csv_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "scores.csv"

    write_csv(output_path, ["path", "score"], [{"path": "clip.wav", "score": 1.25}])

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"path": "clip.wav", "score": "1.25"}]


def test_bool_env_honors_truthy_falsy_and_default_values(monkeypatch) -> None:
    monkeypatch.delenv("MACCHINA_RICORDI_FLAG", raising=False)
    assert bool_env("MACCHINA_RICORDI_FLAG", default=True) is True

    monkeypatch.setenv("MACCHINA_RICORDI_FLAG", "yes")
    assert bool_env("MACCHINA_RICORDI_FLAG", default=False) is True

    monkeypatch.setenv("MACCHINA_RICORDI_FLAG", "off")
    assert bool_env("MACCHINA_RICORDI_FLAG", default=True) is False


def test_add_shared_scoring_args_registers_expected_defaults() -> None:
    parser = argparse.ArgumentParser()

    add_shared_scoring_args(parser)
    args = parser.parse_args([])

    assert args.sr == 32_000
    assert args.hop_length == 512
    assert args.res_type == "soxr_hq"
    assert args.tempo_method == "tempogram"
