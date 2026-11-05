from __future__ import annotations

import json

import pytest

from macchina_ricordi.instrument.stream.dataset import (
    coverage_table,
    discover_sessions,
    load_session_meta,
    load_stream_dataset,
    write_stream_dataset,
)


def _score(**overrides):
    payload = {
        "score": 3.0,
        "onset_density": 5.0,
        "onset_count": 75,
        "duration_seconds": 15.0,
        "silence_fraction": 0.0,
        "max_silence_gap_seconds": 0.0,
        "arc_coherence_score": 1.5,
        "timbral_coherence": 0.9,
        "novelty_whitened": 0.2,
        "candidates": [
            {"candidate": 0, "selected": False, "silence_fraction": 0.5, "arc_coherence_score": 0.4},
            {"candidate": 1, "selected": True, "silence_fraction": 0.0, "arc_coherence_score": 1.5},
        ],
    }
    payload.update(overrides)
    return payload


def _write_session(root, run_id, *, n=8, control=None, manifest=True, scored=True):
    session = root / run_id
    (session / "rounds").mkdir(parents=True)
    base_control = {
        "command": "continue",
        "tau_seconds": 5.0,
        "completion_duration_seconds": 10.0,
        "temperature": 1.0,
        "top_k": 250,
        "candidates": 2,
        "model_xp": ".dora/xps/abc12345",
    }
    base_control.update(control or {})

    lines = []
    for i in range(n):
        row = {
            "round_idx": i,
            "take_idx": 0,
            "timestamp": f"2026-08-01T08:{i:02d}:00Z",
            "active_control": dict(base_control, temperature=1.0 + 0.1 * i),
            "round_wav": f"rounds/round_{i:03d}.wav",
            "model_xp": f"/abs/path/{base_control['model_xp']}",
            "score": _score() if scored else {},
        }
        lines.append(json.dumps(row))
    (session / "performance.jsonl").write_text("\n".join(lines) + "\n")

    if manifest:
        (session / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "seed_mode": "uncond",
                    "started_at": "2026-08-01T08:00:00Z",
                    "final_xp_path": f"/abs/path/{base_control['model_xp']}",
                    "novelty_calibration": {"path": "calib.json", "dims": 117},
                }
            )
        )
    (session / "events.jsonl").write_text(
        json.dumps({"timestamp": "2026-08-01T08:00:00Z", "round_idx": 1, "kind": "throttle", "level": "info"}) + "\n"
    )
    return session


@pytest.fixture()
def streams_root(tmp_path):
    root = tmp_path / "streams"
    root.mkdir()
    _write_session(root, "20260801T000000Z", n=8)
    _write_session(root, "20260802T000000Z", n=6, control={"model_xp": ".dora/xps/def67890"}, manifest=False)
    _write_session(root, "20260601T000000Z", n=9, scored=False)
    return root


def test_discover_skips_unscored_sessions(streams_root):
    metas = discover_sessions(streams_root)
    assert [m.run_id for m in metas] == ["20260801T000000Z", "20260802T000000Z"]


def test_session_meta_resolves_model_from_rows_without_manifest(streams_root):
    meta = load_session_meta(streams_root / "20260802T000000Z")
    assert meta.has_manifest is False
    assert meta.manifest_xp is None
    assert meta.model_xp == "def67890"
    assert meta.novelty_calibration_path is None


def test_load_flattens_rounds_candidates_and_events(streams_root):
    ds = load_stream_dataset(streams_root)
    assert len(ds.rounds) == 14
    assert len(ds.candidates) == 28
    assert len(ds.events) == 2
    assert ds.rounds["model"].nunique() == 2
    assert "ctl_temperature" in ds.control_columns
    assert "timbral_coherence" in ds.indicator_columns
    assert "ctl_temperature" not in ds.indicator_columns


def test_derived_columns(streams_root):
    ds = load_stream_dataset(streams_root)
    row = ds.rounds.iloc[0]
    assert row["d_onset_count_per_second"] == pytest.approx(5.0)
    assert row["d_round_duration_seconds"] == pytest.approx(15.0)
    assert row["d_completion_duration_seconds"] == pytest.approx(10.0)
    assert ds.rounds.groupby("run_id")["d_round_seq"].max().tolist() == [7, 5]


def test_legacy_round_duration_is_reconciled(tmp_path):
    root = tmp_path / "streams"
    root.mkdir()
    _write_session(
        root,
        "20260728T000000Z",
        n=6,
        control={"completion_duration_seconds": None, "round_duration_seconds": 15.0},
    )
    ds = load_stream_dataset(root)
    assert ds.rounds["d_round_duration_seconds"].eq(15.0).all()
    assert ds.rounds["d_completion_duration_seconds"].eq(10.0).all()


def test_sessions_filter(streams_root):
    ds = load_stream_dataset(streams_root, sessions=["20260801T000000Z"])
    assert ds.rounds["run_id"].unique().tolist() == ["20260801T000000Z"]


def test_coverage_table_flags_partial_fields(streams_root):
    ds = load_stream_dataset(streams_root)
    cov = coverage_table(ds)
    calib = cov[(cov["column"] == "meta_novelty_calibration") & (cov["run_id"] == "__all__")].iloc[0]
    assert calib["coverage"] == pytest.approx(8 / 14)
    assert set(cov["run_id"]) == {"__all__", "20260801T000000Z", "20260802T000000Z"}


def test_write_dataset_emits_sidecar(streams_root, tmp_path):
    ds = load_stream_dataset(streams_root)
    status_path = write_stream_dataset(ds, tmp_path / "out")
    status = json.loads(status_path.read_text())
    assert status["schema"] == "macchina_ricordi.stream_dataset.v1"
    assert status["rounds"] == 14
    assert [s["run_id"] for s in status["sessions"]] == ["20260801T000000Z", "20260802T000000Z"]
    assert (tmp_path / "out" / "rounds.csv").exists()
    assert (tmp_path / "out" / "coverage.csv").exists()


def test_empty_root(tmp_path):
    ds = load_stream_dataset(tmp_path / "nope")
    assert ds.rounds.empty
    assert coverage_table(ds).empty
