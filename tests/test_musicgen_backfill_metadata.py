from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.tidy.backfill import (
    BackfillReport,
    BackfillResult,
    _build_backfill_init_metadata,
    _infer_workspace_from_datasource,
    _load_dataset_fingerprint,
    _load_merged_from_sidecar,
    _load_perturbed_from_sidecar,
    _parse_overrides,
    _reconstruct_command,
    backfill_xps,
    build_backfill_metadata,
)
from macchina_ricordi.training.plan import TRAIN_METADATA_FILENAME


class TestParseOverrides:
    def test_empty_or_missing_file_means_no_overrides(self, tmp_path: Path) -> None:
        empty = tmp_path / "overrides.yaml"
        empty.write_text("", encoding="utf-8")
        assert _parse_overrides(empty) == {}
        assert _parse_overrides(tmp_path / "nope.yaml") == {}

    def test_parses_entries_last_duplicate_wins_and_keeps_paths(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.yaml"
        path.write_text(
            "- solver=musicgen/musicgen_base_32khz\n- optim.epochs=5\n- optim.epochs=50\n"
            "- label=my_run\n- continue_from=/path/to/checkpoint.th\n",
            encoding="utf-8",
        )
        parsed = _parse_overrides(path)
        assert parsed["solver"] == "musicgen/musicgen_base_32khz"
        assert parsed["optim.epochs"] == "50"
        assert parsed["label"] == "my_run"
        assert parsed["continue_from"] == "/path/to/checkpoint.th"


class TestInferWorkspace:
    def test_typical_egs_path(self, tmp_path: Path) -> None:
        ws = tmp_path / "my_workspace"
        egs = ws / "egs" / "macchina_ricordi_micro_uncond" / "train"
        egs.mkdir(parents=True)
        (ws / "logs").mkdir()
        result = _infer_workspace_from_datasource(str(egs))
        assert result == ws

    def test_needs_a_logs_dir_and_a_datasource(self, tmp_path: Path) -> None:
        assert _infer_workspace_from_datasource(None) is None
        ws = tmp_path / "bare"
        egs = ws / "egs" / "ds" / "train"
        egs.mkdir(parents=True)
        # No logs/ dir, so can't identify as workspace
        assert _infer_workspace_from_datasource(str(egs)) is None


def test_a_pretrained_uri_or_missing_checkpoint_has_no_sidecar(tmp_path: Path) -> None:
    assert _load_merged_from_sidecar(None) is None
    assert _load_merged_from_sidecar("//pretrained/facebook/encodec_32khz") is None
    assert _load_merged_from_sidecar(str(tmp_path / "nope" / "checkpoint.th")) is None
    assert _load_perturbed_from_sidecar("//pretrained/facebook/encodec_32khz") is None


def test_merged_and_perturbed_sidecars_load_next_to_their_checkpoint(tmp_path: Path) -> None:
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "merged_from.json").write_text(json.dumps({"method": "slerp", "alpha": 0.7}), encoding="utf-8")
    merged = _load_merged_from_sidecar(str(merged_dir / "checkpoint.th"))
    assert merged is not None
    assert merged["method"] == "slerp"

    perturbed_dir = tmp_path / "perturbed"
    perturbed_dir.mkdir()
    (perturbed_dir / "perturbed_from.json").write_text(
        json.dumps({"sigma": 0.02, "layer_scope": "lm-head"}),
        encoding="utf-8",
    )
    perturbed = _load_perturbed_from_sidecar(str(perturbed_dir / "checkpoint.th"))
    assert perturbed is not None
    assert perturbed["sigma"] == 0.02


class TestLoadFingerprint:
    def test_missing_workspace_or_fingerprint_is_none(self, tmp_path: Path) -> None:
        assert _load_dataset_fingerprint(None) is None
        ws = tmp_path / "ws"
        (ws / "logs").mkdir(parents=True)
        assert _load_dataset_fingerprint(ws) is None

    def test_loads_valid_fingerprint(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        (ws / "logs").mkdir(parents=True)
        (ws / "logs" / "dataset_fingerprint.json").write_text(
            json.dumps({"train_aggregate_sha256": "abc123", "train_count": 10}),
            encoding="utf-8",
        )
        fp = _load_dataset_fingerprint(ws)
        assert fp is not None
        assert fp["train_aggregate_sha256"] == "abc123"


class TestReconstructCommand:
    def test_overrides_are_rendered_in_known_order(self) -> None:
        overrides = {"optim.epochs": "50", "label": "my_run", "continue_from": "/tmp/ckpt.th"}
        cmd = _reconstruct_command(overrides)
        assert "optim.epochs=50" in cmd
        assert "label=my_run" in cmd
        assert "continue_from=/tmp/ckpt.th" in cmd
        assert "solver=musicgen/musicgen_base_32khz" in cmd
        # Both label and optim.epochs are known keys, both present
        label_idx = cmd.index("label=my_run")
        epochs_idx = cmd.index("optim.epochs=50")
        assert epochs_idx < label_idx  # optim.epochs comes before label in known_order


class TestBuildBackfillMetadata:
    def test_basic(self, tmp_path: Path) -> None:
        xp = tmp_path / "abc123"
        xp.mkdir()
        overrides = {"optim.epochs": "50", "label": "test_run"}
        md = build_backfill_metadata(xp, overrides, datasource_train=None)
        assert md["xp_sig"] == "abc123"
        assert md["effective_label"] == "test_run"
        assert "init_from_checkpoint" not in md

    def test_continue_from_without_a_sidecar_is_a_local_backfill(self, tmp_path: Path) -> None:
        xp = tmp_path / "abc123"
        xp.mkdir()
        overrides = {"continue_from": "/some/xp/checkpoint.th"}
        md = build_backfill_metadata(xp, overrides, datasource_train=None)
        assert "init_from_checkpoint" in md
        assert md["init_from_checkpoint"]["origin"] == "local"
        assert md["init_from_checkpoint"]["_backfilled"] is True

    @pytest.mark.parametrize(
        ("sidecar", "payload", "key", "field", "expected"),
        [
            ("merged_from.json", {"method": "slerp", "alpha": 0.7, "sources": {}}, "merged_from", "method", "slerp"),
            (
                "perturbed_from.json",
                {"sigma": 0.02, "layer_scope": "attention"},
                "perturbed_from",
                "sigma",
                0.02,
            ),
        ],
    )
    def test_a_merged_or_perturbed_source_is_carried_into_the_init_record(
        self, tmp_path: Path, sidecar: str, payload: dict, key: str, field: str, expected: object
    ) -> None:
        xp = tmp_path / "abc123"
        xp.mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / sidecar).write_text(json.dumps(payload), encoding="utf-8")
        overrides = {"continue_from": str(source_dir / "checkpoint.th")}
        md = build_backfill_metadata(xp, overrides, datasource_train=None)
        ic = md["init_from_checkpoint"]
        assert ic["source_resume_metadata"] is not None
        assert ic["source_resume_metadata"][key][field] == expected


class TestBuildBackfillInitMetadata:
    def test_without_merge_it_warns(self) -> None:
        result = _build_backfill_init_metadata(continue_from="/tmp/ckpt.th", merged_from=None)
        assert result["origin"] == "local"
        assert result["source_resume_metadata"] is None
        assert len(result["warnings"]) == 1

    def test_merge_and_perturb_are_recorded_as_the_source(self) -> None:
        mf = {"method": "linear", "alpha": 0.5}
        merged = _build_backfill_init_metadata(continue_from="/tmp/ckpt.th", merged_from=mf)
        assert merged["source_resume_metadata"]["merged_from"] == mf

        pf = {"sigma": 0.02, "layer_scope": "lm-head"}
        perturbed = _build_backfill_init_metadata(
            continue_from="/tmp/ckpt.th",
            merged_from=None,
            perturbed_from=pf,
        )
        assert perturbed["source_resume_metadata"]["perturbed_from"] == pf


class TestBackfillXps:
    def test_empty_dora_dir(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        (dora / "xps").mkdir(parents=True)
        report = backfill_xps(dora, dry_run=True)
        assert report.total == 0

    def test_xp_with_full_hydra(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        xp = dora / "xps" / "abc123"
        hydra = xp / ".hydra"
        hydra.mkdir(parents=True)
        (hydra / "overrides.yaml").write_text("- optim.epochs=50\n- label=my_xp\n", encoding="utf-8")
        (hydra / "config.yaml").write_text("datasource:\n  train: /tmp/nope\n", encoding="utf-8")

        report = backfill_xps(dora, dry_run=True)
        assert report.total == 1
        assert report.written == 1
        r = report.results[0]
        assert r.xp_sig == "abc123"
        assert r.label == "my_xp"

    def test_xp_no_hydra(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        xp = dora / "xps" / "no_hydra"
        xp.mkdir(parents=True)

        report = backfill_xps(dora, dry_run=True)
        assert report.total == 1
        assert report.results[0].action == "skipped_no_hydra"

    def test_skips_existing_metadata(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        xp = dora / "xps" / "abc123"
        hydra = xp / ".hydra"
        hydra.mkdir(parents=True)
        (hydra / "overrides.yaml").write_text("- optim.epochs=50\n", encoding="utf-8")
        (hydra / "config.yaml").write_text("{}\n", encoding="utf-8")
        (xp / TRAIN_METADATA_FILENAME).write_text('{"existing": true}', encoding="utf-8")

        report = backfill_xps(dora, dry_run=True)
        assert report.results[0].action == "skipped_exists"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        xp = dora / "xps" / "abc123"
        hydra = xp / ".hydra"
        hydra.mkdir(parents=True)
        (hydra / "overrides.yaml").write_text("- optim.epochs=50\n- label=forced\n", encoding="utf-8")
        (hydra / "config.yaml").write_text("{}\n", encoding="utf-8")
        (xp / TRAIN_METADATA_FILENAME).write_text('{"existing": true}', encoding="utf-8")

        report = backfill_xps(dora, dry_run=False, force=True)
        assert report.results[0].action == "written"

        # Verify file was overwritten
        new_md = json.loads((xp / TRAIN_METADATA_FILENAME).read_text(encoding="utf-8"))
        assert new_md["effective_label"] == "forced"

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        xp = dora / "xps" / "abc123"
        hydra = xp / ".hydra"
        hydra.mkdir(parents=True)
        (hydra / "overrides.yaml").write_text("- optim.epochs=50\n", encoding="utf-8")
        (hydra / "config.yaml").write_text("{}\n", encoding="utf-8")

        report = backfill_xps(dora, dry_run=True)
        assert report.results[0].action == "written"
        assert not (xp / TRAIN_METADATA_FILENAME).exists()

    def test_commit_writes(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        xp = dora / "xps" / "abc123"
        hydra = xp / ".hydra"
        hydra.mkdir(parents=True)
        (hydra / "overrides.yaml").write_text("- optim.epochs=50\n- label=committed\n", encoding="utf-8")
        (hydra / "config.yaml").write_text("{}\n", encoding="utf-8")

        report = backfill_xps(dora, dry_run=False)
        assert report.results[0].action == "written"
        assert (xp / TRAIN_METADATA_FILENAME).exists()
        md = json.loads((xp / TRAIN_METADATA_FILENAME).read_text(encoding="utf-8"))
        assert md["xp_sig"] == "abc123"
        assert md["effective_label"] == "committed"
        assert md["_backfilled"] is True
        assert "written_at" in md


class TestBackfillReport:
    def test_properties(self) -> None:
        results = (
            BackfillResult(xp_sig="a", action="written"),
            BackfillResult(xp_sig="b", action="written"),
            BackfillResult(xp_sig="c", action="skipped_exists"),
            BackfillResult(xp_sig="d", action="error", error="boom"),
        )
        report = BackfillReport(results=results, dry_run=True)
        assert report.total == 4
        assert report.written == 2
        assert report.skipped == 1
        assert report.errors == 1
