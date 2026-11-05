from __future__ import annotations

import json
from pathlib import Path

from macchina_ricordi.models.catalog import (
    Catalog,
    gather_models,
)


class TestGatherModels:
    def test_single_xp(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        xp = dora / "xps" / "abc123"
        xp.mkdir(parents=True)
        (xp / "checkpoint.th").write_bytes(b"x" * 100)
        (xp / "macchina_ricordi_train_metadata.json").write_text(
            json.dumps(
                {
                    "xp_sig": "abc123",
                    "xp_folder": str(xp),
                    "effective_label": "test_run",
                    "workspace_dir": str(tmp_path / ".my_workspace"),
                    "dataset_fingerprint": {
                        "train_count": 10,
                        "valid_count": 2,
                        "train_aggregate_sha256": "abc123",
                        "profile": "micro-unconditional",
                    },
                    "dataset_fingerprint_digest": "abc123",
                    "command": ["dora", "run", "optim.epochs=50"],
                    "written_at": "2026-01-01T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )

        models = gather_models(dora)
        assert len(models) == 1
        m = models[0]
        assert m.sig == "abc123"
        assert m.label == "test_run"
        assert m.workspace == ".my_workspace"
        assert m.train_count == 10
        assert m.epochs == 50

    def test_with_init_from_checkpoint(self, tmp_path: Path) -> None:
        dora = tmp_path / ".dora"
        xp = dora / "xps" / "abc123"
        xp.mkdir(parents=True)
        (xp / "macchina_ricordi_train_metadata.json").write_text(
            json.dumps(
                {
                    "xp_sig": "abc123",
                    "xp_folder": str(xp),
                    "effective_label": "warm_start",
                    "workspace_dir": str(tmp_path / ".ws"),
                    "command": ["dora", "run"],
                    "init_from_checkpoint": {
                        "user_input": "/tmp/merged/checkpoint.th",
                        "resolved_local_path": "/tmp/merged/checkpoint.th",
                        "source_resume_metadata": {
                            "merged_from": {
                                "method": "slerp",
                                "alpha": 0.7,
                                "sources": {
                                    "a": {"path": "/a/checkpoint_150.th"},
                                    "b": {"path": "/b/checkpoint_50.th"},
                                },
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        models = gather_models(dora)
        assert len(models) == 1
        m = models[0]
        assert m.continue_from == "/tmp/merged/checkpoint.th"
        assert m.merged_from is not None
        assert m.merged_from.method == "slerp"
        assert m.merged_from.alpha == 0.7
        assert m.merged_from.source_a_sig == "a-150"
        assert m.merged_from.source_b_sig == "b-50"


class TestBuildCatalog:
    def test_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".dora" / "xps").mkdir(parents=True)
        (tmp_path / "merged_checkpoints").mkdir()
        (tmp_path / "artifacts" / "generated_audio").mkdir(parents=True)

        # Monkey-patch get_repo_root to return tmp_path
        import macchina_ricordi.models.catalog as cat_mod

        cat = cat_mod.build_catalog(repo_root=tmp_path)
        assert isinstance(cat, Catalog)
        assert len(cat.models) == 0
        assert len(cat.merges) == 0
        assert len(cat.perturbations) == 0
        assert len(cat.generations) == 0
        assert cat.generated_at
