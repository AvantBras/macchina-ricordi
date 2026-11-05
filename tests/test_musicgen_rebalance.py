from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.data.weights import (
    REBALANCE_SIDECAR_FILENAME,
    clear_manifest_weights,
    coverage_weights,
    delete_weights_overlay,
    fit_clusters,
    inverse_share_weights,
    list_weights_overlays,
    load_rebalance_plan,
    load_weights_overlay,
    plan_folder_mix,
    plan_rebalance,
    read_manifest_rows,
    read_rebalance_sidecar,
    rebalance_label_suffix,
    save_rebalance_plan,
    save_weights_overlay,
    weights_digest,
)


def _write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def _entry(path: str, duration: float = 10.0) -> dict:
    return {
        "path": path,
        "duration": duration,
        "sample_rate": 44_100,
        "amplitude": None,
        "weight": None,
        "info_path": None,
    }


# ── weighting math ─────────────────────────────────────────────────────


def test_inverse_share_upweights_small_clusters() -> None:
    labels = np.array([0] * 90 + [1] * 10)
    weights = inverse_share_weights(labels, 2, alpha=1.0, cap=10.0)
    assert weights[1] > weights[0]
    # mean weight over segments stays 1
    assert weights[labels].mean() == pytest.approx(1.0)


def test_inverse_share_respects_cap() -> None:
    labels = np.array([0] * 99 + [1] * 1)
    weights = inverse_share_weights(labels, 2, alpha=1.0, cap=4.0)
    # pre-normalisation ratio is clipped to cap^2 at most
    assert weights[1] / weights[0] <= 4.0 * 4.0 + 1e-9


def test_coverage_upweights_unproduced_cluster() -> None:
    labels_data = np.array([0] * 50 + [1] * 50)
    labels_gens = np.array([0] * 40)  # model never produces cluster 1
    weights = coverage_weights(labels_data, labels_gens, 2, cap=4.0)
    assert weights[1] > weights[0]
    assert weights[labels_data].mean() == pytest.approx(1.0)


def test_coverage_balanced_model_keeps_weights_flat() -> None:
    labels_data = np.array([0] * 50 + [1] * 50)
    labels_gens = np.array([0] * 20 + [1] * 20)
    weights = coverage_weights(labels_data, labels_gens, 2, cap=4.0)
    assert weights[0] == pytest.approx(weights[1], rel=1e-6)


# ── clustering ─────────────────────────────────────────────────────────


def test_fit_clusters_separates_obvious_groups() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, size=(40, 8))
    b = rng.normal(5.0, 0.1, size=(40, 8))
    model, labels = fit_clusters(np.vstack([a, b]), k=2, pca_dims=4, seed=0)
    assert model.k == 2
    assert len(set(labels[:40])) == 1
    assert len(set(labels[40:])) == 1
    assert labels[0] != labels[40]
    # predict() reproduces training labels
    assert (model.predict(np.vstack([a, b])) == labels).all()


# ── manifest I/O ───────────────────────────────────────────────────────


def test_clear_manifest_weights_resets_and_removes_sidecar(tmp_path: Path) -> None:
    manifest = tmp_path / "data.jsonl.gz"
    _write_manifest(manifest, [{**_entry("/a.mp3"), "weight": 2.0}])
    sidecar = tmp_path / REBALANCE_SIDECAR_FILENAME
    sidecar.write_text("{}", encoding="utf-8")

    clear_manifest_weights(manifest)
    assert read_manifest_rows(manifest)[0]["weight"] is None
    assert not sidecar.exists()


def test_weights_digest_is_order_independent_and_value_sensitive() -> None:
    d1 = weights_digest({"/a": 1.0, "/b": 2.0})
    d2 = weights_digest({"/b": 2.0, "/a": 1.0})
    d3 = weights_digest({"/a": 1.0, "/b": 2.5})
    assert d1 == d2
    assert d1 != d3
    assert len(d1) == 8


def test_rebalance_label_suffix() -> None:
    assert rebalance_label_suffix(None) == ""
    assert rebalance_label_suffix({}) == ""
    assert rebalance_label_suffix({"weights_digest": "deadbeefcafe"}) == "_rwdeadbeef"


def test_read_rebalance_sidecar_tolerates_garbage(tmp_path: Path) -> None:
    assert read_rebalance_sidecar(tmp_path) is None
    (tmp_path / REBALANCE_SIDECAR_FILENAME).write_text("not json", encoding="utf-8")
    assert read_rebalance_sidecar(tmp_path) is None


# ── end-to-end workspace rebalance (fake features, no audio decode) ────


def _fake_feature_fn(path: str) -> np.ndarray | None:
    # deterministic two-mode features: 'loud' files vs 'quiet' files
    base = 5.0 if "loud" in path else 0.0
    rng = np.random.default_rng(abs(hash(path)) % (2**32))
    return (base + rng.normal(0.0, 0.05, size=6)).astype(np.float32)


def _seed_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / ".workspace_t"
    train_dir = workspace / "egs" / "dset" / "train"
    entries = [_entry(f"/audio/loud_{i}.mp3") for i in range(9)]
    entries += [_entry(f"/audio/quiet_{i}.mp3") for i in range(3)]
    _write_manifest(train_dir / "data.jsonl.gz", entries)
    return workspace


def _plan(workspace: Path, **kwargs):  # type: ignore[no-untyped-def]
    return plan_rebalance(workspace_dir=workspace, dataset_name="dset", k=2, feature_fn=_fake_feature_fn, **kwargs)


def test_inverse_share_upweights_the_rare_cluster(tmp_path: Path) -> None:
    plan = _plan(_seed_workspace(tmp_path), cap=4.0)
    assert plan.mode == "inverse-share"
    quiet = [w for p, w in plan.weights_by_path.items() if "quiet" in p]
    loud = [w for p, w in plan.weights_by_path.items() if "loud" in p]
    assert min(quiet) > max(loud)


def test_coverage_mode_upweights_what_the_model_under_produces(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    gens_dir = tmp_path / "gens"
    gens_dir.mkdir()
    for i in range(4):  # the model only ever produces 'loud'-mode audio
        (gens_dir / f"loud_gen_{i}.wav").write_bytes(b"fake")
    plan = _plan(workspace, gens_dirs=[gens_dir], cap=4.0)
    assert plan.mode == "coverage"
    quiet = [w for p, w in plan.weights_by_path.items() if "quiet" in p]
    loud = [w for p, w in plan.weights_by_path.items() if "loud" in p]
    assert min(quiet) > max(loud)
    assert sorted(row.get("gen_share") for row in plan.cluster_table) == [0.0, 1.0]


def test_planning_requires_enough_segments(tmp_path: Path) -> None:
    train_dir = tmp_path / "egs" / "dset" / "train"
    _write_manifest(train_dir / "data.jsonl.gz", [_entry("/audio/loud_0.mp3")])
    with pytest.raises(RuntimeError, match="readable segments"):
        plan_rebalance(workspace_dir=tmp_path, dataset_name="dset", k=10, feature_fn=_fake_feature_fn)


def test_plan_examples_and_progress(tmp_path: Path) -> None:
    seen: list[tuple[str, int, int]] = []
    plan = _plan(
        _seed_workspace(tmp_path),
        examples_per_cluster=2,
        progress=lambda stage, done, total: seen.append((stage, done, total)),
    )
    assert seen[-1] == ("data", 12, 12)
    assert all(len(row["examples"]) == 2 for row in plan.cluster_table)
    quiet_cluster = next(row for row in plan.cluster_table if row["segments"] == 3)
    assert all("quiet" in path for path in quiet_cluster["examples"])


def test_overlay_is_saved_outside_the_workspace(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    manifest = workspace / "egs" / "dset" / "train" / "data.jsonl.gz"
    before = manifest.read_bytes()
    plan = load_rebalance_plan(save_rebalance_plan(_plan(workspace), tmp_path / "plan.json"))

    overlay = save_weights_overlay(plan, name="flat_a05", repo_root=tmp_path)

    assert manifest.read_bytes() == before
    assert not (manifest.parent / REBALANCE_SIDECAR_FILENAME).exists()
    assert overlay.folder == (tmp_path / "artifacts" / "weights" / ".workspace_t" / "flat_a05").resolve()
    weighted = {row["path"]: row["weight"] for row in read_manifest_rows(overlay.train_dir / "data.jsonl.gz")}
    assert weighted == pytest.approx(plan.weights_by_path)
    assert overlay.label_suffix == f"_rw{plan.weights_digest[:8]}"
    assert (overlay.workspace, overlay.dataset_name, overlay.is_stale()) == (".workspace_t", "dset", False)

    loaded = load_weights_overlay(tmp_path, ".workspace_t", "flat_a05")
    assert (loaded.weights_digest, loaded.mode, loaded.k) == (plan.weights_digest, plan.mode, 2)
    assert [o.name for o in list_weights_overlays(tmp_path, ".workspace_t")] == ["flat_a05"]


def test_overlay_names_are_unique_and_checked(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    plan = _plan(workspace)
    save_weights_overlay(plan, name="one", repo_root=tmp_path)
    with pytest.raises(FileExistsError, match="already exist"):
        save_weights_overlay(plan, name="one", repo_root=tmp_path)
    with pytest.raises(ValueError, match="invalid weights name"):
        save_weights_overlay(plan, name="Has Space", repo_root=tmp_path)
    with pytest.raises(FileNotFoundError, match=r"saved: one"):
        load_weights_overlay(tmp_path, ".workspace_t", "two")
    delete_weights_overlay(tmp_path, ".workspace_t", "one")
    assert list_weights_overlays(tmp_path, ".workspace_t") == []


def test_delete_refuses_a_name_that_points_elsewhere(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    save_weights_overlay(_plan(workspace), name="keep", repo_root=tmp_path)
    with pytest.raises(ValueError, match="invalid weights name"):
        delete_weights_overlay(tmp_path, ".workspace_t", "../.workspace_t/keep")
    assert [o.name for o in list_weights_overlays(tmp_path, ".workspace_t")] == ["keep"]


def test_overlay_goes_stale_when_the_workspace_manifest_changes(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    plan = _plan(workspace)
    overlay = save_weights_overlay(plan, name="one", repo_root=tmp_path)
    _write_manifest(workspace / "egs" / "dset" / "train" / "data.jsonl.gz", [_entry("/audio/other.mp3")])
    assert overlay.is_stale()
    with pytest.raises(RuntimeError, match="changed since"):
        save_weights_overlay(plan, name="two", repo_root=tmp_path)


def test_folder_mix_gives_each_folder_its_share_of_draws(tmp_path: Path) -> None:
    workspace = tmp_path / ".workspace_m"
    manifest = workspace / "egs" / "dset" / "train" / "data.jsonl.gz"
    _write_manifest(
        manifest,
        [_entry(f"/clips/a__{i}.flac") for i in range(6)]
        + [_entry(f"/clips/b__{i}.flac") for i in range(3)]
        + [_entry("/clips/c__0.flac")],
    )
    plan = plan_folder_mix(manifest, {"a": 1, "b": 1, "c": 2})

    draws = {folder: sum(w for p, w in plan.weights_by_path.items() if f"/{folder}__" in p) for folder in "abc"}
    total = sum(draws.values())
    assert {folder: draws[folder] / total for folder in "abc"} == pytest.approx({"a": 0.25, "b": 0.25, "c": 0.5})
    assert np.mean(list(plan.weights_by_path.values())) == pytest.approx(1.0)

    save_weights_overlay(plan, name="mix", repo_root=tmp_path)
    loaded = load_weights_overlay(tmp_path, ".workspace_m", "mix")
    assert (loaded.mode, loaded.k, loaded.mix) == ("folder-mix", None, {"a": 1, "b": 1, "c": 2})


def test_folder_mix_refuses_folders_it_cannot_weigh(tmp_path: Path) -> None:
    manifest = tmp_path / "egs" / "dset" / "train" / "data.jsonl.gz"
    _write_manifest(manifest, [_entry("/clips/a__0.flac"), _entry("/clips/b__0.flac")])
    with pytest.raises(ValueError, match="not in the mix: b"):
        plan_folder_mix(manifest, {"a": 1})
    with pytest.raises(ValueError, match="no train clips from: z"):
        plan_folder_mix(manifest, {"a": 1, "b": 1, "z": 1})
    _write_manifest(manifest, [_entry("/clips/loose.flac")])
    with pytest.raises(ValueError, match="no 'folder__' prefix"):
        plan_folder_mix(manifest, {"loose": 1})
