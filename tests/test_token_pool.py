from __future__ import annotations

import dataclasses
import json

from macchina_ricordi.training.pool import (
    PoolSpec,
    cache_override,
    max_repeats,
    pool_of_config,
    pool_problems,
    pool_size,
    resume_problem,
)


def _spec(**changes):
    spec = PoolSpec(
        dataset=".workspace_x",
        fingerprint="13b1a67bd03e",
        train_dir="/data/x/train",
        segment_duration=20.0,
        codec="//pretrained/facebook/encodec_32khz",
        batch_size=4,
        updates_per_epoch=2000,
        epochs=50,
        crops=50_000,
    )
    return dataclasses.replace(spec, **changes)


def _status(spec: PoolSpec) -> dict:
    return {
        "dataset_fingerprint": spec.fingerprint,
        "dataset": spec.train_dir,
        "segment_duration": spec.segment_duration,
        "codec": spec.codec,
        "batch_size": spec.batch_size,
        "updates_per_epoch": spec.updates_per_epoch,
        "epochs": spec.epochs,
        "crops": spec.crops,
        "repeats_per_crop": spec.repeats,
    }


def test_pool_size_caps_repeats_or_covers_every_second():
    assert pool_size(9_209, epochs=50, batch_size=4, updates_per_epoch=2000, repeats=8) == 50_000
    assert pool_size(31_016, epochs=20, batch_size=4, updates_per_epoch=2000, repeats=8) == 32_000  # the floor


def test_small_families_repeat_eight_times_and_larger_families_four():
    assert max_repeats("nano-family") == 8
    assert max_repeats("pico-family") == 8
    assert max_repeats("deep-stack") == 4
    assert max_repeats("base") == 4
    assert max_repeats(None) == 4


def test_spec_name_and_repeats():
    spec = _spec()
    assert spec.name == "workspace_x-13b1a67b-20s-b4-50k-50ep"
    assert _spec(seed=2).name.endswith("-seed2")
    assert spec.repeats == 8


def test_pool_problems_name_each_mismatch():
    spec = _spec()
    assert pool_problems(_status(spec), spec) == []  # crop seed 0 is implied by pools built before seeds existed
    assert pool_problems(_status(spec), _spec(epochs=40)) == []  # fewer epochs than built is fine
    assert pool_problems(None, spec)
    problems = pool_problems(_status(spec), _spec(epochs=60, segment_duration=10.0, seed=1))
    assert len(problems) == 3


def test_resume_past_the_pool_is_refused(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "pool_status.json").write_text(json.dumps({"epochs": 50}), encoding="utf-8")
    xp = tmp_path / "xp"
    (xp / ".hydra").mkdir(parents=True)
    (xp / ".hydra" / "config.yaml").write_text(f"cache:\n  path: {pool / 'cache'}\n  write: false\n", encoding="utf-8")
    assert resume_problem(xp, 50) is None
    assert "50 epochs" in resume_problem(xp, 60)
    fresh = tmp_path / "fresh"
    (fresh / ".hydra").mkdir(parents=True)
    (fresh / ".hydra" / "config.yaml").write_text("cache:\n  path: null\n", encoding="utf-8")
    assert resume_problem(fresh, 500) is None


def test_cache_path_points_at_the_pool_or_its_older_zip_cache(tmp_path):
    native, zips = tmp_path / "native", tmp_path / "zips"
    for pool in (native, zips):
        pool.mkdir()
        (pool / "pool_status.json").write_text("{}", encoding="utf-8")
    (zips / "cache").mkdir()
    assert cache_override(native) == f"cache.path={native}"
    assert cache_override(zips) == f"cache.path={zips / 'cache'}"
    assert pool_of_config({"cache": {"path": str(native)}}) == native
    assert pool_of_config({"cache": {"path": str(zips / "cache")}}) == zips
    assert pool_of_config({"cache": {"path": str(native), "write": True}}) is None


def test_a_longer_pool_with_the_same_seed_draws_the_same_crops():
    from macchina_ricordi.training.pool import _draws_same_crops

    spec = _spec()
    assert _draws_same_crops(_status(spec), _spec(epochs=100, crops=120_000))
    assert not _draws_same_crops(_status(spec), _spec(seed=1))
