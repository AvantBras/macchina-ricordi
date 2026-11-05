from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from macchina_ricordi.gui.training import (
    LaunchRequest,
    TrainingManager,
    TrainingRequestError,
)

# Stands in for `python -m macchina_ricordi.cli train run --job <id>`: it starts a child (Dora) and waits
# for it, then exits on its own with a code that says the child was stopped. The trailing "macchina_ricordi.cli"
# argument puts that string in the process's command line, which is how a job's process is recognised.
FAKE_CLI = (
    sys.executable,
    "-c",
    "import subprocess, sys; "
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
    "print('dora started', flush=True); "
    "sys.exit(0 if child.wait() == 0 else 3)",
    "macchina_ricordi.cli",
)


def _manifest(path: Path, durations: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index, duration in enumerate(durations):
            handle.write(json.dumps({"path": f"clip{index}.wav", "duration": duration, "sample_rate": 32000}) + "\n")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    for split in ("train", "valid"):
        _manifest(tmp_path / ".workspace_t" / "egs" / "ds" / split / "data.jsonl.gz", [10.0, 10.0, 9.5])
    (tmp_path / ".workspace_empty").mkdir()
    (tmp_path / ".dora" / "xps").mkdir(parents=True)
    return tmp_path


def _planner(repo: Path, sig: str = "abcd1234"):
    calls = []

    def plan(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            xp_sig=sig,
            xp_folder=repo / ".dora" / "xps" / sig,
            effective_label="label_x",
            command=["dora", "run", "optim.epochs=3"],
            train_metadata={"segment_duration": 10.0, "clip_slice_length_s": 10.0},
            segment_warnings=(),
            init_from_checkpoint=None,
            dataset_fingerprint_digest="d1",
        )

    plan.calls = calls
    return plan


def _manager(repo: Path, **kwargs) -> TrainingManager:
    return TrainingManager(repo, planner=_planner(repo), cli_prefix=FAKE_CLI, stop_grace_seconds=5, **kwargs)


def _request(**overrides) -> LaunchRequest:
    return LaunchRequest.from_payload({"profile": "reshape-nano", "workspace": ".workspace_t", **overrides})


class TestLaunchAndStop:
    def test_launch_requires_the_matching_preview(self, repo: Path) -> None:
        manager = _manager(repo)
        with pytest.raises(TrainingRequestError, match="preview again"):
            manager.launch(_request(), preview_key="nope")
        preview = manager.preview(_request())
        with pytest.raises(TrainingRequestError, match="preview again"):
            manager.launch(_request(run_label="changed"), preview_key=preview["key"])


def test_pool_preview_plans_with_the_pool_cache(repo: Path, monkeypatch) -> None:
    from macchina_ricordi.training import pool as token_pool

    pool_dir = repo / "artifacts" / "token_pools" / "p"
    spec = SimpleNamespace(crops=1000, repeats=8.0, epochs=3, pool_bytes=lambda: 10**6)
    monkeypatch.setattr(token_pool, "plan_pool", lambda *a, **k: token_pool.PoolPlan(spec, pool_dir, False, 0.5, None))
    manager = _manager(repo)
    request = _request(pool=True, pool_seed="2")
    assert request.cli_args()[-4:-1] == ["--pool", "--pool-seed", "2"]
    preview = manager.preview(request)
    assert manager.planner.calls[-1]["overrides"][-1] == f"cache.path={pool_dir}"
    assert preview["pool"]["crops"] == 1000 and not preview["pool"]["built"]
    with pytest.raises(TrainingRequestError, match="cache.path itself"):
        _request(pool=True, overrides=["cache.path=/x"])
