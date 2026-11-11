"""A warm-up that spans the whole run must not crash the vendored AudioCraft cosine schedule.

The decay divides by ``total_steps - warmup_steps`` once warm-up ends. With the MusicGen solver's default warm-up of
4000 updates, a 2-epoch run of 2000 updates per epoch has warm-up equal to the whole run, and its last update raised
ZeroDivisionError. The checkout's copy (``patches/audiocraft-cosine-warmup-bounds.patch``) must match upstream exactly
wherever upstream works.

Needs torch and the AudioCraft checkout from ``ricordi runtime setup``; skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = REPO_ROOT / ".audiocraft"
SCHEDULER = "audiocraft/optim/cosine_lr_scheduler.py"

pytestmark = [
    pytest.mark.skipif(not (CHECKOUT / SCHEDULER).exists(), reason="AudioCraft checkout is not set up"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


def _upstream_source() -> str:
    # HEAD of the checkout is the pinned upstream commit; the working-tree copy is the patched one.
    completed = subprocess.run(
        ["git", "-C", str(CHECKOUT), "show", f"HEAD:{SCHEDULER}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("AudioCraft checkout has no git history to read upstream from")
    return completed.stdout


def _patched_source() -> str:
    return (CHECKOUT / SCHEDULER).read_text(encoding="utf-8")


def _load(tmp_path: Path, name: str, source: str) -> ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lr_trace(scheduler_cls, *, total: int, warmup: int):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = scheduler_cls(optimizer, total_steps=total, warmup_steps=warmup, lr_min_ratio=0.05, cycle_length=1.3)
    trace = [optimizer.param_groups[0]["lr"]]
    for _ in range(total):
        optimizer.step()
        scheduler.step()
        trace.append(optimizer.param_groups[0]["lr"])
    return trace, scheduler


def test_valid_schedule_matches_upstream_exactly(tmp_path: Path) -> None:
    upstream = _load(tmp_path, "upstream", _upstream_source())
    patched = _load(tmp_path, "patched", _patched_source())
    expected, _ = _lr_trace(upstream.CosineLRScheduler, total=6000, warmup=4000)
    actual, scheduler = _lr_trace(patched.CosineLRScheduler, total=6000, warmup=4000)
    assert actual == expected
    assert scheduler.effective_warmup_steps == 4000


def test_warmup_equal_to_run_no_longer_divides_by_zero(tmp_path: Path) -> None:
    upstream = _load(tmp_path, "upstream", _upstream_source())
    patched = _load(tmp_path, "patched", _patched_source())
    with pytest.raises(ZeroDivisionError):
        _lr_trace(upstream.CosineLRScheduler, total=4000, warmup=4000)
    trace, scheduler = _lr_trace(patched.CosineLRScheduler, total=4000, warmup=4000)
    assert scheduler.effective_warmup_steps == 400
    assert trace[400] == pytest.approx(1.0)  # the peak lands where the shortened warm-up ends
    assert trace[399] < trace[400]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(trace[400:], trace[401:], strict=False))
    assert trace[-1] < 0.5


def test_warmup_longer_than_run_is_cut_and_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    patched = _load(tmp_path, "patched", _patched_source())
    with caplog.at_level(logging.WARNING):
        trace, scheduler = _lr_trace(patched.CosineLRScheduler, total=3000, warmup=4000)
    assert scheduler.effective_warmup_steps == 300
    assert trace[300] == pytest.approx(1.0)
    assert trace[-1] < 0.5  # it decays now, instead of stopping at a warm-up that never peaks
    assert "Using 300 warm-up updates" in caplog.text
