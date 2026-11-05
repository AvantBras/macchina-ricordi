from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.models.perturb import (
    perturb_checkpoint_files,
    plan_checkpoint_perturb,
)


def _t(values, *, dtype: str = "float32", floating: bool = True) -> FakeTensor:  # type: ignore[no-untyped-def]
    return FakeTensor(values, dtype=dtype, floating=floating)


class FakeTensor:
    def __init__(self, values, *, dtype: str = "float32", floating: bool = True):  # type: ignore[no-untyped-def]
        self.data = np.asarray(values, dtype=np.float64 if floating else np.int64)
        self.dtype = dtype
        self._floating = floating

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def is_floating_point(self) -> bool:
        return self._floating

    def numel(self) -> int:
        return int(self.data.size)

    def detach(self) -> FakeTensor:
        return self

    def double(self) -> FakeTensor:
        return self

    def sum(self) -> float:
        return float(self.data.sum())

    def clone(self) -> FakeTensor:
        return FakeTensor(np.array(self.data, copy=True), dtype=self.dtype, floating=self._floating)

    def __mul__(self, other):  # type: ignore[no-untyped-def]
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data * other_data, dtype=self.dtype, floating=self._floating)

    def __rmul__(self, other):  # type: ignore[no-untyped-def]
        return self.__mul__(other)

    def __add__(self, other):  # type: ignore[no-untyped-def]
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data + other_data, dtype=self.dtype, floating=self._floating)


class FakeGenerator:
    def __init__(self, device: str | None = None):
        self.device = device
        self.rng = np.random.default_rng(0)

    def manual_seed(self, seed: int) -> FakeGenerator:
        self.rng = np.random.default_rng(seed)
        return self


class FakeTorch:
    Tensor = FakeTensor
    Generator = FakeGenerator

    def __init__(self, loads: dict[Path, dict[str, object]] | None = None):
        self.loads = loads or {}
        self.saved: dict[Path, object] = {}

    def load(self, path: Path, **kwargs):  # type: ignore[no-untyped-def]
        return self.loads[path]

    def save(self, state: object, path: Path) -> None:
        self.saved[path] = state
        path.write_bytes(b"fake-perturbed-checkpoint")

    @staticmethod
    def randn_like(tensor: FakeTensor, *, generator: FakeGenerator | None = None) -> FakeTensor:
        rng = generator.rng if generator is not None else np.random.default_rng(0)
        return FakeTensor(rng.standard_normal(tensor.shape), dtype=tensor.dtype)


def test_plan_checkpoint_perturb_builds_default_dry_run_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "checkpoint.th"
    output = tmp_path / "perturbed" / "checkpoint.th"
    helper = tmp_path / "run_musicgen_perturb.py"
    python = tmp_path / "python"
    for path in (source, helper, python):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    monkeypatch.setattr("macchina_ricordi.models.perturb.require_musicgen_python", lambda explicit: str(python))
    monkeypatch.setattr("macchina_ricordi.models.perturb._perturb_helper_script", lambda: helper)

    plan = plan_checkpoint_perturb(
        source_checkpoint=source,
        output_checkpoint=output,
        sigma=0.02,
        dry_run=True,
    )

    assert plan.sigma == 0.02
    assert plan.layer_scope == "all"
    assert plan.seed == 0
    assert plan.sidecar_path == output.parent / "perturbed_from.json"
    assert plan.command[:4] == [str(python), str(helper), str(source.resolve()), str(output.resolve())]
    assert "--sigma" in plan.command
    assert "--layer-scope" in plan.command


def test_perturb_checkpoint_files_writes_sidecar_with_source_lineage(tmp_path: Path) -> None:
    source = tmp_path / "source" / "checkpoint.th"
    output = tmp_path / "perturbed" / "checkpoint.th"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    (source.parent / "merged_from.json").write_text(json.dumps({"method": "linear", "alpha": 0.5}), encoding="utf-8")
    torch = FakeTorch({source.resolve(): {"model": {"weight": _t([1.0])}}})

    sidecar = perturb_checkpoint_files(
        source_checkpoint=source.resolve(),
        output_checkpoint=output,
        sigma=0.01,
        layer_scope="all",
        seed=42,
        force=False,
        torch_module=torch,
    )

    assert output.exists()
    assert (output.parent / "perturbed_from.json").exists()
    persisted = json.loads((output.parent / "perturbed_from.json").read_text(encoding="utf-8"))
    assert persisted == sidecar
    assert sidecar["sigma"] == 0.01
    assert sidecar["layer_scope"] == "all"
    assert sidecar["seed"] == 42
    assert sidecar["sources"]["source"]["sha256"]
    assert sidecar["source_resume_metadata"]["merged_from"]["method"] == "linear"
    assert sidecar["perturbed_roots"][0]["path"] == "model"
    assert sidecar["perturbed_roots"][0]["perturbed_tensor_count"] == 1
    assert sidecar["movement"]["between"] == ["source", "output"]
    assert sidecar["movement"]["groups"][0]["name"] == "weight" and sidecar["movement"]["total"]["rel"] > 0
