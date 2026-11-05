from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.models.merge import (
    merge_checkpoint_states,
    plan_checkpoint_merge,
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

    def clone(self) -> FakeTensor:
        return FakeTensor(np.array(self.data, copy=True), dtype=self.dtype, floating=self._floating)

    def to(self, *, dtype: str) -> FakeTensor:
        return FakeTensor(self.data, dtype=dtype, floating=self._floating)

    def reshape(self, *shape) -> FakeTensor:  # type: ignore[no-untyped-def]
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return FakeTensor(self.data.reshape(*shape), dtype=self.dtype, floating=self._floating)

    def item(self) -> float:
        return float(self.data.item())

    def __mul__(self, other):  # type: ignore[no-untyped-def]
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data * other_data, dtype=self.dtype, floating=self._floating)

    def __rmul__(self, other):  # type: ignore[no-untyped-def]
        return self.__mul__(other)

    def __add__(self, other):  # type: ignore[no-untyped-def]
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data + other_data, dtype=self.dtype, floating=self._floating)

    def __truediv__(self, other):  # type: ignore[no-untyped-def]
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data / other_data, dtype=self.dtype, floating=self._floating)


class FakeLinalg:
    @staticmethod
    def vector_norm(tensor: FakeTensor) -> FakeTensor:
        return FakeTensor(np.linalg.norm(tensor.data), dtype=tensor.dtype)


class FakeTorch:
    Tensor = FakeTensor
    float32 = "float32"
    float64 = "float64"
    linalg = FakeLinalg()

    def __init__(self, loads: dict[Path, dict[str, object]] | None = None):
        self.loads = loads or {}
        self.saved: dict[Path, object] = {}

    def load(self, path: Path, **kwargs):  # type: ignore[no-untyped-def]
        return self.loads[path]

    def save(self, state: object, path: Path) -> None:
        self.saved[path] = state
        path.write_bytes(b"fake-merged-checkpoint")

    @staticmethod
    def equal(left: FakeTensor, right: FakeTensor) -> bool:
        return bool(np.array_equal(left.data, right.data))

    @staticmethod
    def sum(tensor: FakeTensor) -> FakeTensor:
        return FakeTensor(np.sum(tensor.data), dtype=tensor.dtype)

    @staticmethod
    def clamp(tensor: FakeTensor, minimum: float, maximum: float) -> FakeTensor:
        return FakeTensor(np.clip(tensor.data, minimum, maximum), dtype=tensor.dtype)

    @staticmethod
    def acos(tensor: FakeTensor) -> FakeTensor:
        return FakeTensor(np.arccos(tensor.data), dtype=tensor.dtype)

    @staticmethod
    def sin(tensor: FakeTensor) -> FakeTensor:
        return FakeTensor(np.sin(tensor.data), dtype=tensor.dtype)


def test_plan_checkpoint_merge_builds_default_dry_run_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint_a = tmp_path / "a" / "checkpoint.th"
    checkpoint_b = tmp_path / "b" / "checkpoint.th"
    output = tmp_path / "merged" / "checkpoint.th"
    helper = tmp_path / "run_musicgen_merge.py"
    python = tmp_path / "python"
    for path in (checkpoint_a, checkpoint_b, helper, python):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    monkeypatch.setattr("macchina_ricordi.models.merge.require_musicgen_python", lambda explicit: str(python))
    monkeypatch.setattr("macchina_ricordi.models.merge._merge_helper_script", lambda: helper)

    plan = plan_checkpoint_merge(
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        output_checkpoint=output,
        dry_run=True,
    )

    assert plan.method == "linear"
    assert plan.alpha == 0.5
    assert plan.sidecar_path == output.parent / "merged_from.json"
    assert plan.command[:4] == [str(python), str(helper), str(checkpoint_a.resolve()), str(checkpoint_b.resolve())]
    assert str(output.resolve()) in plan.command


def test_merge_checkpoint_states_linear_alpha_values() -> None:
    torch = FakeTorch()
    state_a = {"model": {"weight": _t([0.0, 4.0])}}
    state_b = {"model": {"weight": _t([8.0, 12.0])}}

    merged = merge_checkpoint_states(state_a, state_b, method="linear", alpha=0.25, torch_module=torch)
    np.testing.assert_allclose(merged.state["model"]["weight"].data, np.array([2.0, 6.0]))

    merged_a = merge_checkpoint_states(state_a, state_b, method="linear", alpha=0.0, torch_module=torch)
    merged_b = merge_checkpoint_states(state_a, state_b, method="linear", alpha=1.0, torch_module=torch)
    np.testing.assert_allclose(merged_a.state["model"]["weight"].data, state_a["model"]["weight"].data)
    np.testing.assert_allclose(merged_b.state["model"]["weight"].data, state_b["model"]["weight"].data)


def test_weight_movement_groups_layers_in_depth_order() -> None:
    torch = pytest.importorskip("torch")
    from macchina_ricordi.models.merge import weight_movement

    before = {
        "model": {
            "emb.0.weight": torch.ones(4),
            "transformer.layers.10.w": torch.ones(4),
            "transformer.layers.2.w": torch.ones(4),
            "linears.0.weight": torch.ones(4),
            "steps": torch.tensor([3]),
        }
    }
    after = {"model": {**before["model"], "transformer.layers.2.w": torch.full((4,), 2.0)}}

    movement = weight_movement(before, after, torch)

    assert [g["name"] for g in movement["groups"]] == ["emb", "layer 2", "layer 10", "linears"]
    layer_2 = movement["groups"][1]
    assert (layer_2["rel"], layer_2["cos"], layer_2["rms"]) == (1.0, 1.0, 1.0)
    assert movement["groups"][0]["rel"] == 0.0
    assert movement["total"]["params"] == 16 and movement["total"]["rel"] == 0.5
