from __future__ import annotations

import math
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..musicgen.config import load_musicgen_project_config, musicgen_script_env
from ..musicgen.runtime import require_musicgen_python
from ..musicgen.shared import sha256_file, utc_now_iso, write_json

MERGE_SIDECAR_SCHEMA_VERSION = 1
MERGE_METHODS = ("linear", "slerp")
MERGE_WEIGHT_ROOTS: tuple[tuple[str, ...], ...] = (
    ("model",),
    ("best_state", "model"),
    ("fsdp_best_state", "model"),
)


@dataclass(frozen=True)
class CheckpointMergePlan:
    checkpoint_a: Path
    checkpoint_b: Path
    output_checkpoint: Path
    sidecar_path: Path
    method: str
    alpha: float
    musicgen_python: str
    command: list[str]
    dry_run: bool
    force: bool


@dataclass(frozen=True)
class CheckpointMergeResult:
    output_checkpoint: Path
    sidecar_path: Path
    dry_run: bool


@dataclass(frozen=True)
class MergedRootStats:
    path: str
    tensor_count: int
    floating_tensor_count: int
    copied_non_floating_tensor_count: int
    parameter_count: int


@dataclass(frozen=True)
class MergeStateResult:
    state: dict[str, Any]
    roots: tuple[MergedRootStats, ...]
    warnings: tuple[str, ...]


def plan_checkpoint_merge(
    *,
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    output_checkpoint: str | Path,
    method: str = "linear",
    alpha: float = 0.5,
    musicgen_python: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> CheckpointMergePlan:
    if method not in MERGE_METHODS:
        raise ValueError(f"Unsupported merge method: {method!r}. Expected one of: {', '.join(MERGE_METHODS)}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1 inclusive.")

    resolved_a = Path(checkpoint_a).expanduser().resolve()
    resolved_b = Path(checkpoint_b).expanduser().resolve()
    resolved_output = Path(output_checkpoint).expanduser().resolve()
    for label, path in (("checkpoint A", resolved_a), ("checkpoint B", resolved_b)):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
    if resolved_a == resolved_b:
        raise ValueError("checkpoint A and checkpoint B must be different files.")
    if resolved_output in (resolved_a, resolved_b):
        raise ValueError("Output checkpoint must not overwrite either source checkpoint.")
    sidecar_path = resolved_output.parent / "merged_from.json"
    if resolved_output.exists() and resolved_output.is_dir():
        raise IsADirectoryError(f"Output checkpoint path is a directory: {resolved_output}")
    if resolved_output.exists() and not force:
        raise FileExistsError(f"Output checkpoint already exists: {resolved_output}. Pass --force to overwrite.")
    if sidecar_path.exists() and not force:
        raise FileExistsError(f"Merge sidecar already exists: {sidecar_path}. Pass --force to overwrite.")

    resolved_python = require_musicgen_python(musicgen_python)
    helper_script = _merge_helper_script()
    if not helper_script.exists():
        raise FileNotFoundError(f"MusicGen merge helper script not found: {helper_script}")

    command = [
        resolved_python,
        str(helper_script),
        str(resolved_a),
        str(resolved_b),
        str(resolved_output),
        "--alpha",
        str(alpha),
        "--method",
        method,
    ]
    if force:
        command.append("--force")

    return CheckpointMergePlan(
        checkpoint_a=resolved_a,
        checkpoint_b=resolved_b,
        output_checkpoint=resolved_output,
        sidecar_path=sidecar_path,
        method=method,
        alpha=alpha,
        musicgen_python=resolved_python,
        command=command,
        dry_run=dry_run,
        force=force,
    )


def execute_checkpoint_merge(plan: CheckpointMergePlan) -> CheckpointMergeResult:
    if plan.dry_run:
        return CheckpointMergeResult(
            output_checkpoint=plan.output_checkpoint,
            sidecar_path=plan.sidecar_path,
            dry_run=True,
        )

    env = musicgen_script_env()
    try:
        subprocess.run(
            plan.command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        detail = stderr or stdout
        message = f"MusicGen checkpoint merge failed with exit code {exc.returncode}."
        if detail:
            message = f"{message} {detail}"
        raise RuntimeError(message) from exc

    return CheckpointMergeResult(
        output_checkpoint=plan.output_checkpoint,
        sidecar_path=plan.sidecar_path,
        dry_run=False,
    )


def merge_checkpoint_files(
    *,
    checkpoint_a: Path,
    checkpoint_b: Path,
    output_checkpoint: Path,
    method: str,
    alpha: float,
    force: bool,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    sidecar_path = output_checkpoint.parent / "merged_from.json"
    if output_checkpoint.exists() and output_checkpoint.is_dir():
        raise IsADirectoryError(f"Output checkpoint path is a directory: {output_checkpoint}")
    if output_checkpoint.exists() and not force:
        raise FileExistsError(f"Output checkpoint already exists: {output_checkpoint}. Pass --force to overwrite.")
    if sidecar_path.exists() and not force:
        raise FileExistsError(f"Merge sidecar already exists: {sidecar_path}. Pass --force to overwrite.")
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    torch = torch_module or _import_torch()
    state_a = _trusted_torch_load(checkpoint_a, torch)
    state_b = _trusted_torch_load(checkpoint_b, torch)
    if not isinstance(state_a, dict) or not isinstance(state_b, dict):
        raise TypeError("Both checkpoints must load to top-level dictionaries.")

    merged = merge_checkpoint_states(
        state_a,
        state_b,
        method=method,
        alpha=alpha,
        torch_module=torch,
    )
    torch.save(merged.state, output_checkpoint)

    output_snapshot = _file_snapshot(output_checkpoint)
    sidecar_payload = {
        "schema_version": MERGE_SIDECAR_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "operation": "musicgen_checkpoint_merge",
        "method": method,
        "alpha": alpha,
        "sources": {
            "a": _file_snapshot(checkpoint_a),
            "b": _file_snapshot(checkpoint_b),
        },
        "output": output_snapshot,
        "merged_roots": [
            {
                "path": root.path,
                "tensor_count": root.tensor_count,
                "floating_tensor_count": root.floating_tensor_count,
                "copied_non_floating_tensor_count": root.copied_non_floating_tensor_count,
                "parameter_count": root.parameter_count,
            }
            for root in merged.roots
        ],
        "warnings": list(merged.warnings),
        # how far apart the parents are: the output sits alpha of the way from A to B
        "movement": {"between": ["a", "b"], **(weight_movement(state_a, state_b, torch) or {})},
    }
    write_json(sidecar_path, sidecar_payload)
    return sidecar_payload


def merge_checkpoint_states(
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    *,
    method: str,
    alpha: float,
    torch_module: Any | None = None,
) -> MergeStateResult:
    if method not in MERGE_METHODS:
        raise ValueError(f"Unsupported merge method: {method!r}. Expected one of: {', '.join(MERGE_METHODS)}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1 inclusive.")

    torch = torch_module or _import_torch()
    output: dict[str, Any] = {}
    roots: list[MergedRootStats] = []
    warnings: list[str] = []

    for root_path in MERGE_WEIGHT_ROOTS:
        has_a = _has_nested(state_a, root_path)
        has_b = _has_nested(state_b, root_path)
        root_label = ".".join(root_path)
        if has_a and has_b:
            source_a = _get_nested(state_a, root_path)
            source_b = _get_nested(state_b, root_path)
            if not isinstance(source_a, dict) or not isinstance(source_b, dict):
                raise TypeError(f"Merge root {root_label} must be a dictionary in both checkpoints.")
            merged_root, stats = _merge_tensor_mapping(
                source_a,
                source_b,
                root_path=root_label,
                method=method,
                alpha=alpha,
                torch_module=torch,
            )
            _set_nested(output, root_path, merged_root)
            roots.append(stats)
        elif has_a != has_b:
            warnings.append(f"Skipped {root_label}: present in only one checkpoint.")

    if not roots:
        raise ValueError(
            "No compatible weight roots found. Expected at least one of: "
            + ", ".join(".".join(path) for path in MERGE_WEIGHT_ROOTS)
        )
    return MergeStateResult(state=output, roots=tuple(roots), warnings=tuple(warnings))


def merge_tensor_values(
    tensor_a: Any,
    tensor_b: Any,
    *,
    method: str,
    alpha: float,
    torch_module: Any | None = None,
) -> Any:
    torch = torch_module or _import_torch()
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Tensor shape mismatch: {tuple(tensor_a.shape)} != {tuple(tensor_b.shape)}")
    if tensor_a.dtype != tensor_b.dtype:
        raise ValueError(f"Tensor dtype mismatch: {tensor_a.dtype} != {tensor_b.dtype}")
    if not tensor_a.is_floating_point() or not tensor_b.is_floating_point():
        if not torch.equal(tensor_a, tensor_b):
            raise ValueError("Non-floating tensors must match exactly.")
        return tensor_a.detach().clone() if hasattr(tensor_a, "detach") else tensor_a

    if method == "linear":
        return _linear_interpolate(tensor_a, tensor_b, alpha)
    if method == "slerp":
        return _slerp(tensor_a, tensor_b, alpha, torch)
    raise ValueError(f"Unsupported merge method: {method!r}.")


def _merge_tensor_mapping(
    mapping_a: dict[str, Any],
    mapping_b: dict[str, Any],
    *,
    root_path: str,
    method: str,
    alpha: float,
    torch_module: Any,
) -> tuple[dict[str, Any], MergedRootStats]:
    keys_a = set(mapping_a)
    keys_b = set(mapping_b)
    missing_from_b = sorted(keys_a - keys_b)
    missing_from_a = sorted(keys_b - keys_a)
    if missing_from_a or missing_from_b:
        details: list[str] = []
        if missing_from_a:
            details.append(f"missing from A: {', '.join(missing_from_a[:5])}")
        if missing_from_b:
            details.append(f"missing from B: {', '.join(missing_from_b[:5])}")
        raise ValueError(f"Weight key mismatch under {root_path}: {'; '.join(details)}")

    merged: dict[str, Any] = {}
    tensor_count = 0
    floating_tensor_count = 0
    copied_non_floating_tensor_count = 0
    parameter_count = 0
    tensor_type = torch_module.Tensor
    for key in sorted(mapping_a):
        value_a = mapping_a[key]
        value_b = mapping_b[key]
        if isinstance(value_a, dict) and isinstance(value_b, dict):
            child, child_stats = _merge_tensor_mapping(
                value_a,
                value_b,
                root_path=f"{root_path}.{key}",
                method=method,
                alpha=alpha,
                torch_module=torch_module,
            )
            merged[key] = child
            tensor_count += child_stats.tensor_count
            floating_tensor_count += child_stats.floating_tensor_count
            copied_non_floating_tensor_count += child_stats.copied_non_floating_tensor_count
            parameter_count += child_stats.parameter_count
            continue
        if not isinstance(value_a, tensor_type) or not isinstance(value_b, tensor_type):
            if value_a == value_b:
                merged[key] = value_a
                continue
            raise TypeError(f"Unsupported non-tensor mismatch under {root_path}.{key}.")

        merged_value = merge_tensor_values(
            value_a,
            value_b,
            method=method,
            alpha=alpha,
            torch_module=torch_module,
        )
        merged[key] = merged_value
        tensor_count += 1
        if value_a.is_floating_point():
            floating_tensor_count += 1
            parameter_count += int(value_a.numel())
        else:
            copied_non_floating_tensor_count += 1

    return merged, MergedRootStats(
        path=root_path,
        tensor_count=tensor_count,
        floating_tensor_count=floating_tensor_count,
        copied_non_floating_tensor_count=copied_non_floating_tensor_count,
        parameter_count=parameter_count,
    )


def _linear_interpolate(tensor_a: Any, tensor_b: Any, alpha: float) -> Any:
    if alpha == 0.0:
        return tensor_a.detach().clone() if hasattr(tensor_a, "detach") else tensor_a
    if alpha == 1.0:
        return tensor_b.detach().clone() if hasattr(tensor_b, "detach") else tensor_b
    return tensor_a * (1.0 - alpha) + tensor_b * alpha


def _slerp(tensor_a: Any, tensor_b: Any, alpha: float, torch_module: Any) -> Any:
    if alpha in (0.0, 1.0):
        return _linear_interpolate(tensor_a, tensor_b, alpha)

    dtype = tensor_a.dtype
    work_dtype = torch_module.float64 if dtype == getattr(torch_module, "float64", None) else torch_module.float32
    flat_a = tensor_a.detach().to(dtype=work_dtype).reshape(-1)
    flat_b = tensor_b.detach().to(dtype=work_dtype).reshape(-1)
    norm_a = torch_module.linalg.vector_norm(flat_a)
    norm_b = torch_module.linalg.vector_norm(flat_b)
    if float(norm_a.item()) == 0.0 or float(norm_b.item()) == 0.0:
        return _linear_interpolate(tensor_a, tensor_b, alpha)

    unit_a = flat_a / norm_a
    unit_b = flat_b / norm_b
    dot = torch_module.clamp(torch_module.sum(unit_a * unit_b), -1.0, 1.0)
    dot_value = float(dot.item())
    if abs(dot_value) > 0.9995:
        return _linear_interpolate(tensor_a, tensor_b, alpha)

    omega = torch_module.acos(dot)
    sin_omega = torch_module.sin(omega)
    if abs(float(sin_omega.item())) < 1e-8:
        return _linear_interpolate(tensor_a, tensor_b, alpha)

    scale_a = torch_module.sin((1.0 - alpha) * omega) / sin_omega
    scale_b = torch_module.sin(alpha * omega) / sin_omega
    merged = (scale_a * flat_a + scale_b * flat_b).reshape(tensor_a.shape)
    return merged.to(dtype=dtype)


_LAYER_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def weight_movement(before: dict[str, Any], after: dict[str, Any], torch_module: Any) -> dict[str, Any] | None:
    """How far ``after`` sits from ``before``, overall and per group (``emb``, each transformer layer, the heads).

    ``rel`` is |after - before| / |before| and ``cos`` the cosine between them, both over the group's floating
    weights; ``rms`` is the root mean square of ``before``, the scale a perturbation's sigma is measured against.
    """
    root = next((path for path in MERGE_WEIGHT_ROOTS if _has_nested(before, path) and _has_nested(after, path)), None)
    if root is None:
        return None
    sums: dict[str, list[float]] = {}  # group -> [params, |before|², |after|², before·after]

    def walk(a: Any, b: Any, prefix: str) -> None:
        for key in a:
            path = f"{prefix}{key}"
            if isinstance(a[key], dict) and isinstance(b.get(key), dict):
                walk(a[key], b[key], f"{path}.")
            elif isinstance(a[key], torch_module.Tensor) and a[key].is_floating_point() and key in b:
                x, y = a[key].detach().double(), b[key].detach().double()
                layer = _LAYER_KEY.search(path)
                acc = sums.setdefault(f"layer {layer.group(1)}" if layer else path.split(".")[0], [0.0] * 4)
                acc[0] += x.numel()
                acc[1] += float((x * x).sum())
                acc[2] += float((y * y).sum())
                acc[3] += float((x * y).sum())

    walk(_get_nested(before, root), _get_nested(after, root), "")

    def stats(params: float, bb: float, aa: float, ab: float) -> dict[str, Any]:
        return {
            "params": int(params),
            "rel": round(math.sqrt(max(aa + bb - 2 * ab, 0.0)) / math.sqrt(bb), 5) if bb else None,
            "cos": round(ab / math.sqrt(aa * bb), 5) if aa and bb else None,
            "rms": round(math.sqrt(bb / params), 5) if params else None,
        }

    def order(name: str) -> tuple[int, int, str]:
        layer = name.removeprefix("layer ")
        return (1, int(layer), "") if layer.isdigit() else (0 if name == "emb" else 2, 0, name)

    total = [sum(column) for column in zip(*sums.values(), strict=True)] if sums else [0.0] * 4
    return {
        "root": ".".join(root),
        "total": stats(*total),
        "groups": [{"name": name, **stats(*sums[name])} for name in sorted(sums, key=order)],
    }


def _trusted_torch_load(path: Path, torch_module: Any) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def _merge_helper_script() -> Path:
    return load_musicgen_project_config().repo_root / "scripts" / "run_musicgen_merge.py"


def _import_torch() -> Any:
    import torch

    return torch


def _has_nested(payload: dict[str, Any], path: Sequence[str]) -> bool:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _get_nested(payload: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for part in path:
        current = current[part]
    return current


def _set_nested(payload: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current: dict[str, Any] = payload
    for part in path[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot set nested merge output under non-dict key: {part}")
        current = child
    current[path[-1]] = value


def _file_snapshot(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_merge_helper_parser() -> Any:
    import argparse

    from ..cli.common import MacchinaRicordiHelpFormatter

    parser = argparse.ArgumentParser(
        description="Merge two local MusicGen checkpoint files.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    parser.add_argument("checkpoint_a", help="First source checkpoint.th path")
    parser.add_argument("checkpoint_b", help="Second source checkpoint.th path")
    parser.add_argument("output_checkpoint", help="Destination checkpoint.th path")
    parser.add_argument("--alpha", type=float, default=0.5, help="Interpolation amount: 0 keeps A, 1 keeps B")
    parser.add_argument("--method", choices=MERGE_METHODS, default="linear", help="Tensor interpolation method")
    parser.add_argument("--force", action="store_true", help="Overwrite output checkpoint if it exists")
    return parser


def merge_helper_main(argv: list[str] | None = None) -> int:
    parser = build_merge_helper_parser()
    args = parser.parse_args(argv)
    sidecar = merge_checkpoint_files(
        checkpoint_a=Path(args.checkpoint_a).expanduser().resolve(),
        checkpoint_b=Path(args.checkpoint_b).expanduser().resolve(),
        output_checkpoint=Path(args.output_checkpoint).expanduser().resolve(),
        method=args.method,
        alpha=args.alpha,
        force=args.force,
    )
    print(f"checkpoint: {sidecar['output']['path']}")
    print(f"merged_from: {Path(sidecar['output']['path']).parent / 'merged_from.json'}")
    print(f"merged_roots: {len(sidecar['merged_roots'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(merge_helper_main(sys.argv[1:]))
