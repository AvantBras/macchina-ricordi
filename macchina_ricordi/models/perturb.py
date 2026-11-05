from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..musicgen.config import load_musicgen_project_config, musicgen_script_env
from ..musicgen.runtime import require_musicgen_python
from ..musicgen.shared import utc_now_iso, write_json
from .merge import (
    _file_snapshot,
    _get_nested,
    _has_nested,
    _import_torch,
    _set_nested,
    _trusted_torch_load,
    weight_movement,
)

PERTURB_SIDECAR_SCHEMA_VERSION = 1
PERTURB_BUILTIN_LAYER_SCOPES = ("all", "attention", "attention-projections", "lm-head")
PERTURB_WEIGHT_ROOTS: tuple[tuple[str, ...], ...] = (
    ("model",),
    ("best_state", "model"),
    ("fsdp_best_state", "model"),
)


@dataclass(frozen=True)
class CheckpointPerturbPlan:
    source_checkpoint: Path
    output_checkpoint: Path
    sidecar_path: Path
    sigma: float
    layer_scope: str
    seed: int
    musicgen_python: str
    command: list[str]
    dry_run: bool
    force: bool


@dataclass(frozen=True)
class CheckpointPerturbResult:
    output_checkpoint: Path
    sidecar_path: Path
    dry_run: bool


@dataclass(frozen=True)
class PerturbedRootStats:
    path: str
    tensor_count: int
    floating_tensor_count: int
    perturbed_tensor_count: int
    copied_floating_tensor_count: int
    copied_non_floating_tensor_count: int
    parameter_count: int
    perturbed_parameter_count: int


@dataclass(frozen=True)
class PerturbStateResult:
    state: dict[str, Any]
    roots: tuple[PerturbedRootStats, ...]
    warnings: tuple[str, ...]


def plan_checkpoint_perturb(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    sigma: float,
    layer_scope: str = "all",
    seed: int = 0,
    musicgen_python: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> CheckpointPerturbPlan:
    _validate_sigma(sigma)
    _validate_layer_scope(layer_scope)
    _validate_seed(seed)

    resolved_source = Path(source_checkpoint).expanduser().resolve()
    resolved_output = Path(output_checkpoint).expanduser().resolve()
    if not resolved_source.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {resolved_source}")
    from .bundle import is_bundle, read_bundle

    bundle_input = resolved_source.is_dir() and is_bundle(resolved_source)
    if bundle_input:
        read_bundle(resolved_source)
    elif not resolved_source.is_file():
        raise FileNotFoundError(f"Source is not a checkpoint file or model bundle: {resolved_source}")
    if resolved_output == resolved_source:
        raise ValueError("Output checkpoint must not overwrite the source checkpoint.")

    if bundle_input and resolved_source in resolved_output.parents:
        raise ValueError("Output bundle must not be inside the source bundle.")
    sidecar_path = resolved_output / "config.json" if bundle_input else resolved_output.parent / "perturbed_from.json"
    if resolved_output.exists() and resolved_output.is_dir() and not bundle_input:
        raise IsADirectoryError(f"Output checkpoint path is a directory: {resolved_output}")
    if resolved_output.exists() and not force:
        raise FileExistsError(f"Output checkpoint already exists: {resolved_output}. Pass --force to overwrite.")
    if sidecar_path.exists() and not force:
        raise FileExistsError(f"Perturb sidecar already exists: {sidecar_path}. Pass --force to overwrite.")

    resolved_python = require_musicgen_python(musicgen_python)
    helper_script = _perturb_helper_script()
    if not helper_script.exists():
        raise FileNotFoundError(f"MusicGen perturb helper script not found: {helper_script}")

    command = [
        resolved_python,
        str(helper_script),
        str(resolved_source),
        str(resolved_output),
        "--sigma",
        str(sigma),
        "--layer-scope",
        layer_scope,
        "--seed",
        str(seed),
    ]
    if force:
        command.append("--force")

    return CheckpointPerturbPlan(
        source_checkpoint=resolved_source,
        output_checkpoint=resolved_output,
        sidecar_path=sidecar_path,
        sigma=sigma,
        layer_scope=layer_scope,
        seed=seed,
        musicgen_python=resolved_python,
        command=command,
        dry_run=dry_run,
        force=force,
    )


def execute_checkpoint_perturb(plan: CheckpointPerturbPlan) -> CheckpointPerturbResult:
    if plan.dry_run:
        return CheckpointPerturbResult(
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
        message = f"MusicGen checkpoint perturb failed with exit code {exc.returncode}."
        if detail:
            message = f"{message} {detail}"
        raise RuntimeError(message) from exc

    return CheckpointPerturbResult(
        output_checkpoint=plan.output_checkpoint,
        sidecar_path=plan.sidecar_path,
        dry_run=False,
    )


def perturb_checkpoint_files(
    *,
    source_checkpoint: Path,
    output_checkpoint: Path,
    sigma: float,
    layer_scope: str,
    seed: int,
    force: bool,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    _validate_sigma(sigma)
    _validate_layer_scope(layer_scope)
    _validate_seed(seed)

    from .bundle import is_bundle

    if source_checkpoint.is_dir() and is_bundle(source_checkpoint):
        return perturb_bundle_files(
            source_bundle=source_checkpoint,
            output_bundle=output_checkpoint,
            sigma=sigma,
            layer_scope=layer_scope,
            seed=seed,
            force=force,
            torch_module=torch_module,
        )
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_checkpoint}")
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Source checkpoint is not a file: {source_checkpoint}")
    if output_checkpoint.expanduser().resolve() == source_checkpoint.expanduser().resolve():
        raise ValueError("Output checkpoint must not overwrite the source checkpoint.")

    sidecar_path = output_checkpoint.parent / "perturbed_from.json"
    if output_checkpoint.exists() and output_checkpoint.is_dir():
        raise IsADirectoryError(f"Output checkpoint path is a directory: {output_checkpoint}")
    if output_checkpoint.exists() and not force:
        raise FileExistsError(f"Output checkpoint already exists: {output_checkpoint}. Pass --force to overwrite.")
    if sidecar_path.exists() and not force:
        raise FileExistsError(f"Perturb sidecar already exists: {sidecar_path}. Pass --force to overwrite.")
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    torch = torch_module or _import_torch()
    state = _trusted_torch_load(source_checkpoint, torch)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must load to a top-level dictionary.")

    perturbed = perturb_checkpoint_state(
        state,
        sigma=sigma,
        layer_scope=layer_scope,
        seed=seed,
        torch_module=torch,
    )
    torch.save(perturbed.state, output_checkpoint)

    source_metadata = _load_source_resume_metadata(source_checkpoint.parent)
    sidecar_payload = {
        "schema_version": PERTURB_SIDECAR_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "operation": "musicgen_checkpoint_perturb",
        "sigma": sigma,
        "layer_scope": layer_scope,
        "seed": seed,
        "sources": {
            "source": _file_snapshot(source_checkpoint),
        },
        "output": _file_snapshot(output_checkpoint),
        "perturbed_roots": [
            {
                "path": root.path,
                "tensor_count": root.tensor_count,
                "floating_tensor_count": root.floating_tensor_count,
                "perturbed_tensor_count": root.perturbed_tensor_count,
                "copied_floating_tensor_count": root.copied_floating_tensor_count,
                "copied_non_floating_tensor_count": root.copied_non_floating_tensor_count,
                "parameter_count": root.parameter_count,
                "perturbed_parameter_count": root.perturbed_parameter_count,
            }
            for root in perturbed.roots
        ],
        "warnings": list(perturbed.warnings),
        "movement": {"between": ["source", "output"], **(weight_movement(state, perturbed.state, torch) or {})},
    }
    if source_metadata:
        sidecar_payload["source_resume_metadata"] = source_metadata

    write_json(sidecar_path, sidecar_payload)
    return sidecar_payload


def perturb_bundle_files(
    *,
    source_bundle: Path,
    output_bundle: Path,
    sigma: float,
    layer_scope: str,
    seed: int,
    force: bool = False,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Write an independently loadable snapshot; no original XP or parent file is required."""
    from .bundle import load_bundle_tensors, read_bundle, write_bundle

    source = read_bundle(source_bundle)
    if source.path == output_bundle.expanduser().resolve() or source.path in output_bundle.resolve().parents:
        raise ValueError("Output bundle must not overwrite or be inside the source bundle.")
    tensors = load_bundle_tensors(source.path, device="cpu")
    perturbed = perturb_checkpoint_state(
        {"model": tensors},
        sigma=sigma,
        layer_scope=layer_scope,
        seed=seed,
        torch_module=torch_module,
    )
    provenance = {
        "operation": "perturb",
        "parent_sha256": source.config["weights"]["sha256"],
        "parent_name": source.config["name"],
        "sigma": sigma,
        "layer_scope": layer_scope,
        "seed": seed,
    }
    child = write_bundle(
        output_bundle,
        perturbed.state["model"],
        model_config=source.config["model_config"],
        codec=source.config["codec"],
        audio=source.config["audio"],
        segment_duration=source.config["segment_duration"],
        name=output_bundle.name,
        provenance=provenance,
        weights_state="model",
        force=force,
        license_text=(source.path / "LICENSE").read_text(encoding="utf-8"),
    )
    return {
        "output": {"path": str(child.path)},
        "provenance_path": str(child.path / "config.json"),
        "perturbed_roots": [
            {"path": root.path, "parameter_count": root.perturbed_parameter_count} for root in perturbed.roots
        ],
    }


def perturb_checkpoint_state(
    state: dict[str, Any],
    *,
    sigma: float,
    layer_scope: str,
    seed: int,
    torch_module: Any | None = None,
) -> PerturbStateResult:
    _validate_sigma(sigma)
    _validate_layer_scope(layer_scope)
    _validate_seed(seed)

    torch = torch_module or _import_torch()
    generator = _build_generator(torch, seed)
    output: dict[str, Any] = {}
    roots: list[PerturbedRootStats] = []
    warnings: list[str] = []

    for root_path in PERTURB_WEIGHT_ROOTS:
        root_label = ".".join(root_path)
        if not _has_nested(state, root_path):
            continue
        source_root = _get_nested(state, root_path)
        if not isinstance(source_root, dict):
            raise TypeError(f"Perturb root {root_label} must be a dictionary.")
        perturbed_root, stats = _perturb_tensor_mapping(
            source_root,
            root_path=root_label,
            sigma=sigma,
            layer_scope=layer_scope,
            torch_module=torch,
            generator=generator,
        )
        _set_nested(output, root_path, perturbed_root)
        roots.append(stats)

    if not roots:
        raise ValueError(
            "No compatible weight roots found. Expected at least one of: "
            + ", ".join(".".join(path) for path in PERTURB_WEIGHT_ROOTS)
        )
    if sum(root.perturbed_tensor_count for root in roots) == 0:
        raise ValueError(f"Layer scope {layer_scope!r} did not match any floating tensors.")

    return PerturbStateResult(state=output, roots=tuple(roots), warnings=tuple(warnings))


def perturb_tensor_value(
    tensor: Any,
    *,
    sigma: float,
    torch_module: Any | None = None,
    generator: Any | None = None,
) -> Any:
    _validate_sigma(sigma)
    if not tensor.is_floating_point():
        return tensor.detach().clone() if hasattr(tensor, "detach") else tensor
    base = tensor.detach().clone() if hasattr(tensor, "detach") else tensor
    if sigma == 0.0:
        return base
    torch = torch_module or _import_torch()
    noise = _randn_like(base, torch_module=torch, generator=generator)
    return base + noise * sigma


def _perturb_tensor_mapping(
    mapping: dict[str, Any],
    *,
    root_path: str,
    sigma: float,
    layer_scope: str,
    torch_module: Any,
    generator: Any | None,
) -> tuple[dict[str, Any], PerturbedRootStats]:
    perturbed: dict[str, Any] = {}
    tensor_count = 0
    floating_tensor_count = 0
    perturbed_tensor_count = 0
    copied_floating_tensor_count = 0
    copied_non_floating_tensor_count = 0
    parameter_count = 0
    perturbed_parameter_count = 0
    tensor_type = torch_module.Tensor

    for key in sorted(mapping):
        value = mapping[key]
        child_path = f"{root_path}.{key}"
        if isinstance(value, dict):
            child, child_stats = _perturb_tensor_mapping(
                value,
                root_path=child_path,
                sigma=sigma,
                layer_scope=layer_scope,
                torch_module=torch_module,
                generator=generator,
            )
            perturbed[key] = child
            tensor_count += child_stats.tensor_count
            floating_tensor_count += child_stats.floating_tensor_count
            perturbed_tensor_count += child_stats.perturbed_tensor_count
            copied_floating_tensor_count += child_stats.copied_floating_tensor_count
            copied_non_floating_tensor_count += child_stats.copied_non_floating_tensor_count
            parameter_count += child_stats.parameter_count
            perturbed_parameter_count += child_stats.perturbed_parameter_count
            continue
        if not isinstance(value, tensor_type):
            perturbed[key] = value
            continue

        tensor_count += 1
        if not value.is_floating_point():
            perturbed[key] = value.detach().clone() if hasattr(value, "detach") else value
            copied_non_floating_tensor_count += 1
            continue

        floating_tensor_count += 1
        parameter_count += int(value.numel())
        if _matches_layer_scope(child_path, layer_scope):
            perturbed[key] = perturb_tensor_value(
                value,
                sigma=sigma,
                torch_module=torch_module,
                generator=generator,
            )
            perturbed_tensor_count += 1
            perturbed_parameter_count += int(value.numel())
        else:
            perturbed[key] = value.detach().clone() if hasattr(value, "detach") else value
            copied_floating_tensor_count += 1

    return perturbed, PerturbedRootStats(
        path=root_path,
        tensor_count=tensor_count,
        floating_tensor_count=floating_tensor_count,
        perturbed_tensor_count=perturbed_tensor_count,
        copied_floating_tensor_count=copied_floating_tensor_count,
        copied_non_floating_tensor_count=copied_non_floating_tensor_count,
        parameter_count=parameter_count,
        perturbed_parameter_count=perturbed_parameter_count,
    )


def _matches_layer_scope(path: str, layer_scope: str) -> bool:
    if layer_scope == "all":
        return True
    lowered = path.lower()
    normalized = lowered.replace("_", "-").replace(".", "-")
    if layer_scope.startswith("contains:"):
        return layer_scope.removeprefix("contains:").lower() in lowered
    if layer_scope == "attention":
        return "attn" in normalized or "attention" in normalized
    if layer_scope == "attention-projections":
        if "attn" not in normalized and "attention" not in normalized:
            return False
        projection_markers = ("q-proj", "k-proj", "v-proj", "in-proj", "out-proj", "proj")
        return any(marker in normalized for marker in projection_markers)
    if layer_scope == "lm-head":
        return any(marker in normalized for marker in ("lm-head", "linears", "final-proj", "output-proj"))
    return False


def _validate_sigma(sigma: float) -> None:
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("--sigma must be a finite value greater than or equal to 0.")


def _validate_layer_scope(layer_scope: str) -> None:
    if layer_scope in PERTURB_BUILTIN_LAYER_SCOPES:
        return
    if layer_scope.startswith("contains:") and layer_scope.removeprefix("contains:"):
        return
    expected = ", ".join(PERTURB_BUILTIN_LAYER_SCOPES) + ", or contains:<checkpoint-key-fragment>"
    raise ValueError(f"Unsupported layer scope: {layer_scope!r}. Expected {expected}.")


def _validate_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("--seed must be greater than or equal to 0.")


def _build_generator(torch_module: Any, seed: int) -> Any | None:
    generator_factory = getattr(torch_module, "Generator", None)
    if generator_factory is None:
        manual_seed = getattr(torch_module, "manual_seed", None)
        if manual_seed is not None:
            manual_seed(seed)
        return None
    try:
        generator = generator_factory(device="cpu")
    except TypeError:
        generator = generator_factory()
    generator.manual_seed(seed)
    return generator


def _randn_like(tensor: Any, *, torch_module: Any, generator: Any | None) -> Any:
    try:
        return torch_module.randn_like(tensor, generator=generator)
    except TypeError:
        kwargs: dict[str, Any] = {}
        if generator is not None:
            kwargs["generator"] = generator
        if hasattr(tensor, "dtype"):
            kwargs["dtype"] = tensor.dtype
        if hasattr(tensor, "device"):
            kwargs["device"] = tensor.device
        return torch_module.randn(tensor.shape, **kwargs)


def _load_source_resume_metadata(source_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, filename in (("merged_from", "merged_from.json"), ("perturbed_from", "perturbed_from.json")):
        payload = _safe_load_json(source_dir / filename)
        if isinstance(payload, dict):
            metadata[key] = payload
    return metadata


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _perturb_helper_script() -> Path:
    return load_musicgen_project_config().repo_root / "scripts" / "run_musicgen_perturb.py"


def build_perturb_helper_parser() -> Any:
    import argparse

    from ..cli.common import MacchinaRicordiHelpFormatter

    parser = argparse.ArgumentParser(
        description="Add controlled Gaussian noise to a local checkpoint or portable model bundle.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    parser.add_argument("source_checkpoint", help="Source trusted checkpoint.th or portable bundle directory")
    parser.add_argument(
        "output_checkpoint", help="Destination checkpoint.th or new bundle directory, matching the input"
    )
    parser.add_argument("--sigma", type=float, required=True, help="Gaussian noise scale")
    parser.add_argument("--layer-scope", default="all", help="Tensor subset to perturb")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Overwrite output checkpoint if it exists")
    return parser


def perturb_helper_main(argv: list[str] | None = None) -> int:
    parser = build_perturb_helper_parser()
    args = parser.parse_args(argv)
    sidecar = perturb_checkpoint_files(
        source_checkpoint=Path(args.source_checkpoint).expanduser().resolve(),
        output_checkpoint=Path(args.output_checkpoint).expanduser().resolve(),
        sigma=args.sigma,
        layer_scope=args.layer_scope,
        seed=args.seed,
        force=args.force,
    )
    print(f"checkpoint: {sidecar['output']['path']}")
    provenance_path = sidecar.get("provenance_path") or str(
        Path(sidecar["output"]["path"]).parent / "perturbed_from.json"
    )
    print(f"provenance: {provenance_path}")
    print(f"perturbed_roots: {len(sidecar['perturbed_roots'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(perturb_helper_main(sys.argv[1:]))
