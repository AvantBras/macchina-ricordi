"""Merge and perturb: derived checkpoints, made by ``model merge|perturb`` run as detached jobs.

Both load whole checkpoints with torch in the AudioCraft runtime, so the server never touches the
weights: it lists the zoo's derived checkpoints (``zoo.scan_derived``, from the ``merged_from.json`` and
``perturbed_from.json`` sidecars) and reads the XPs' Hydra configs, which give each checkpoint's model
shape. A merge needs both shapes to match, or the helper fails after loading gigabytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..models.provenance import resolve_checkpoint
from ..models.zoo import Zoo, build_zoo
from .jobs import DetachedJobs, relative

JOBS_DIR = "artifacts/derive_jobs"
JOB_SCHEMA = "macchina_ricordi.gui.derive_job.v1"
EXIT_MARKER = "[macchina_ricordi derive job exit"
OUTPUT_DIRS = {"merge": "artifacts/checkpoints/merged", "perturb": "artifacts/checkpoints/perturbed"}
MERGE_METHODS = ("linear", "slerp")
LAYER_SCOPES = ("all", "attention", "attention-projections", "lm-head")
SHAPE_KEYS = ("dim", "num_layers", "num_heads", "hidden_scale", "n_q", "card", "bias_ff", "bias_attn")
# heads split the same weight matrices differently: a mismatch merges, but the result is odd
WEIGHT_SHAPE_KEYS = tuple(key for key in SHAPE_KEYS if key != "num_heads")
PREVIEW_TTL_SECONDS = 30 * 60
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_TRANSFORMER_BLOCK = re.compile(r"^transformer_lm:\n((?:[ \t]+.*\n?)+)", re.MULTILINE)
_CONDITIONERS = re.compile(r"^conditioners:(.*)$", re.MULTILINE)


class DeriveRequestError(ValueError):
    pass


@dataclass(frozen=True)
class DeriveRequest:
    kind: str
    sources: tuple[str, ...]
    name: str
    method: str = "linear"
    alpha: float = 0.5
    sigma: float = 0.005
    layer_scope: str = "all"
    seed: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DeriveRequest:
        kind = str(payload.get("kind") or "")
        if kind not in OUTPUT_DIRS:
            raise DeriveRequestError("kind must be merge or perturb")
        sources = payload.get("sources") or []
        if not isinstance(sources, list) or not all(isinstance(s, str) and s for s in sources):
            raise DeriveRequestError("sources must be a list of checkpoint paths")
        if len(sources) != (2 if kind == "merge" else 1):
            raise DeriveRequestError("a merge takes two checkpoints" if kind == "merge" else "perturb takes one")
        try:
            alpha = float(payload.get("alpha", 0.5))
            sigma = float(payload.get("sigma", 0.005))
            seed = int(payload.get("seed", 0))
        except (TypeError, ValueError) as exc:
            raise DeriveRequestError("alpha, sigma and seed must be numbers") from exc
        method = str(payload.get("method") or "linear")
        scope = str(payload.get("layer_scope") or "all").strip()
        if kind == "merge":
            if method not in MERGE_METHODS:
                raise DeriveRequestError("the method is linear or slerp")
            if not 0.0 <= alpha <= 1.0:
                raise DeriveRequestError("alpha is between 0 (all A) and 1 (all B)")
        else:
            if not math.isfinite(sigma) or not 0.0 < sigma <= 1.0:
                raise DeriveRequestError("sigma is between 0 and 1; 0.005 to 0.02 is gentle")
            if scope not in LAYER_SCOPES and not (scope.startswith("contains:") and scope.removeprefix("contains:")):
                raise DeriveRequestError(f"the layer scope is one of {', '.join(LAYER_SCOPES)}, or contains:<fragment>")
            if seed < 0:
                raise DeriveRequestError("the seed is 0 or more")
        name = str(payload.get("name") or "").strip()
        if not _NAME.match(name):
            raise DeriveRequestError("the name uses letters, digits, '.', '_' or '-' (80 at most)")
        return cls(
            kind=kind,
            sources=tuple(sources),
            name=name,
            method=method if kind == "merge" else "linear",
            alpha=alpha if kind == "merge" else 0.5,
            sigma=sigma if kind == "perturb" else 0.005,
            layer_scope=scope if kind == "perturb" else "all",
            seed=seed if kind == "perturb" else 0,
        )

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]


def xp_sig_of(path: Path) -> str | None:
    parts = path.parts
    for index in range(len(parts) - 2):
        if parts[index] == ".dora" and parts[index + 1] == "xps":
            return parts[index + 2]
    return None


def read_model_shape(config_path: Path) -> dict[str, Any] | None:
    """The weight-shaping keys of an XP's resolved Hydra config, read without a YAML parser."""
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    block = _TRANSFORMER_BLOCK.search(text)
    if block is None:
        return None
    indent = re.match(r"[ \t]+", block.group(1)).group(0)  # nested mappings sit deeper; skip them
    values = dict(re.findall(rf"^{indent}(\w+):[ \t]*(.*?)[ \t]*$", block.group(1), re.MULTILINE))
    shape: dict[str, Any] = {key: values.get(key) for key in SHAPE_KEYS}
    conditioners = _CONDITIONERS.search(text)
    shape["conditioned"] = bool(conditioners) and conditioners.group(1).strip() not in ("null", "{}")
    return shape


def shape_label(shape: dict[str, Any] | None) -> str | None:
    if not shape:
        return None
    return f"dim {shape.get('dim')} · {shape.get('num_layers')} layers · {shape.get('num_heads')} heads" + (
        " · conditioned" if shape.get("conditioned") else ""
    )


def _gib(size: int) -> str:
    return f"{size / 2**30:.1f} GiB"


def _mem_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass
class DeriveManager:
    repo_root: Path
    cli_prefix: tuple[str, ...] | None = None
    musicgen_check: Callable[[], str | None] | None = None
    """Returns why merge and perturb cannot run here (no AudioCraft runtime), or None."""
    mem_available: Callable[[], int | None] = _mem_available
    zoo: Callable[[], Zoo] | None = None
    """The server's cached zoo; without one, each listing reads the models afresh."""
    _previews: dict[str, tuple[DeriveRequest, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        self._jobs = DetachedJobs(self.repo_root, JOBS_DIR, JOB_SCHEMA, EXIT_MARKER)

    # ── what exists ──

    def model_shape(self, checkpoint: Path) -> dict[str, Any] | None:
        """From the XP the file belongs to, else (for a derived file) from the models behind it."""
        sig = xp_sig_of(checkpoint)
        sigs = (sig,) if sig else resolve_checkpoint(str(checkpoint)).parent_sigs()
        for parent in sigs:
            shape = read_model_shape(self.repo_root / ".dora" / "xps" / parent / ".hydra" / "config.yaml")
            if shape:
                return shape
        return None

    def _describe_source(self, path: Path) -> dict[str, Any]:
        sig = xp_sig_of(path)
        return {
            "path": relative(path, self.repo_root),
            "sig": sig,
            "file": path.name,
            "exists": path.is_file(),
        }

    def checkpoints(self) -> list[dict[str, Any]]:
        """The zoo's merge and perturbation outputs (``zoo.scan_derived``), newest first."""
        zoo = self.zoo() if self.zoo is not None else build_zoo(self.repo_root, include_sessions=False)
        rows = []
        for derived in zoo.derived:
            payload, checkpoint = derived.payload, derived.checkpoint
            output = payload.get("output") or {}
            rows.append(
                {
                    "kind": derived.kind,
                    "name": checkpoint.parent.name if checkpoint.name == "checkpoint.th" else checkpoint.stem,
                    "path": relative(checkpoint, self.repo_root),
                    "exists": checkpoint.is_file(),
                    "size_mb": round(checkpoint.stat().st_size / 1e6, 1) if checkpoint.is_file() else None,
                    "created_at": payload.get("created_at"),
                    "method": payload.get("method"),
                    "alpha": payload.get("alpha"),
                    "sigma": payload.get("sigma"),
                    "layer_scope": payload.get("layer_scope"),
                    "seed": payload.get("seed"),
                    "movement": payload.get("movement"),
                    "sources": [self._describe_source(Path(path)) for _, path in derived.sources],
                    "shape": shape_label(self.model_shape(checkpoint)),
                    "children": list(zoo.warm_started_from(checkpoint, sha256=output.get("sha256"))),
                    "warnings": payload.get("warnings") or [],
                    # files in the same folder whose sidecar was overwritten by a later one
                    "unrecorded": [relative(path, self.repo_root) for path in derived.unrecorded],
                }
            )
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
        return rows

    def overview(self) -> dict[str, Any]:
        return {
            "checkpoints": self.checkpoints(),
            "jobs": self.jobs(),
            "runtime": self.musicgen_check() if self.musicgen_check else None,
            "layer_scopes": list(LAYER_SCOPES),
        }

    def _checkpoint(self, value: str) -> Path:
        path = (self.repo_root / value).resolve()
        if not path.is_relative_to(self.repo_root) or path.suffix != ".th" or not path.is_file():
            raise DeriveRequestError(f"no checkpoint file {value!r} in this repo")
        return path

    def describe(self, value: str) -> dict[str, Any]:
        path = self._checkpoint(value)
        shape = self.model_shape(path)
        origin = resolve_checkpoint(str(path))
        return {
            **self._describe_source(path),
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "shape": shape,
            "shape_label": shape_label(shape),
            "derived": origin.kind if origin.kind in ("merge", "perturb") else None,
        }

    # ── preview / launch ──

    def _output(self, request: DeriveRequest) -> Path:
        return self.repo_root / OUTPUT_DIRS[request.kind] / request.name / "checkpoint.th"

    def _cli(self) -> list[str]:
        return list(self.cli_prefix or (sys.executable, "-m", "macchina_ricordi.cli"))

    def _command(self, request: DeriveRequest, sources: list[Path], output: Path) -> list[str]:
        paths = [relative(path, self.repo_root) for path in [*sources, output]]
        if request.kind == "merge":
            return [
                *self._cli(), "model", "merge", *paths,
                "--alpha", f"{request.alpha:g}", "--method", request.method,
            ]  # fmt: skip
        return [
            *self._cli(), "model", "perturb", *paths,
            "--sigma", f"{request.sigma:g}", "--layer-scope", request.layer_scope, "--seed", str(request.seed),
        ]  # fmt: skip

    def preview(self, request: DeriveRequest) -> dict[str, Any]:
        sources = [self._checkpoint(value) for value in request.sources]
        output = self._output(request)
        blockers, warnings = [], []
        if len(set(sources)) != len(sources):
            blockers.append("pick two different checkpoints")
        if output.parent.exists():
            blockers.append(f"{relative(output.parent, self.repo_root)} exists; pick another name")
        problem = self.musicgen_check() if self.musicgen_check else None
        if problem:
            blockers.append(problem)
        shapes = [self.model_shape(path) for path in sources]
        if request.kind == "merge":
            if all(shapes):
                differ = [key for key in WEIGHT_SHAPE_KEYS if shapes[0].get(key) != shapes[1].get(key)]
                if shapes[0].get("conditioned") != shapes[1].get("conditioned"):
                    differ.append("conditioning")
                if differ:
                    blockers.append(
                        "the two models are not the same shape (" + ", ".join(differ) + "): "
                        "their weights cannot be blended"
                    )
                elif shapes[0].get("num_heads") != shapes[1].get("num_heads"):
                    warnings.append(
                        "the two models split attention into a different number of heads: the weights blend, "
                        "but each head of the result mixes unrelated parts of the two"
                    )
            else:
                warnings.append(
                    "the shape of one of the models is unknown (not from an XP), so it is not checked: "
                    "the merge fails if they differ"
                )
        sizes = [path.stat().st_size for path in sources]
        memory = sum(sizes) + max(sizes) // 2
        available = self.mem_available()
        if available is not None and memory > available:
            warnings.append(
                f"it loads {_gib(sum(sizes))} of checkpoints into memory and about {_gib(available)} is free; "
                "it may swap, or be killed"
            )
        key = hashlib.sha256(
            (request.key() + "".join(f"{p}:{p.stat().st_mtime_ns}" for p in sources)).encode()
        ).hexdigest()[:20]
        if not blockers:
            with self._lock:
                now = time.time()
                self._previews = {k: v for k, v in self._previews.items() if now - v[1] < PREVIEW_TTL_SECONDS}
                self._previews[key] = (request, now)
        return {
            "kind": request.kind,
            "sources": [
                {**self._describe_source(path), "size_mb": round(size / 1e6, 1), "shape": shape_label(shape)}
                for path, size, shape in zip(sources, sizes, shapes, strict=True)
            ],
            "output": relative(output, self.repo_root),
            "memory_gib": round(memory / 2**30, 1),
            "command": shlex.join(self._command(request, sources, output)),
            "blockers": blockers,
            "warnings": warnings,
            "key": None if blockers else key,
        }

    def launch(self, request: DeriveRequest, key: str) -> dict[str, Any]:
        with self._lock:
            preview = self._previews.get(key)
            if preview is None or preview[0] != request or time.time() - preview[1] > PREVIEW_TTL_SECONDS:
                raise DeriveRequestError("the form changed or the preview is old; preview again")
            running = [job for job in self.jobs() if job["active"]]
            if running:
                raise DeriveRequestError(f"{running[0]['title']} is still running; wait for it")
        fresh = self.preview(request)
        if fresh["key"] != key:
            raise DeriveRequestError("the checkpoints changed since the preview; preview again")
        sources = [self._checkpoint(value) for value in request.sources]
        output = self._output(request)
        if request.kind == "merge":
            title = f"merge {request.name} ({request.method}, alpha {request.alpha:g})"
        else:
            title = f"perturb {request.name} (sigma {request.sigma:g}, {request.layer_scope})"
        record = self._jobs.launch(
            request.kind,
            title,
            [self._command(request, sources, output)],
            {"output": relative(output, self.repo_root), "request": asdict(request)},
        )
        with self._lock:
            self._previews.pop(key, None)
        return self.job(record["id"])

    # ── jobs ──

    def _summarize(self, record: dict[str, Any]) -> dict[str, Any]:
        summary = self._jobs.summarize(record)
        output = record.get("output")
        return {
            **summary,
            "output": output,
            "ready": summary["state"] == "completed" and bool(output) and (self.repo_root / str(output)).is_file(),
        }

    def jobs(self) -> list[dict[str, Any]]:
        return [self._summarize(record) for _, record in self._jobs.records()]

    def job(self, job_id: str) -> dict[str, Any]:
        record = self._jobs.record(job_id)
        if record is None:
            raise DeriveRequestError(f"no merge or perturb job {job_id!r}")
        return {**self._summarize(record), "log": self._jobs.log(record)}
