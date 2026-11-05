from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..musicgen.shared import utc_now_iso, write_json
from ..training.plan import TRAIN_METADATA_FILENAME


@dataclass(frozen=True)
class BackfillResult:
    xp_sig: str
    action: str  # "written", "skipped_exists", "skipped_no_hydra", "error"
    metadata_path: str | None = None
    workspace_dir: str | None = None
    label: str | None = None
    merged_from: bool = False
    lineage_sidecar: bool = False
    error: str | None = None


@dataclass(frozen=True)
class BackfillReport:
    results: tuple[BackfillResult, ...]
    dry_run: bool

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def written(self) -> int:
        return sum(1 for r in self.results if r.action == "written")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.action.startswith("skipped"))

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.action == "error")


def _parse_overrides(overrides_path: Path) -> dict[str, str]:
    """Parse a Dora overrides.yaml into a {key: value} dict.

    When the same key appears more than once (e.g. two optim.epochs= entries),
    the last occurrence wins — matching Dora semantics.
    """
    if not overrides_path.exists():
        return {}
    lines = overrides_path.read_text(encoding="utf-8").strip().splitlines()
    parsed: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        if "=" in line:
            key, _, value = line.partition("=")
            parsed[key] = value
    return parsed


def _infer_workspace_from_datasource(datasource_path: str | None) -> Path | None:
    """Heuristic: strip trailing /egs/<dataset>/<split> to find the workspace root."""
    if datasource_path is None:
        return None
    path = Path(datasource_path)
    # Walk up until we find a directory that looks like a workspace
    for candidate in (path, *path.parents):
        if (candidate / "logs").is_dir() and (candidate / "egs").is_dir():
            return candidate
    return None


def _load_merged_from_sidecar(continue_from: str | None) -> dict | None:
    """If continue_from points to a local file, look for merged_from.json in its parent dir."""
    if continue_from is None:
        return None
    # Only local paths (not //pretrained/ refs)
    if continue_from.startswith("//"):
        return None
    sidecar = Path(continue_from).parent / "merged_from.json"
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_perturbed_from_sidecar(continue_from: str | None) -> dict | None:
    if continue_from is None:
        return None
    if continue_from.startswith("//"):
        return None
    sidecar = Path(continue_from).parent / "perturbed_from.json"
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_dataset_fingerprint(workspace_dir: Path | None) -> dict | None:
    if workspace_dir is None:
        return None
    fingerprint_path = workspace_dir / "logs" / "dataset_fingerprint.json"
    if not fingerprint_path.exists():
        return None
    try:
        return json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _reconstruct_command(overrides: dict[str, str]) -> list[str]:
    """Reconstruct a Dora command list from parsed overrides.

    Best-effort: base solver/dset/conditioner/model/encodec defaults
    are assumed to be the micro-unconditional profile defaults,
    then overrides are appended.
    """
    base = [
        "solver=musicgen/musicgen_base_32khz",
        "dset=audio/macchina_ricordi_micro_uncond",
        "conditioner=none",
        "model/lm/model_scale=micro",
        "compression_model_checkpoint=//pretrained/facebook/encodec_32khz",
    ]
    # Append all overrides in a stable order: known keys first, then the rest
    known_order = [
        "dataset.batch_size",
        "dataset.segment_duration",
        "dataset.num_workers",
        "dataset.valid.num_samples",
        "generate.every",
        "evaluate.every",
        "checkpoint.save_every",
        "checkpoint.keep_last",
        "optim.optimizer",
        "optim.lr",
        "optim.adam.weight_decay",
        "optim.epochs",
        "transformer_lm.memory_efficient",
        "autocast",
        "label",
        "continue_from",
    ]
    extras: dict[str, str] = {}
    for key in known_order:
        if key in overrides:
            base.append(f"{key}={overrides[key]}")
    for key, value in sorted(overrides.items()):
        if key not in known_order:
            extras[key] = value
    for key, value in extras.items():
        base.append(f"{key}={value}")
    return base


def build_backfill_metadata(
    xp_folder: Path,
    overrides: dict[str, str],
    datasource_train: str | None,
) -> dict:
    """Assemble metadata dict matching the structure written by plan_musicgen_training()."""
    workspace_dir = _infer_workspace_from_datasource(datasource_train)
    fingerprint = _load_dataset_fingerprint(workspace_dir)

    label = overrides.get("label")
    continue_from = overrides.get("continue_from")
    merged_from = _load_merged_from_sidecar(continue_from)
    perturbed_from = _load_perturbed_from_sidecar(continue_from)

    command = _reconstruct_command(overrides)

    fingerprint_summary = None
    fingerprint_digest = None
    if isinstance(fingerprint, dict):
        fingerprint_summary = {
            key: fingerprint.get(key)
            for key in (
                "profile",
                "dataset_name",
                "train_aggregate_sha256",
                "valid_aggregate_sha256",
                "train_count",
                "valid_count",
                "schema_version",
            )
            if key in fingerprint
        }
        digest = fingerprint.get("train_aggregate_sha256", "")
        if isinstance(digest, str) and len(digest) >= 12:
            fingerprint_digest = digest[:12]

    metadata: dict = {
        "profile": "micro-unconditional",
        "dataset_name": "macchina_ricordi_micro_uncond",
        "workspace_dir": str(workspace_dir) if workspace_dir is not None else None,
        "dataset_fingerprint_digest": fingerprint_digest,
        "dataset_fingerprint": fingerprint_summary,
        "command": command,
        "effective_label": label,
        "auto_dataset_label": False,
        "xp_sig": xp_folder.name,
        "xp_folder": str(xp_folder),
    }

    if continue_from is not None:
        metadata["init_from_checkpoint"] = _build_backfill_init_metadata(
            continue_from=continue_from,
            merged_from=merged_from,
            perturbed_from=perturbed_from,
        )

    return metadata


def _build_backfill_init_metadata(
    *,
    continue_from: str,
    merged_from: dict | None,
    perturbed_from: dict | None = None,
) -> dict:
    source_resume_metadata: dict | None = None
    warnings: list[str] = []

    lineage_metadata: dict[str, dict] = {}
    if isinstance(merged_from, dict):
        lineage_metadata["merged_from"] = merged_from
    if isinstance(perturbed_from, dict):
        lineage_metadata["perturbed_from"] = perturbed_from

    if lineage_metadata:
        source_resume_metadata = lineage_metadata
        warnings.append(
            "Reconstructed from backfill: source checkpoint has local lineage sidecars "
            "without dataset fingerprint; dataset compatibility cannot be inspected."
        )
    else:
        warnings.append(
            "Reconstructed from backfill: source checkpoint metadata unavailable; "
            "dataset compatibility cannot be inspected."
        )

    init_metadata: dict = {
        "origin": "local",
        "user_input": continue_from,
        "resolved_local_path": continue_from,
        "source_resume_metadata": source_resume_metadata,
        "dataset_fingerprint_matches_current_workspace": None,
        "warnings": warnings,
    }
    # Add backfill marker so consumers know this was reconstructed
    init_metadata["_backfilled"] = True
    return init_metadata


def backfill_xps(
    dora_dir: Path,
    *,
    dry_run: bool = True,
    force: bool = False,
) -> BackfillReport:
    """Scan .dora/xps/* and backfill macchina_ricordi_train_metadata.json for each XP.

    Args:
        dora_dir: Path to the `.dora` directory.
        dry_run: If True, report what would be done without writing.
        force: If True, overwrite existing metadata files.

    Returns:
        BackfillReport summarising actions taken.
    """
    xps_dir = dora_dir / "xps"
    if not xps_dir.is_dir():
        return BackfillReport(results=(), dry_run=dry_run)

    results: list[BackfillResult] = []
    for xp_folder in sorted(xps_dir.iterdir()):
        if not xp_folder.is_dir():
            continue

        sig = xp_folder.name
        result = _backfill_one_xp(xp_folder, sig, dry_run=dry_run, force=force)
        results.append(result)

    return BackfillReport(results=tuple(results), dry_run=dry_run)


def _backfill_one_xp(
    xp_folder: Path,
    sig: str,
    *,
    dry_run: bool,
    force: bool,
) -> BackfillResult:
    hydra_dir = xp_folder / ".hydra"
    if not hydra_dir.is_dir():
        return BackfillResult(xp_sig=sig, action="skipped_no_hydra")

    metadata_path = xp_folder / TRAIN_METADATA_FILENAME
    if metadata_path.exists() and not force:
        return BackfillResult(xp_sig=sig, action="skipped_exists", metadata_path=str(metadata_path))

    try:
        overrides = _parse_overrides(hydra_dir / "overrides.yaml")
        config_path = hydra_dir / "config.yaml"
        datasource_train = None
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                datasource = config.get("datasource", {})
                if isinstance(datasource, dict):
                    datasource_train = datasource.get("train")

        metadata = build_backfill_metadata(
            xp_folder=xp_folder,
            overrides=overrides,
            datasource_train=datasource_train,
        )
    except Exception as exc:
        return BackfillResult(xp_sig=sig, action="error", error=str(exc))

    if dry_run:
        return BackfillResult(
            xp_sig=sig,
            action="written",
            metadata_path=str(metadata_path),
            workspace_dir=metadata.get("workspace_dir"),
            label=metadata.get("effective_label"),
            merged_from=_metadata_has_merged_sidecar(metadata),
            lineage_sidecar=_metadata_has_lineage_sidecar(metadata),
        )

    metadata["written_at"] = utc_now_iso()
    metadata["_backfilled"] = True
    write_json(metadata_path, metadata)

    return BackfillResult(
        xp_sig=sig,
        action="written",
        metadata_path=str(metadata_path),
        workspace_dir=metadata.get("workspace_dir"),
        label=metadata.get("effective_label"),
        merged_from=_metadata_has_merged_sidecar(metadata),
        lineage_sidecar=_metadata_has_lineage_sidecar(metadata),
    )


def _metadata_has_merged_sidecar(metadata: dict) -> bool:
    init_metadata = metadata.get("init_from_checkpoint")
    if not isinstance(init_metadata, dict):
        return False
    source_metadata = init_metadata.get("source_resume_metadata")
    return isinstance(source_metadata, dict) and isinstance(source_metadata.get("merged_from"), dict)


def _metadata_has_lineage_sidecar(metadata: dict) -> bool:
    init_metadata = metadata.get("init_from_checkpoint")
    if not isinstance(init_metadata, dict):
        return False
    source_metadata = init_metadata.get("source_resume_metadata")
    return isinstance(source_metadata, dict) and (
        isinstance(source_metadata.get("merged_from"), dict) or isinstance(source_metadata.get("perturbed_from"), dict)
    )


def format_backfill_report(report: BackfillReport) -> str:
    lines: list[str] = [
        f"dry_run: {report.dry_run}",
        f"total_xps: {report.total}",
        f"written: {report.written}",
        f"skipped: {report.skipped}",
        f"errors: {report.errors}",
        "",
    ]
    if report.total == 0:
        lines.append("(no XPs found)")
        return "\n".join(lines)

    header = f"{'sig':<10} {'action':<18} {'label':<35} {'lineage':<8} workspace"
    lines.append(header)
    lines.append("-" * len(header))

    for r in report.results:
        lineage_flag = "yes" if r.lineage_sidecar or r.merged_from else "no"
        label = (r.label or "")[:34]
        ws = (r.workspace_dir or "")[:60]
        if r.action == "error":
            ws = r.error or ""
        lines.append(f"{r.xp_sig:<10} {r.action:<18} {label:<35} {lineage_flag:<8} {ws}")

    return "\n".join(lines)
