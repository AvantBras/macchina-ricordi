from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from .config import (
    MusicGenProjectConfig,
    build_musicgen_subprocess_env,
    load_musicgen_project_config,
)
from .shared import dora_candidates, run_checked
from .templates import DATASET_CONFIG_NAME
from .runtime import resolve_audiocraft_dir, resolve_musicgen_python


@dataclass(frozen=True)
class SetupCheck:
    name: str
    status: str
    message: str


@dataclass(frozen=True)
class MusicGenPreflightReport:
    musicgen_python: Path
    audiocraft_dir: Path | None
    checks: tuple[SetupCheck, ...]

    @property
    def is_ready(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "musicgen_python": str(self.musicgen_python),
            "audiocraft_dir": None if self.audiocraft_dir is None else str(self.audiocraft_dir),
            "is_ready": self.is_ready,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "message": check.message,
                }
                for check in self.checks
            ],
        }


@dataclass(frozen=True)
class MusicGenSetupResult:
    musicgen_python: Path
    audiocraft_dir: Path
    audiocraft_commit_sha: str


@dataclass(frozen=True)
class _ResolvedAudiocraftCheckout:
    path: Path | None
    source: str
    error: str | None = None


@dataclass(frozen=True)
class _GitStatusEntry:
    index_status: str
    worktree_status: str
    path: str


def _probe_musicgen_python(musicgen_python: Path) -> dict[str, object]:
    script = """
import importlib.util
import json
import sys

payload = {
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "modules": {},
}
for name in ("audiocraft", "dora"):
    spec = importlib.util.find_spec(name)
    module_payload = {"available": spec is not None}
    if spec is not None:
        module_payload["origin"] = spec.origin
        if spec.submodule_search_locations:
            module_payload["locations"] = list(spec.submodule_search_locations)
    payload["modules"][name] = module_payload
print(json.dumps(payload))
""".strip()
    completed = subprocess.run(
        [str(musicgen_python), "-c", script],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def _resolve_audiocraft_checkout(explicit_dir: str | None) -> _ResolvedAudiocraftCheckout:
    source = "explicit" if explicit_dir else "default"
    candidate_path = (
        Path(explicit_dir).expanduser().resolve()
        if explicit_dir
        else load_musicgen_project_config().audiocraft_dir.resolve()
    )
    try:
        return _ResolvedAudiocraftCheckout(
            path=resolve_audiocraft_dir(explicit_dir),
            source=source,
        )
    except (ValueError, FileNotFoundError) as exc:
        return _ResolvedAudiocraftCheckout(
            path=candidate_path,
            source=source,
            error=str(exc),
        )
def _parse_git_status_porcelain(output: str) -> tuple[_GitStatusEntry, ...]:
    entries: list[_GitStatusEntry] = []
    for line in output.splitlines():
        if not line:
            continue
        if len(line) < 4:
            raise RuntimeError(f"Unexpected git status output: {line!r}")
        entries.append(
            _GitStatusEntry(
                index_status=line[0],
                worktree_status=line[1],
                path=line[3:],
            )
        )
    return tuple(entries)


def _allowed_generated_dataset_config_paths() -> set[str]:
    return {f"config/dset/audio/{DATASET_CONFIG_NAME}.yaml"}


def _is_generated_dataset_config_path(path: str) -> bool:
    # Profiles come and go (a removed profile leaves its yaml behind), and every
    # dataset config this project writes is named macchina_ricordi_*.
    if path in _allowed_generated_dataset_config_paths():
        return True
    name = path.removeprefix("config/dset/audio/")
    return name != path and "/" not in name and name.startswith("macchina_ricordi_") and name.endswith(".yaml")


def _is_allowed_generated_dataset_change(entry: _GitStatusEntry) -> bool:
    if not _is_generated_dataset_config_path(entry.path):
        return False
    if (entry.index_status, entry.worktree_status) == ("?", "?"):
        return True
    return entry.index_status == " " and entry.worktree_status == "M"


def _files_covered_by_patches(repo_root: Path) -> set[str]:
    patch_dir = repo_root / "patches"
    covered: set[str] = set()
    if not patch_dir.exists():
        return covered
    for patch_file in sorted(patch_dir.glob("*.patch")):
        for line in patch_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("diff --git a/"):
                parts = line.split()
                if len(parts) >= 3:
                    covered.add(parts[2][2:])  # strip "a/" prefix
    return covered


def _ensure_dataset_configs(checkout_dir: Path, config: MusicGenProjectConfig) -> None:
    """Create any missing dset/audio/*.yaml configs that Hydra/Dora needs for config resolution.

    These are generated files — the setup system tolerates them as dirty.  We create
    stubs here so that ``runtime setup`` leaves the checkout in a
    generation-ready state even when a workspace hasn't been prepared yet.
    """
    target = checkout_dir / "config" / "dset" / "audio" / f"{DATASET_CONFIG_NAME}.yaml"
    if not target.exists():
        # A placeholder that names no real workspace: every launch and resume writes the real paths first.
        egs_base = config.repo_root / ".macchina_ricordi_dataset_unset" / "egs" / DATASET_CONFIG_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "# @package __global__\n"
            "# Auto-generated by macchina_ricordi runtime setup.\n"
            "# Edits to the datasource paths are expected and safe.\n"
            "\n"
            "datasource:\n"
            "  max_sample_rate: 32000\n"
            "  max_channels: 1\n"
            "\n"
            f"  train: {egs_base / 'train'}\n"
            f"  valid: {egs_base / 'valid'}\n"
            f"  evaluate: {egs_base / 'valid'}\n"
            f"  generate: {egs_base / 'valid'}\n",
            encoding="utf-8",
        )


def _apply_repo_patches(checkout_dir: Path, repo_root: Path) -> None:
    patch_dir = repo_root / "patches"
    if not patch_dir.exists():
        return
    for patch_file in sorted(patch_dir.glob("*.patch")):
        run_checked(["git", "apply", str(patch_file)], cwd=checkout_dir)


def _ensure_repo_managed_audiocraft_checkout(config: MusicGenProjectConfig) -> Path:
    checkout_dir = config.audiocraft_dir
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)

    if not checkout_dir.exists():
        run_checked(
            ["git", "clone", config.audiocraft_repo_url, str(checkout_dir)],
            cwd=config.repo_root,
        )
    elif not (checkout_dir / ".git").exists():
        raise RuntimeError(
            f"Repo-managed AudioCraft path exists but is not a git checkout: {checkout_dir}"
        )

    patch_covered = _files_covered_by_patches(config.repo_root)
    status = run_checked(
        ["git", "status", "--porcelain"],
        cwd=checkout_dir,
        capture_output=True,
    )
    dirty_entries = _parse_git_status_porcelain(status.stdout)
    blocking_entries = [
        entry.path for entry in dirty_entries
        if not _is_allowed_generated_dataset_change(entry)
        and entry.path not in patch_covered
    ]
    if blocking_entries:
        raise RuntimeError(
            f"Repo-managed AudioCraft checkout has blocking local changes: {checkout_dir}. "
            f"Blocking paths: {', '.join(sorted(blocking_entries))}. "
            "Commit or discard those changes before rerunning setup."
        )

    run_checked(["git", "fetch", "--force", "origin"], cwd=checkout_dir)
    run_checked(
        ["git", "checkout", "--force", "--detach", config.audiocraft_commit_sha],
        cwd=checkout_dir,
    )
    run_checked(["git", "clean", "-fd"], cwd=checkout_dir)
    _apply_repo_patches(checkout_dir, config.repo_root)
    _ensure_dataset_configs(checkout_dir, config)
    return checkout_dir


def setup_repo_managed_musicgen_runtime(
    *,
    explicit_root: str | Path | None = None,
) -> MusicGenSetupResult:
    config = load_musicgen_project_config(explicit_root)
    env = build_musicgen_subprocess_env(explicit_root=explicit_root)

    run_checked(
        [
            "uv",
            "sync",
            "--project",
            str(config.project_dir),
            "--python",
            config.python_request,
            "--locked",
        ],
        cwd=config.repo_root,
        env=env,
    )
    checkout_dir = _ensure_repo_managed_audiocraft_checkout(config)
    run_checked(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(config.musicgen_python),
            "--no-deps",
            "-e",
            str(checkout_dir),
        ],
        cwd=config.repo_root,
        env=env,
    )

    return MusicGenSetupResult(
        musicgen_python=config.musicgen_python,
        audiocraft_dir=checkout_dir,
        audiocraft_commit_sha=config.audiocraft_commit_sha,
    )


def run_musicgen_preflight(
    *,
    explicit_python: str | None = None,
    explicit_audiocraft_dir: str | None = None,
) -> MusicGenPreflightReport:
    checks: list[SetupCheck] = []
    musicgen_python = Path(resolve_musicgen_python(explicit_python)).expanduser()
    if musicgen_python.exists() and musicgen_python.is_file():
        checks.append(
            SetupCheck(
                name="musicgen_python",
                status="ok",
                message=f"Using Python executable at {musicgen_python}",
            )
        )
    else:
        checks.append(
            SetupCheck(
                name="musicgen_python",
                status="fail",
                message=(
                    f"Python executable not found: {musicgen_python}. "
                    "Run `make setup-musicgen` or set --musicgen-python."
                ),
            )
        )
        return MusicGenPreflightReport(
            musicgen_python=musicgen_python,
            audiocraft_dir=None,
            checks=tuple(checks),
        )

    dora_path = next((candidate for candidate in dora_candidates(musicgen_python) if candidate.exists()), None)
    if dora_path is not None:
        checks.append(
            SetupCheck(
                name="dora_cli",
                status="ok",
                message=f"Found Dora launcher at {dora_path}",
            )
        )
    else:
        checks.append(
            SetupCheck(
                name="dora_cli",
                status="fail",
                message=(
                    "Could not find a dora launcher next to the selected Python. "
                    "Run `make setup-musicgen` to rebuild the runtime."
                ),
            )
        )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        checks.append(
            SetupCheck(
                name="ffmpeg_cli",
                status="ok",
                message=f"Found ffmpeg at {ffmpeg_path}",
            )
        )
    else:
        checks.append(
            SetupCheck(
                name="ffmpeg_cli",
                status="warn",
                message=(
                    "ffmpeg was not found on PATH. AudioCraft training can still start, "
                    "but sample generation/export may fail at generate stages or final epochs."
                ),
            )
        )

    try:
        probe = _probe_musicgen_python(musicgen_python)
    except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        checks.append(
            SetupCheck(
                name="musicgen_probe",
                status="fail",
                message=f"Failed to inspect the musicgen environment: {exc}",
            )
        )
        return MusicGenPreflightReport(
            musicgen_python=musicgen_python,
            audiocraft_dir=None,
            checks=tuple(checks),
        )

    checks.append(
        SetupCheck(
            name="musicgen_probe",
            status="ok",
            message=(
                f"Python {probe['python_version']} reports executable "
                f"{probe['python_executable']}"
            ),
        )
    )

    modules = probe.get("modules", {})
    for module_name in ("audiocraft", "dora"):
        module_payload = modules.get(module_name, {})
        if module_payload.get("available"):
            origin = module_payload.get("origin") or module_payload.get("locations", ["<unknown>"])[0]
            checks.append(
                SetupCheck(
                    name=f"{module_name}_import",
                    status="ok",
                    message=f"{module_name} import is available from {origin}",
                )
            )
        else:
            checks.append(
                SetupCheck(
                    name=f"{module_name}_import",
                    status="fail",
                    message=f"{module_name} is not importable from {musicgen_python}",
                )
            )

    resolved_audiocraft = _resolve_audiocraft_checkout(explicit_audiocraft_dir)
    audiocraft_dir = resolved_audiocraft.path if resolved_audiocraft.error is None else None
    if audiocraft_dir is None:
        if resolved_audiocraft.source == "explicit":
            message = (
                "AudioCraft checkout validation failed for "
                f"--audiocraft-dir {resolved_audiocraft.path}: {resolved_audiocraft.error}"
            )
        else:
            message = (
                f"{resolved_audiocraft.error} "
                "Run `make setup-musicgen` or pass --audiocraft-dir."
            )
        checks.append(
            SetupCheck(
                name="audiocraft_checkout",
                status="fail",
                message=message,
            )
        )
        return MusicGenPreflightReport(
            musicgen_python=musicgen_python,
            audiocraft_dir=None,
            checks=tuple(checks),
        )

    checks.append(
        SetupCheck(
            name="audiocraft_checkout",
            status="ok",
            message=f"Using AudioCraft checkout at {audiocraft_dir}",
        )
    )

    audiocraft_module = modules.get("audiocraft", {})
    origin = audiocraft_module.get("origin")
    if origin:
        origin_path = Path(str(origin)).resolve()
        if origin_path.is_relative_to(audiocraft_dir.resolve()):
            checks.append(
                SetupCheck(
                    name="audiocraft_checkout_match",
                    status="ok",
                    message="The importable audiocraft package matches the configured checkout.",
                )
            )
        else:
            checks.append(
                SetupCheck(
                    name="audiocraft_checkout_match",
                    status="warn",
                    message=(
                        "The importable audiocraft package does not come from the configured "
                        f"checkout ({origin_path}). Run `make setup-musicgen` to reinstall the "
                        "repo-managed checkout."
                    ),
                )
            )

    return MusicGenPreflightReport(
        musicgen_python=musicgen_python,
        audiocraft_dir=audiocraft_dir,
        checks=tuple(checks),
    )
