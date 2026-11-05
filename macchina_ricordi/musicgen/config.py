from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Values used to build the CLI (``--help``, argument defaults and choices) when the package
# runs without its source checkout. The checkout is the sibling ``musicgen/pyproject.toml``
# that carries the repository's own [tool.macchina_ricordi.*] defaults; a valid checkout
# always overrides these with its configured values.
FALLBACK_DEFAULT_VALID_FRACTION = 0.1
FALLBACK_DEFAULT_SPLIT_SEED = 42
FALLBACK_DEFAULT_RECIPES: tuple[str, ...] = ("reshape-nano", "micro-unconditional")

# Points a non-editable install of the package at its source checkout, so the workspace
# commands can find musicgen/pyproject.toml again (see README, "Install").
REPO_ROOT_ENV = "MACCHINA_RICORDI_ROOT"


class WorkspaceMissingError(RuntimeError):
    """A command needs the repo's source checkout (musicgen/pyproject.toml), which is absent.

    Raised instead of a bare file-not-found so the CLI can print actionable clone/install
    guidance instead of a traceback. Audio and analysis commands never need a checkout and
    do not hit this.
    """


def workspace_guidance(looked_at: Path) -> str:
    return (
        "This command needs the macchina_ricordi source checkout (musicgen/pyproject.toml), "
        "which was not found under:\n"
        f"  {looked_at}\n"
        "\n"
        "Install from a source checkout and run the command inside it, for example:\n"
        "  git clone <your macchina-ricordi repository> macchina-ricordi\n"
        "  cd macchina-ricordi\n"
        "  uv sync --extra dev        # installs the package, its tests and the `ricordi` command\n"
        "  uv run ricordi runtime check   # confirm the AudioCraft/Dora runtime is set up\n"
        "\n"
        "Use `uv run ricordi <command>` (or `python -m macchina_ricordi.cli <command>`) from\n"
        "that checkout. If the package is installed non-editable, set "
        f"{REPO_ROOT_ENV} to its checkout directory so it can find musicgen/pyproject.toml."
    )


@dataclass(frozen=True)
class Recipe:
    """How to train: Dora overrides for the model's shape, codec, loader and schedule. Never which data."""

    name: str
    overrides: tuple[str, ...]
    """Fully composed: the recipes it extends, then its own lines (a later line replaces an earlier one's key)."""
    extends: str | None = None
    description: str | None = None

    def value(self, key: str) -> str | None:
        for override in reversed(self.overrides):
            if override_key(override) == key:
                return override.split("=", 1)[1]
        return None


@dataclass(frozen=True)
class MusicGenProjectConfig:
    repo_root: Path
    project_dir: Path
    env_dir: Path
    dora_dir: Path
    musicgen_python: Path
    audiocraft_dir: Path
    python_request: str
    audiocraft_repo_url: str
    audiocraft_commit_sha: str
    default_recipe: str
    default_valid_fraction: float
    default_split_seed: int
    recipes: Mapping[str, Recipe]
    codecs: Mapping[str, str]

    @property
    def recipe_choices(self) -> tuple[str, ...]:
        return tuple(self.recipes.keys())


def override_key(override: str) -> str:
    """``optim.lr=1e-4`` -> ``optim.lr``; ``model/lm/model_scale=micro`` -> ``model/lm/model_scale``."""
    return override.split("=", 1)[0].lstrip("+~")


def get_repo_root(explicit_root: str | Path | None = None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()
    env_root = os.environ.get(REPO_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    # Editable installs keep this file inside the checkout, so the repo root sits three parents up.
    return Path(__file__).resolve().parents[2]


def find_workspace_root(explicit_root: str | Path | None = None) -> Path | None:
    """The repo root when it is a valid checkout, else ``None``.

    The marker of a checkout is the sibling ``musicgen/pyproject.toml`` that carries
    [tool.macchina_ricordi.*]. This never raises, so CLI construction and audio commands
    can run without a checkout.
    """
    repo_root = get_repo_root(explicit_root)
    if (repo_root / "musicgen" / "pyproject.toml").is_file():
        return repo_root
    return None


def _venv_python_path(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def normalize_python_executable_path(path: str | Path) -> Path:
    """Return an absolute interpreter path without dereferencing virtualenv symlinks."""
    return Path(path).expanduser().absolute()


def load_musicgen_project_config(
    explicit_root: str | Path | None = None,
) -> MusicGenProjectConfig:
    repo_root = get_repo_root(explicit_root)
    project_dir = repo_root / "musicgen"
    pyproject_path = project_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        raise WorkspaceMissingError(workspace_guidance(repo_root))
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    tool_payload = payload.get("tool", {})
    macchina_ricordi_payload = tool_payload.get("macchina_ricordi", {})
    runtime_payload = macchina_ricordi_payload.get("runtime", {})
    audiocraft_payload = macchina_ricordi_payload.get("audiocraft", {})
    defaults_payload = macchina_ricordi_payload.get("defaults", {})
    recipes, codecs = _load_recipes(macchina_ricordi_payload, pyproject_path)

    default_recipe = str(defaults_payload.get("recipe", next(iter(recipes.keys()))))
    if default_recipe not in recipes:
        raise RuntimeError(f"{pyproject_path} default recipe {default_recipe!r} is not a recipe.")

    env_dir = repo_root / runtime_payload.get("project_environment", ".venv-musicgen")
    dora_dir = repo_root / runtime_payload.get("dora_dir", ".dora")
    audiocraft_dir = repo_root / runtime_payload.get("checkout_dir", ".audiocraft")
    python_request = str(runtime_payload.get("python_request", "3.11"))

    return MusicGenProjectConfig(
        repo_root=repo_root,
        project_dir=project_dir,
        env_dir=env_dir,
        dora_dir=dora_dir,
        musicgen_python=_venv_python_path(env_dir),
        audiocraft_dir=audiocraft_dir,
        python_request=python_request,
        audiocraft_repo_url=str(audiocraft_payload["repo_url"]),
        audiocraft_commit_sha=str(audiocraft_payload["commit_sha"]),
        default_recipe=default_recipe,
        default_valid_fraction=float(defaults_payload.get("valid_fraction", 0.1)),
        default_split_seed=int(defaults_payload.get("split_seed", 42)),
        recipes=recipes,
        codecs=codecs,
    )


def _load_recipes(
    macchina_ricordi_payload: Mapping[str, object], path: Path
) -> tuple[dict[str, Recipe], dict[str, str]]:
    raw_recipes = macchina_ricordi_payload.get("recipes")
    if not isinstance(raw_recipes, dict) or not raw_recipes:
        if "profiles" in macchina_ricordi_payload:
            raise RuntimeError(
                f"{path} still uses [tool.macchina_ricordi.profiles]. Profiles became recipes on 2026-10-05: "
                "[tool.macchina_ricordi.recipes.<name>] with `extends` and `overrides`, and no workspace or dataset name. "
                "See docs/musicgen.md."
            )
        raise RuntimeError(f"{path} is missing [tool.macchina_ricordi.recipes] entries.")
    codecs = macchina_ricordi_payload.get("codecs") or {}
    if not isinstance(codecs, dict) or not all(isinstance(v, str) for v in codecs.values()):
        raise RuntimeError(f"{path}: [tool.macchina_ricordi.codecs] must map names to checkpoint strings.")

    composed: dict[str, dict[str, str]] = {}

    def compose(name: str, chain: tuple[str, ...]) -> dict[str, str]:
        if name in composed:
            return composed[name]
        if name in chain:
            raise RuntimeError(f"{path}: recipes extend each other in a loop: {' -> '.join((*chain, name))}")
        raw = raw_recipes.get(name)
        if not isinstance(raw, dict):
            raise RuntimeError(f"{path}: recipe {chain[-1]!r} extends {name!r}, which is not a recipe.")
        lines: dict[str, str] = dict(compose(str(raw["extends"]), (*chain, name))) if raw.get("extends") else {}
        codec = raw.get("codec")
        if codec is not None:
            if codec not in codecs:
                raise RuntimeError(
                    f"{path}: recipe {name!r} uses codec {codec!r}, which is not in [tool.macchina_ricordi.codecs]."
                )
            lines["compression_model_checkpoint"] = f"compression_model_checkpoint={codecs[codec]}"
        overrides = raw.get("overrides", [])
        if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
            raise RuntimeError(f"{path}: recipe {name!r} must define `overrides` as a list of strings.")
        for item in overrides:
            if override_key(item) == "dset":
                raise RuntimeError(
                    f"{path}: recipe {name!r} sets {item!r}. A recipe never names data: the workspace given to "
                    "`train run --workspace-dir` decides it."
                )
            lines[override_key(item)] = item
        composed[name] = lines
        return lines

    recipes: dict[str, Recipe] = {}
    for name, raw in raw_recipes.items():
        if not isinstance(raw, dict):
            raise RuntimeError(f"{path}: [tool.macchina_ricordi.recipes.{name}] must be a table.")
        lines = compose(str(name), ())
        if raw.get("abstract"):
            continue
        recipes[str(name)] = Recipe(
            name=str(name),
            overrides=tuple(lines.values()),
            extends=str(raw["extends"]) if raw.get("extends") else None,
            description=str(raw["description"]) if raw.get("description") else None,
        )
    if not recipes:
        raise RuntimeError(
            f"{path} defines no concrete recipe (every [tool.macchina_ricordi.recipes] entry is abstract)."
        )
    return recipes, {str(k): str(v) for k, v in codecs.items()}


def get_recipe(name: str, *, explicit_root: str | Path | None = None) -> Recipe:
    config = load_musicgen_project_config(explicit_root)
    try:
        return config.recipes[name]
    except KeyError as exc:  # pragma: no cover - argparse validates user input
        raise ValueError(f"Unknown recipe: {name}") from exc


def default_recipe_name(*, explicit_root: str | Path | None = None) -> str:
    if find_workspace_root(explicit_root) is None:
        return FALLBACK_DEFAULT_RECIPES[0]
    return load_musicgen_project_config(explicit_root).default_recipe


def recipe_choices(*, explicit_root: str | Path | None = None) -> tuple[str, ...]:
    if find_workspace_root(explicit_root) is None:
        return FALLBACK_DEFAULT_RECIPES
    return load_musicgen_project_config(explicit_root).recipe_choices


def musicgen_default_valid_fraction(*, explicit_root: str | Path | None = None) -> float:
    if find_workspace_root(explicit_root) is None:
        return FALLBACK_DEFAULT_VALID_FRACTION
    return load_musicgen_project_config(explicit_root).default_valid_fraction


def musicgen_default_split_seed(*, explicit_root: str | Path | None = None) -> int:
    if find_workspace_root(explicit_root) is None:
        return FALLBACK_DEFAULT_SPLIT_SEED
    return load_musicgen_project_config(explicit_root).default_split_seed


def build_musicgen_subprocess_env(
    *,
    explicit_root: str | Path | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    config = load_musicgen_project_config(explicit_root)
    env = dict(os.environ if base_env is None else base_env)
    env["UV_PROJECT_ENVIRONMENT"] = str(config.env_dir)
    env.setdefault("AUDIOCRAFT_DORA_DIR", str(config.dora_dir))
    return env


def musicgen_script_env(*, dora_dir: str | Path | None = None, repo_root: str | Path | None = None) -> dict[str, str]:
    """Env for a helper script run inside the AudioCraft venv: the repo importable, the Dora dir pinned."""
    config = load_musicgen_project_config()
    env = build_musicgen_subprocess_env()
    env["AUDIOCRAFT_DORA_DIR"] = str(dora_dir or config.dora_dir)
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root or config.repo_root), env.get("PYTHONPATH")]))
    return env


def bootstrap_audiocraft_dora_dir(
    *,
    explicit_root: str | Path | None = None,
) -> None:
    if "AUDIOCRAFT_DORA_DIR" in os.environ:
        return
    try:
        config = load_musicgen_project_config(explicit_root)
    except Exception:
        repo_root = get_repo_root(explicit_root)
        os.environ.setdefault("AUDIOCRAFT_DORA_DIR", str((repo_root / ".dora").resolve()))
        return
    os.environ.setdefault("AUDIOCRAFT_DORA_DIR", str(config.dora_dir.resolve()))
