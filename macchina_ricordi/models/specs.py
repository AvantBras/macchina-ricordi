"""Per-model facts the instrument needs while it is being played.

These lived in ``scripts/stream_stats/common.py``, where only the analysis
scripts could see them. The stream needs them too. The trained
``segment_duration`` is two different boundaries at once:

- **Hard.** AudioCraft caps a continuation prompt at ``min(duration,
  max_duration)``, so a prompt that reaches the model's trained segment length
  raises ``AssertionError: Prompt is longer than audio to generate``. Since
  ``tau`` is a live ←/→ control, that abort is reachable from the keyboard on
  the 10 s-segment models. ``clamp_prompt_seconds`` is the only thing here that
  decides anything, and it exists to prevent that crash.
- **Soft.** Overrunning the horizon degrades quality without failing —
  ``73e1a314`` is 97.7 % gap-free at 15 s rounds and 68 % at 17 s. Overrunning
  is sometimes the sound being sought, so this stays advisory.

``clean_share`` and ``top_k_cliff`` carry the other two pieces of tuning advice
the corpus supports, and they are advice: `StreamControl` keeps its
``temperature = 1.0`` default and nothing here turns a knob. Temperature ≥ 1.4
on any model that produces silence at all is the causal half of the project's
standing rule — with the player removed, ``f05eba09`` goes from 39.4% silence at
T=1.0 to 4.1% at T=2.0. The ``top_k = 25`` cliff is sharp, sits exactly there,
and exists in only two of the four models tested.

The per-model numbers live in ``musicgen/models.toml`` (``measured`` and ``study``).

These are historical author measurements on particular models, not a public
benchmark dataset or universal sampling guarantees. The registry retains their summaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Below this, a continuation prompt is close enough to the trained segment length
# that the first generation window would be nearly all prompt echo.
MIN_PROMPT_HEADROOM_SECONDS = 1.0

# Advisory band: warn once the whole window (prompt + completion) is within this
# fraction of the horizon, not only past it — `d371db2c` already degrades as it
# approaches 15 s, its exact trained length.
HORIZON_WARN_FRACTION = 0.9

# The causal half of the standing rule: temperature ≥ 1.4 on any model that
# produces silence at all. Midwife (99.1% of takes clean) is the only model in
# the corpus clean enough to sit above the exemption line.
SILENCE_TEMPERATURE_FLOOR = 1.4
CLEAN_SHARE_EXEMPT = 0.99


@dataclass(frozen=True)
class StudiedModel:
    """Architecture, provenance and measured behaviour of one in-corpus model, as a study recorded it.

    Not the box spec (``appliance.spec.ModelSpec``, the only ``ModelSpec``); this is what the stream reads for advice.

    ``clean_share`` is the share of *candidate* takes with literally zero
    silence — the uncensored view, since best-of-N already selects winners for
    it. ``top_k_cliff`` is the top_k at or below which silence spikes, where
    such a cliff exists; ``None`` means the model was immune or untested.
    """

    label: str
    dim: int
    layers: int
    segment_duration: float
    epochs: int
    warm_started: bool
    clean_share: float | None = None
    top_k_cliff: int | None = None


def _specs_from_registry() -> dict[str, StudiedModel]:
    """The studied models, as recorded in ``musicgen/models.toml`` (title, ``study`` and ``measured``)."""
    from .record import load_registry

    specs: dict[str, StudiedModel] = {}
    for sig, entry in load_registry().items():
        study = entry.study
        if not study:
            continue
        specs[sig] = StudiedModel(
            label=entry.title or sig,
            dim=int(study["dim"]),
            layers=int(study["layers"]),
            segment_duration=float(study["segment_duration"]),
            epochs=int(study["epochs"]),
            warm_started=bool(study["warm_started"]),
            clean_share=entry.measured.get("clean_share"),
            top_k_cliff=entry.measured.get("top_k_cliff"),
        )
    return specs


# ``model_scale=micro`` is 2048d/2L — the original wide-shallow micro. Every other
# entry overrides it towards depth, which is the reshape line of work.
MODEL_SPECS: dict[str, StudiedModel] = _specs_from_registry()


def model_sig(model: str | Path | None) -> str | None:
    """Reduce an xp path (absolute, relative, or already a sig) to its signature."""
    if model in (None, ""):
        return None
    return Path(str(model)).name or None


def spec_for(model: str | Path | None) -> StudiedModel | None:
    sig = model_sig(model)
    return MODEL_SPECS.get(sig) if sig else None


@lru_cache(maxsize=64)
def _segment_duration_from_argv(xp_path: str) -> float | None:
    """``dataset.segment_duration`` off an XP's ``.argv.json``, if it is set there.

    Cached: an experiment folder's training arguments do not change under a
    running session, and this is read on the round loop's path.
    """
    try:
        argv = json.loads((Path(xp_path) / ".argv.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(argv, list):
        return None
    for part in argv:
        if isinstance(part, str) and part.startswith("dataset.segment_duration="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def resolve_segment_duration(model: str | Path | None) -> float | None:
    """The model's trained segment length, or ``None`` when it cannot be known.

    The XP's own ``.argv.json`` wins: it is ground truth and covers models that
    were never in the statistics corpus. `MODEL_SPECS` is the fallback for a sig
    whose folder has moved or gone.
    """
    if model in (None, ""):
        return None
    argv_value = _segment_duration_from_argv(str(model))
    if argv_value is not None and argv_value > 0:
        return argv_value
    spec = spec_for(model)
    return spec.segment_duration if spec else None


def clamp_prompt_seconds(tau_seconds: float, *, segment_duration: float | None) -> float:
    """Longest continuation prompt this model can be asked for, at or below ``tau``.

    Returns ``tau_seconds`` unchanged when it is already safe or the horizon is
    unknown. This is the hard guard: past it AudioCraft aborts the round.
    """
    if segment_duration is None or segment_duration <= 0:
        return float(tau_seconds)
    ceiling = segment_duration - MIN_PROMPT_HEADROOM_SECONDS
    if ceiling <= 0:
        return float(tau_seconds)
    return min(float(tau_seconds), ceiling)


@dataclass(frozen=True)
class Advisory:
    """One line of tuning advice about the current settings. Never acted on."""

    key: str
    level: str
    message: str


def advisories(
    model: str | Path | None,
    *,
    tau_seconds: float | None = None,
    completion_duration_seconds: float | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
) -> tuple[Advisory, ...]:
    """Advisory lines for one model + control combination, in severity order.

    Every entry is a reading of the stream-statistics corpus against the knobs as
    they stand; none of them changes anything. Unknown model or missing knob
    values simply produce fewer lines.
    """
    out: list[Advisory] = []
    sig = model_sig(model)
    horizon = resolve_segment_duration(model)

    if horizon is not None and tau_seconds is not None and completion_duration_seconds is not None:
        window = float(tau_seconds) + float(completion_duration_seconds)
        if window > horizon:
            out.append(
                Advisory(
                    "horizon",
                    "warning",
                    f"prompt+completion {window:.1f}s overruns {sig or 'this model'}'s "
                    f"{horizon:g}s horizon — expect gaps",
                )
            )
        elif window >= horizon * HORIZON_WARN_FRACTION:
            out.append(
                Advisory(
                    "horizon",
                    "info",
                    f"prompt+completion {window:.1f}s is near the {horizon:g}s horizon",
                )
            )

    spec = spec_for(model)
    if (
        temperature is not None
        and spec is not None
        and spec.clean_share is not None
        and spec.clean_share < CLEAN_SHARE_EXEMPT
        and float(temperature) < SILENCE_TEMPERATURE_FLOOR
    ):
        out.append(
            Advisory(
                "temperature",
                "warning",
                f"temperature {float(temperature):.2f} is below the {SILENCE_TEMPERATURE_FLOOR:g} "
                f"floor for a model that gaps ({spec.clean_share:.0%} of takes clean)",
            )
        )

    if top_k is not None and spec is not None and spec.top_k_cliff is not None and 0 < int(top_k) <= spec.top_k_cliff:
        out.append(
            Advisory(
                "top_k",
                "warning",
                f"top_k {int(top_k)} sits on this model's silence cliff (top_k {spec.top_k_cliff}); 50 is already safe",
            )
        )

    return tuple(out)
