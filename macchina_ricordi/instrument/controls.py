"""What the controls are called, what they mean, and how far they go: one vocabulary for the stream and the appliance.

The stream (``play stream``, its TUI and its monitor) and the appliance's player and panel are two instruments over
the same gestures. A knob turned on one should mean the same thing on the other, so its name, range and default
live here once. Both read this module; it imports nothing outside the standard library, so the appliance, which
ships without torch or AudioCraft, carries it too.

Each control has:

- ``key``: the appliance's name (``Knobs`` field, panel slider id); ``stream``: the ``StreamControl`` field;
- ``limits``: what any player accepts. ``None`` as an upper limit means "the model decides" (``tau`` stops at the
  model's trained horizon on the stream);
- ``appliance``: a narrower range the appliance keeps to, where it has one, and why;
- ``panel``: the range and step of the appliance panel's slider; the stream's TUI keeps its own keyboard steps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Control:
    key: str
    stream: str
    label: str
    flag: str
    unit: str
    default: float | None
    limits: tuple[float, float | None]
    panel: tuple[float, float, float]
    """``(min, max, step)`` of the panel's slider."""
    appliance: tuple[float, float] | None = None
    why_narrower: str = ""
    help: str = ""

    def clamp(self, value: float, *, appliance: bool = False, ceiling: float | None = None) -> float:
        """``value`` within this control's limits (the appliance's own, with ``appliance=True``)."""
        low, high = self.appliance if appliance and self.appliance else self.limits
        if ceiling is not None:
            high = ceiling if high is None else min(high, ceiling)
        value = max(low, float(value))
        return value if high is None else min(high, value)

    def range_text(self, *, appliance: bool = False) -> str:
        low, high = self.appliance if appliance and self.appliance else self.limits
        return f"{low:g}–{high:g}" if high is not None else f"from {low:g}"


@dataclass(frozen=True)
class Gesture:
    key: str
    stream: str
    """The ``StreamControl.command`` that performs it."""
    label: str
    weight: float | None = None
    """How much of the pinned moment is blended into the past, when the gesture blends one."""


TEMPERATURE = Control(
    key="temperature",
    stream="temperature",
    label="Wildness",
    flag="--temperature",
    unit="",
    default=1.0,
    limits=(0.05, 3.0),
    panel=(0.3, 2.0, 0.05),
    help="Low is careful and can fall silent.",
)
TOP_K = Control(
    key="top_k",
    stream="top_k",
    label="Variety",
    flag="--top-k",
    unit="",
    default=250,
    limits=(0, 2048),
    panel=(10, 1000, 10),
    help="How many options it considers each step (0 = all).",
)
TAU = Control(
    key="tau_seconds",
    stream="tau_seconds",
    label="Memory",
    flag="--tau",
    unit="s",
    default=5.0,
    limits=(0.5, None),
    panel=(0.5, 5.0, 0.5),
    appliance=(0.5, 5.0),
    why_narrower="a box's context window is five seconds",
    help="Seconds of what just played the model re-reads.",
)
ROUND = Control(
    key="round_seconds",
    stream="completion_duration_seconds",
    label="Stride",
    flag="--round-seconds",
    unit="s",
    default=None,
    limits=(1.0, 60.0),
    panel=(1.0, 10.0, 0.5),
    appliance=(1.0, 10.0),
    why_narrower="a CPU has to keep up in real time",
    help="New seconds each round adds. Longer answers the controls more slowly.",
)
TILT_LOW = Control(
    key="tilt_low_db",
    stream="prompt_tilt_low_db",
    label="Low",
    flag="--tilt-low",
    unit="dB",
    default=0.0,
    limits=(-12.0, 12.0),
    panel=(-12.0, 12.0, 1.5),
    help="dB on the past's lows (below 250 Hz).",
)
TILT_MID = Control(
    key="tilt_mid_db",
    stream="prompt_tilt_mid_db",
    label="Mid",
    flag="--tilt-mid",
    unit="dB",
    default=0.0,
    limits=(-12.0, 12.0),
    panel=(-12.0, 12.0, 1.5),
    help="dB on the past's mids.",
)
TILT_HIGH = Control(
    key="tilt_high_db",
    stream="prompt_tilt_high_db",
    label="High",
    flag="--tilt-high",
    unit="dB",
    default=0.0,
    limits=(-12.0, 12.0),
    panel=(-12.0, 12.0, 1.5),
    help="dB on the past's highs (above 4 kHz).",
)
TAPE_SPEED = Control(
    key="prompt_rate",
    stream="prompt_rate",
    label="Tape speed",
    flag="--tape-speed",
    unit="×",
    default=1.0,
    limits=(0.5, 2.0),
    panel=(0.9, 1.1, 0.01),
    appliance=(0.9, 1.1),
    why_narrower="past about ±15% EnCodec's round trip adds artefacts of its own",
    help="The past played back to the model slower and lower, or faster and higher.",
)

CONTROLS: tuple[Control, ...] = (TEMPERATURE, TOP_K, TAU, ROUND, TILT_LOW, TILT_MID, TILT_HIGH, TAPE_SPEED)
STEERING: tuple[Control, ...] = (TILT_LOW, TILT_MID, TILT_HIGH, TAPE_SPEED)
"""The controls that change what the model believes its past was; "Back to neutral" resets exactly these."""

PIN = Gesture("pin", "pin", "Pin this moment")
BRING_BACK = Gesture("inject", "inject", "Bring it back", weight=0.5)
BRING_BACK_LIGHTLY = Gesture("light", "inject_light", "Bring it back, lightly", weight=0.2)
BRING_BACK_STRONGLY = Gesture("heavy", "inject_heavy", "Bring it back, strongly", weight=0.6)
RECALL_MIX = 0.5
"""How much of a pinned moment a periodic recall (``--recall-every``, the stream's ``motif_period``) blends in."""
GESTURES: tuple[Gesture, ...] = (PIN, BRING_BACK, BRING_BACK_LIGHTLY, BRING_BACK_STRONGLY)

# What "Low", "Mid" and "High" mean: crossovers one octave wide, smooth, so a band edge does not ring.
TILT_LOW_CROSSOVER_HZ = 250.0
TILT_HIGH_CROSSOVER_HZ = 4000.0
TILT_TRANSITION_OCTAVES = 1.0
MAX_TILT_DB = TILT_LOW.limits[1]

_BY_KEY = {control.key: control for control in CONTROLS}
_BY_STREAM = {control.stream: control for control in CONTROLS}


def control(key: str) -> Control:
    """A control by its appliance key or its stream field."""
    found = _BY_KEY.get(key) or _BY_STREAM.get(key)
    if found is None:
        raise KeyError(f"no control called {key!r}; known: {', '.join(_BY_KEY)}")
    return found


def neutral() -> dict[str, float]:
    """The steering controls at rest, by appliance key."""
    return {c.key: float(c.default) for c in STEERING if c.default is not None}
