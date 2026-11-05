"""Play a box and find out what temperature it needs to stay alive.

These models have a silent attractor, and a *stream* cannot shrug it off the way
one-shot generation can: a rolling context that has gone quiet predicts more
quiet from itself indefinitely. Which models fall in, and at what temperature,
is a property of the weights — so it is measured once, at pack time, with the
runtime that ships, and travels inside the box.

Two things this has to get right, both learned by getting them wrong:

- **The watchdog must be off.** With it on, the probe measures the backstop
  recovering rather than the model failing, and every temperature looks clean.
- **Takes must be long.** The collapse is progressive. On ``nano_disco_60ep`` at
  temperature 1.0, over eight seeds, 12-second takes caught it 1 time in 8,
  30-second takes 4, and 60-second takes 5. A short probe is a probe that passes.
- **Takes must be many.** A model near the line fails rarely, not never. Six
  takes passed ``nano_arcand_20ep`` at 1.2; twelve fresh ones found a take 4%
  quiet at 1.2 and none at 1.4. Six takes will miss a once-in-fifteen failure
  more often than not, so the default is twelve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .engine import Engine
from .loader import load_player
from .player import Knobs

LADDER = (1.0, 1.2, 1.4, 1.6)


def quiet_share(audio: np.ndarray, *, frame: int = 2048, floor_db: float = -40.0) -> float:
    usable = audio[: len(audio) // frame * frame].reshape(-1, frame).astype(np.float64)
    if not len(usable):
        return 0.0
    db = 20 * np.log10(np.sqrt((usable**2).mean(axis=1)) + 1e-9)
    return float((db < floor_db).mean())


def probe_tuning(
    box: Path,
    *,
    takes: int = 12,
    seconds: float = 45.0,
    ladder: tuple[float, ...] = LADDER,
    threshold: float = 0.02,
    threads: int | None = None,
    report=print,
) -> dict:
    measured: dict[str, float] = {}
    recommended = ladder[-1]
    for temperature in ladder:
        worst = 0.0
        for seed in range(takes):
            player, _ = load_player(box, threads=threads, seed=seed)
            player.revive_after = 0  # measure the model, not the watchdog
            audio = Engine(player, knobs=Knobs(temperature=temperature)).render(seconds)
            worst = max(worst, quiet_share(audio))
        measured[f"quiet_at_{temperature:g}"] = round(worst, 4)
        report(f"  probe T={temperature:g}: worst take {worst * 100:5.1f}% quiet")
        if worst < threshold:
            recommended = temperature
            break
    measured["temperature"] = recommended
    measured["probe"] = {"takes": takes, "seconds": seconds, "threshold": threshold}
    return measured
