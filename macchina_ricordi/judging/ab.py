"""Blind A/B listening sessions — the ground truth every metric is fitted against.

The session shows nothing but "A" and "B". Which model, which temperature, which sweep
and which side is which are all hidden until the verdict is recorded, because the point
is to collect judgements that are not contaminated by knowing what should win.

Pair selection prefers *cross-group* pairs (different model / profile / sweep), since a
pair drawn from one model tells you about sampling noise while a cross-model pair tells
you which model to keep. Already-judged pairs are skipped so an evening adds new
information instead of re-litigating.

Clips come from sweep ``index.csv`` files or, since the stream is where most listening
now happens, straight from a stream session's ``performance.jsonl``. The session
directory is read, never written: it is a read-only artifact by the monitor convention,
so the grouping and the metrics are assembled in memory.

Judgements are scarce, so pairs can be *targeted*: ``target_metric`` prefers pairs the
metric strongly separates, and ``disagree`` prefers pairs where two metrics rank the
sides oppositely. A targeted pair is not an isolated one — high contrast on one axis
usually means contrast on others — so targeting buys informativeness, not proof.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..sessions.takes import GROUP_BY_CONDITION, PlayableTake, knob_text, playable_takes
from .preference import PreferenceRecord, pair_key
from .questions import PAIRWISE, PAIRWISE_CONTEXT, Question, get_question

NOVELTY_MEMORY_ROUNDS = 8
CONTEXT_MIN_SHARED_ROUNDS = 5
CONTEXT_TARGET_SECONDS = 21.0
CONTEXT_PROTOCOL = "shared_memory.v1"

_VERDICT_ALIASES = {
    "a": "a",
    "1": "a",
    "left": "a",
    "b": "b",
    "2": "b",
    "right": "b",
    "t": "tie",
    "tie": "tie",
    "=": "tie",
    "r": "replay",
    "replay": "replay",
    "c": "context",
    "context": "context",
    "s": "skip",
    "skip": "skip",
    "q": "quit",
    "quit": "quit",
}


@dataclass(frozen=True)
class AbCandidate:
    """One playable clip. ``session_id`` / ``round_idx`` are carried for stream rounds:
    the shared-context protocol needs to know which session a clip came from and where in
    it, and they cost nothing to keep for sweep clips that have neither."""

    path: str
    group: str
    metrics: dict[str, float] = field(default_factory=dict)
    label: str = ""
    session_id: str = ""
    round_idx: int | None = None
    session_position: int | None = None
    prompt_duration_seconds: float = 0.0
    memory_paths: tuple[str, ...] = ()
    memory_prompt_durations_seconds: tuple[float, ...] = ()
    playback_path: str = field(default="", compare=False, repr=False)
    controls: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def playable_path(self) -> str:
        return self.playback_path or self.path


@dataclass(frozen=True)
class AbPair:
    pair_id: str
    first: AbCandidate
    second: AbCandidate
    context_paths: tuple[str, ...] = ()
    context_prompt_durations_seconds: tuple[float, ...] = field(default=(), repr=False)
    context_excerpt_seconds: float | None = None
    context_reel_path: str = field(default="", compare=False, repr=False)

    @property
    def key(self) -> str:
        return pair_key(self.first.path, self.second.path)


def parse_verdict(raw: str) -> str | None:
    return _VERDICT_ALIASES.get(raw.strip().lower())


# ── candidate sourcing ────────────────────────────────────────────────


def _candidate(take: PlayableTake) -> AbCandidate:
    return AbCandidate(
        path=take.path,
        group=take.group,
        metrics=dict(take.metrics),
        label=take.label,
        session_id=take.session_id if take.kind != "sweep" else "",
        round_idx=take.round_idx,
        session_position=take.position,
        prompt_duration_seconds=take.prompt_duration_seconds,
        # the model as one more knob, so a model session pairs and tallies it as a parameter session does a knob
        controls={**take.controls, **({"model": take.model_name} if take.model_name else {})},
    )


def load_sweep_candidates(
    sweep_dirs: Iterable[str | Path],
    *,
    group_column: str = "sweep_id",
) -> list[AbCandidate]:
    """A sweep's takes as A/B candidates, grouped by an ``index.csv`` column."""
    takes = playable_takes(sweep_dirs, group_column=group_column)
    return [_candidate(take) for take in takes if take.kind == "sweep"]


GROUP_BY_MODEL = "model_xp"
GROUP_BY_SESSION = "session"
GROUP_BY_BUCKET_PREFIX = "bucket:"
# What a session tests. ``metric``: both sides from one condition (``sessions.takes.GROUP_BY_CONDITION``), so a
# verdict is about what the metric measures, not about which model or knobs made the clip. ``parameter:<knob>``:
# one condition but for that knob, which differs. ``model``: the same, with the model as the knob. ``groups``: across
# groups.
TESTING = ("metric", "model", "groups")
PARAMETER_PREFIX = "parameter:"


def parse_testing(value: str) -> str:
    """``metric``, ``model``, ``groups`` or ``parameter:<knob>``; raises ``ValueError`` on anything else."""
    if value in TESTING or (value.startswith(PARAMETER_PREFIX) and value[len(PARAMETER_PREFIX) :].strip()):
        return value
    raise ValueError(f"testing must be metric, model, groups or parameter:<knob>, got {value!r}")


def varied_knob(testing: str) -> str | None:
    if testing == "model":
        return "model"
    return testing[len(PARAMETER_PREFIX) :].strip() if testing.startswith(PARAMETER_PREFIX) else None


def testing_group_by(testing: str, group_by: str) -> str:
    """What candidates are grouped by for a session testing ``testing``; ``group_by`` serves ``groups`` only."""
    if testing == "metric":
        return GROUP_BY_CONDITION
    knob = varied_knob(testing)
    return f"{GROUP_BY_CONDITION}:{knob}" if knob else group_by


def matchups(judged: Iterable[tuple[AbPair, str]], knob: str) -> list[dict[str, Any]]:
    """Wins per pair of ``knob`` values, lower value first: "temperature 1 vs 1.3: 2–8"."""
    table: dict[tuple[str, str], list[int]] = {}
    for pair, outcome in judged:
        if outcome not in ("a", "b", "tie"):
            continue
        sides = sorted(
            ((pair.first.controls.get(knob), "a"), (pair.second.controls.get(knob), "b")),
            key=lambda side: (0, side[0], "") if isinstance(side[0], int | float) else (1, 0, str(side[0])),
        )
        (low, low_side), (high, _) = sides
        tally = table.setdefault((knob_text(low), knob_text(high)), [0, 0, 0])
        tally[1 if outcome == "tie" else 0 if outcome == low_side else 2] += 1
    return [
        {"low": low, "high": high, "low_wins": tally[0], "ties": tally[1], "high_wins": tally[2]}
        for (low, high), tally in table.items()
    ]


def _tercile_buckets(
    candidates: Sequence[AbCandidate],
    metric: str,
) -> dict[str, str]:
    """Label each clip low/high on ``metric``, with terciles taken *within each session*.

    Per-session terciles because a stream drifts: a round that is high-novelty for its
    own evening can sit below another evening's median, and pooling the two would build
    contrast pairs out of session differences. The middle tercile is dropped, so any
    cross-group pair is a genuine high-vs-low contrast.
    """
    by_session: dict[str, list[AbCandidate]] = {}
    for candidate in candidates:
        if metric in candidate.metrics:
            by_session.setdefault(candidate.session_id, []).append(candidate)

    labels: dict[str, str] = {}
    for members in by_session.values():
        ordered = sorted(members, key=lambda item: item.metrics[metric])
        if len(ordered) < 3:
            continue
        cut = len(ordered) // 3
        for candidate in ordered[:cut]:
            labels[candidate.path] = f"{GROUP_BY_BUCKET_PREFIX}{metric}:low"
        for candidate in ordered[len(ordered) - cut :]:
            labels[candidate.path] = f"{GROUP_BY_BUCKET_PREFIX}{metric}:high"
    return labels


def _bucketed(candidates: list[AbCandidate], group_by: str) -> list[AbCandidate]:
    if not group_by.startswith(GROUP_BY_BUCKET_PREFIX):
        return candidates
    labels = _tercile_buckets(candidates, group_by[len(GROUP_BY_BUCKET_PREFIX) :])
    return [replace(candidate, group=labels[candidate.path]) for candidate in candidates if candidate.path in labels]


def load_stream_candidates(
    session_dirs: Iterable[str | Path],
    *,
    group_by: str = GROUP_BY_MODEL,
    scored_only: bool = True,
) -> list[AbCandidate]:
    """Stream rounds as A/B candidates, each carrying the rounds it heard before it.

    Metrics are the round's ``score`` payload as logged — no re-scoring, which is what
    makes ``novelty_raw`` / ``novelty_whitened`` testable at all, since they are
    context-dependent and cannot be recomputed from the wav alone. For the full offline
    suite (``dev_*``, ``ssm_*``) run ``score rescore`` over ``rounds/`` and hand the CSV
    to ``judge prefs``.

    ``group_by`` decides what a cross-group pair means: ``model_xp`` compares models,
    ``session`` compares evenings, and ``bucket:METRIC`` compares the top and bottom
    tercile of one metric. ``scored_only=False`` keeps rounds logged without a score, for re-scoring.
    """
    return _stream_candidates(playable_takes(session_dirs, group_by=group_by), group_by, scored_only)


def _stream_candidates(takes: Iterable[PlayableTake], group_by: str, scored_only: bool) -> list[AbCandidate]:
    candidates: list[AbCandidate] = []
    memory: dict[str, deque[tuple[str, float]]] = {}
    for take in takes:
        if take.kind != "stream":
            continue
        heard = memory.setdefault(take.session_id, deque(maxlen=NOVELTY_MEMORY_ROUNDS))
        previous = tuple(heard)
        heard.append((take.path, take.prompt_duration_seconds))
        if scored_only and not take.metrics:
            continue
        candidates.append(
            replace(
                _candidate(take),
                memory_paths=tuple(item[0] for item in previous),
                memory_prompt_durations_seconds=tuple(item[1] for item in previous),
            )
        )
    return _bucketed(candidates, group_by)


def load_candidates(
    sources: Iterable[str | Path],
    *,
    group_column: str = "sweep_id",
    group_by: str = GROUP_BY_MODEL,
    scored_only: bool = True,
) -> list[AbCandidate]:
    """Load whatever each source is: a stream session, a sweep, or any other session, by folder or id."""
    takes = playable_takes(sources, group_column=group_column, group_by=group_by)
    candidates = [_candidate(take) for take in takes if take.kind == "sweep"]
    candidates += _stream_candidates(takes, group_by, scored_only)
    candidates += _bucketed([_candidate(take) for take in takes if take.kind not in ("stream", "sweep")], group_by)
    return candidates


# ── pair targeting ────────────────────────────────────────────────────


def _metric_scale(candidates: Sequence[AbCandidate], metric: str) -> float:
    """Spread of a metric over the pool, so margins on different metrics are comparable."""
    values = [candidate.metrics[metric] for candidate in candidates if metric in candidate.metrics]
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    return std if std > 1e-9 else 1.0


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _target_metric_pairs(
    possible: Sequence[tuple[AbCandidate, AbCandidate]],
    metric: str,
    *,
    scale: float,
    quantile: float,
) -> list[tuple[AbCandidate, AbCandidate]]:
    scored: list[tuple[float, tuple[AbCandidate, AbCandidate]]] = []
    for first, second in possible:
        value_a = first.metrics.get(metric)
        value_b = second.metrics.get(metric)
        if value_a is None or value_b is None:
            continue
        scored.append((abs(value_a - value_b) / scale, (first, second)))
    if not scored:
        return []
    threshold = _quantile([margin for margin, _ in scored], quantile)
    return [pair for margin, pair in scored if margin >= threshold]


def _disagree_pairs(
    possible: Sequence[tuple[AbCandidate, AbCandidate]],
    metrics: Sequence[str],
    *,
    scales: Sequence[float],
    quantile: float,
) -> list[tuple[AbCandidate, AbCandidate]]:
    """Pairs the two metrics rank oppositely, ranked by the weaker of the two margins.

    Ranking on the *minimum* margin is the point: a pair where one metric is emphatic and
    the other nearly tied is a pair the near-tied metric has no opinion about, so it
    settles nothing.
    """
    first_metric, second_metric = metrics
    first_scale, second_scale = scales
    scored: list[tuple[float, tuple[AbCandidate, AbCandidate]]] = []
    for left, right in possible:
        left_one, right_one = left.metrics.get(first_metric), right.metrics.get(first_metric)
        left_two, right_two = left.metrics.get(second_metric), right.metrics.get(second_metric)
        if None in (left_one, right_one, left_two, right_two):
            continue
        delta_one = (left_one - right_one) / first_scale
        delta_two = (left_two - right_two) / second_scale
        if delta_one == 0.0 or delta_two == 0.0 or (delta_one > 0) == (delta_two > 0):
            continue
        scored.append((min(abs(delta_one), abs(delta_two)), (left, right)))
    if not scored:
        return []
    threshold = _quantile([margin for margin, _ in scored], quantile)
    return [pair for margin, pair in scored if margin >= threshold]


def _shared_memory(
    first: AbCandidate,
    second: AbCandidate,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return the exact memory intersection in stream-log chronology.

    A multiset intersection preserves distinct repeated rows if a reroll happens to point
    at the same wav path more than once.
    """
    remaining = Counter(second.memory_paths)
    shared: list[str] = []
    prompts: list[float] = []
    first_prompts = first.memory_prompt_durations_seconds
    for index, path in enumerate(first.memory_paths):
        if remaining[path] > 0:
            shared.append(path)
            prompts.append(first_prompts[index] if index < len(first_prompts) else 0.0)
            remaining[path] -= 1
    return tuple(shared), tuple(prompts)


def select_pairs(
    candidates: Sequence[AbCandidate],
    *,
    count: int,
    seed: int = 0,
    cross_group: bool = True,
    exclude_keys: Iterable[str] = (),
    target_metric: str | None = None,
    disagree: Sequence[str] | None = None,
    target_quantile: float = 0.5,
    context_window: int | None = None,
    allow_same_group_fallback: bool | None = None,
    within_group: bool = False,
    vary: str | None = None,
) -> list[AbPair]:
    """Draw distinct, unjudged pairs; sides are shuffled so the session stays blind.

    ``within_group`` keeps only pairs whose sides share a group, the opposite of ``cross_group``. ``vary`` also
    keeps only pairs whose sides both set that control, to different values.

    ``target_metric`` keeps only the pairs in the top ``target_quantile`` of |ΔM|, so no
    judgement is spent on a pair the metric barely distinguishes. ``disagree`` keeps the
    pairs where two metrics point opposite ways — the maximally informative ones when the
    open question is which of two correlated metrics the ear actually follows.

    Targeting never falls back to untargeted pairs: an empty result means the pool has no
    informative pairs left, which the caller should say out loud rather than paper over
    with pairs that answer a different question.
    """
    rng = random.Random(seed)
    if context_window is not None and context_window not in (1, 2, 3):
        raise ValueError(f"context_window must be 1, 2, or 3, got {context_window}")
    if allow_same_group_fallback is None:
        allow_same_group_fallback = context_window is None
    excluded = set(exclude_keys)
    pool = list(candidates)
    if len(pool) < 2:
        return []

    possible: list[tuple[AbCandidate, AbCandidate]] = []
    shared_context: dict[str, tuple[str, ...]] = {}
    shared_prompts: dict[str, tuple[float, ...]] = {}
    for index, first in enumerate(pool):
        for second in pool[index + 1 :]:
            if cross_group and first.group == second.group:
                continue
            if within_group and first.group != second.group:
                continue
            if vary and (
                vary not in first.controls
                or vary not in second.controls
                or first.controls[vary] == second.controls[vary]
            ):
                continue
            key = pair_key(first.path, second.path)
            if key in excluded:
                continue
            if context_window is not None:
                if (
                    not first.session_id
                    or first.session_id != second.session_id
                    or first.session_position is None
                    or second.session_position is None
                ):
                    continue
                gap = abs(first.session_position - second.session_position)
                if gap < 1 or gap > context_window:
                    continue
                shared, prompts = _shared_memory(first, second)
                if len(shared) < CONTEXT_MIN_SHARED_ROUNDS:
                    continue
                shared_context[key] = shared
                shared_prompts[key] = prompts
            possible.append((first, second))

    if not possible and cross_group and allow_same_group_fallback:
        return select_pairs(
            candidates,
            count=count,
            seed=seed,
            cross_group=False,
            exclude_keys=excluded,
            target_metric=target_metric,
            disagree=disagree,
            target_quantile=target_quantile,
            context_window=context_window,
            allow_same_group_fallback=allow_same_group_fallback,
            within_group=within_group,
            vary=vary,
        )

    if disagree:
        metrics = list(disagree)
        if len(metrics) != 2:
            raise ValueError(f"disagree needs exactly two metrics, got {metrics}")
        possible = _disagree_pairs(
            possible,
            metrics,
            scales=[_metric_scale(pool, metric) for metric in metrics],
            quantile=target_quantile,
        )
    elif target_metric:
        possible = _target_metric_pairs(
            possible,
            target_metric,
            scale=_metric_scale(pool, target_metric),
            quantile=target_quantile,
        )

    rng.shuffle(possible)
    pairs: list[AbPair] = []
    seen: set[str] = set()
    used_paths: dict[str, int] = {}
    for first, second in possible:
        if len(pairs) >= count:
            break
        key = pair_key(first.path, second.path)
        if key in seen:
            continue
        # Spread the listening over the pool instead of hammering a few clips.
        if used_paths.get(first.path, 0) > 2 or used_paths.get(second.path, 0) > 2:
            continue
        seen.add(key)
        used_paths[first.path] = used_paths.get(first.path, 0) + 1
        used_paths[second.path] = used_paths.get(second.path, 0) + 1
        left, right = (first, second) if rng.random() < 0.5 else (second, first)
        context_paths = shared_context.get(key, ())
        excerpt_seconds = min(4.0, CONTEXT_TARGET_SECONDS / len(context_paths)) if context_paths else None
        pairs.append(
            AbPair(
                pair_id=f"pair_{len(pairs):03d}",
                first=left,
                second=right,
                context_paths=context_paths,
                context_prompt_durations_seconds=shared_prompts.get(key, ()),
                context_excerpt_seconds=excerpt_seconds,
            )
        )
    return pairs


# ── playback ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlaybackClip:
    path: str | Path
    seconds: float | None = None


def find_player() -> list[str] | None:
    """Return an argv prefix for a non-interactive CLI player, if one exists."""
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    if shutil.which("play"):
        return ["play", "-q"]
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    return None


def _player_command(
    path: str | Path,
    *,
    seconds: float | None,
    player: Sequence[str],
) -> list[str]:
    argv = list(player)
    executable = Path(player[0]).name
    if seconds and executable == "ffplay":
        argv += ["-t", str(seconds)]
    if seconds and executable == "play":
        return [*argv, str(path), "trim", "0", str(seconds)]
    return [*argv, str(path)]


def play_clip(path: str | Path, *, seconds: float | None = None, player: list[str] | None = None) -> bool:
    resolved_player = player or find_player()
    if resolved_player is None:
        return False
    argv = _player_command(path, seconds=seconds, player=resolved_player)
    try:
        subprocess.run(argv, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


class NonBlockingClipPlayer:
    """Play clips through a replaceable subprocess without blocking key input."""

    def __init__(self, *, player: list[str] | None = None) -> None:
        self.player = player if player is not None else find_player()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[Any] | None = None
        self._current_path: Path | None = None
        self._generation = 0
        self._sequence_thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self.player is not None

    @property
    def player_name(self) -> str | None:
        return Path(self.player[0]).name if self.player else None

    def _terminate(self, process: subprocess.Popen[Any]) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.25)
        except (OSError, subprocess.SubprocessError):
            pass

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            process = self._process
            self._process = None
            self._current_path = None
        if process is not None:
            self._terminate(process)

    def _active(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def _launch(self, path: Path, seconds: float | None, generation: int) -> subprocess.Popen[Any] | None:
        if self.player is None or not self._active(generation):
            return None
        try:
            process = subprocess.Popen(
                _player_command(path, seconds=seconds, player=self.player),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        with self._lock:
            if generation != self._generation:
                orphan = True
            else:
                orphan = False
                self._process = process
                self._current_path = path
        if orphan:
            self._terminate(process)
            return None
        return process

    def play(self, path: str | Path, *, seconds: float | None = None) -> bool:
        self.stop()
        with self._lock:
            generation = self._generation
        return self._launch(Path(path), seconds, generation) is not None

    def play_sequence(
        self,
        paths: Sequence[str | Path | PlaybackClip],
        *,
        seconds: float | None = None,
    ) -> bool:
        if self.player is None:
            return False
        sequence = [
            item if isinstance(item, PlaybackClip) else PlaybackClip(path=item, seconds=seconds) for item in paths
        ]
        if not sequence:
            return False
        self.stop()
        with self._lock:
            generation = self._generation

        def run() -> None:
            for item in sequence:
                if not self._active(generation):
                    return
                process = self._launch(Path(item.path), item.seconds, generation)
                if process is None:
                    return
                while self._active(generation) and process.poll() is None:
                    time.sleep(0.05)

        self._sequence_thread = threading.Thread(
            target=run,
            name="macchina_ricordi-ab-playback",
            daemon=True,
        )
        self._sequence_thread.start()
        return True

    def current_path(self) -> Path | None:
        with self._lock:
            process = self._process
            path = self._current_path
        if process is not None and process.poll() is not None:
            with self._lock:
                if self._process is process:
                    self._process = None
                    self._current_path = None
            return None
        return path

    def close(self) -> None:
        self.stop()
        thread = self._sequence_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._sequence_thread = None


# ── session ───────────────────────────────────────────────────────────


def _presented_row(pair: AbPair, outcome: str) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "path_a": pair.first.path,
        "path_b": pair.second.path,
        "group_a": pair.first.group,
        "group_b": pair.second.group,
        "context_paths": list(pair.context_paths),
        "outcome": outcome,
        **pair_context_metadata(pair),
    }


def pair_context_metadata(pair: AbPair) -> dict[str, Any]:
    if not pair.context_paths:
        return {}
    first_position = pair.first.session_position
    second_position = pair.second.session_position
    gap = abs(first_position - second_position) if first_position is not None and second_position is not None else None
    return {
        "context_protocol": CONTEXT_PROTOCOL,
        "source_session_id": pair.first.session_id,
        "session_position_a": first_position,
        "session_position_b": second_position,
        "round_gap": gap,
        "previous_memory_size": NOVELTY_MEMORY_ROUNDS,
        "shared_round_count": len(pair.context_paths),
        "context_excerpt_seconds": pair.context_excerpt_seconds,
        "context_excerpt_source": "fresh_audio_tail",
        "playback_a": "fresh_audio_after_prompt",
        "playback_b": "fresh_audio_after_prompt",
        "prompt_duration_seconds_a": pair.first.prompt_duration_seconds,
        "prompt_duration_seconds_b": pair.second.prompt_duration_seconds,
    }


def run_ab_session(
    pairs: Sequence[AbPair],
    *,
    store_path: str | Path,
    question: str = "better",
    listener: str = "me",
    seconds: float | None = None,
    session_id: str = "",
    input_fn: Any = input,
    print_fn: Any = print,
    play_fn: Any = None,
    context: dict[str, Any] | None = None,
    sidecar: Mapping[str, Any] | None = None,
) -> list[PreferenceRecord]:
    """Play each pair blind, ask ``question``, record the verdict. Returns what was written.

    The session also drops a sidecar next to the store describing how it was assembled,
    including the pairs that were skipped or never reached — the store says what was
    judged, the sidecar says what was offered.
    """
    from .preference import append_preference, new_session_id, write_session_sidecar

    asked: Question = get_question(question)
    if asked.type not in (PAIRWISE, PAIRWISE_CONTEXT):
        raise ValueError(f"question {asked.id!r} is {asked.type}, which this pairwise session cannot ask honestly")

    player = find_player()
    if player is None and play_fn is None:
        print_fn("No CLI audio player found (tried ffplay, play, paplay, aplay).")
        return []

    run_id = session_id or new_session_id()
    written: list[PreferenceRecord] = []
    presented: list[dict[str, Any]] = []
    outcomes: dict[str, str] = {pair.pair_id: "unreached" for pair in pairs}
    controls = "a/b/tie, r=replay, s=skip, q=quit"
    if asked.type == PAIRWISE_CONTEXT:
        controls = "a/b/tie, c=context, r=replay, s=skip, q=quit"
    prompt = f"  {asked.prompt} [{controls}]: "

    def play(path: str, limit: float | None) -> None:
        if play_fn is not None:
            play_fn(path)
        else:
            play_clip(path, seconds=limit, player=player)

    def play_pair(pair: AbPair) -> None:
        if asked.type == PAIRWISE_CONTEXT:
            if not pair.context_reel_path:
                raise ValueError(f"pair {pair.pair_id} has no prepared context reel")
            print_fn("  playing context ...")
            play(pair.context_reel_path, None)
        print_fn("  playing A ...")
        play(pair.first.playable_path, seconds)
        print_fn("  playing B ...")
        play(pair.second.playable_path, seconds)

    def finish(message: str) -> list[PreferenceRecord]:
        presented.extend(
            _presented_row(pair, outcomes[pair.pair_id]) for pair in pairs if outcomes[pair.pair_id] == "unreached"
        )
        sidecar_path = write_session_sidecar(
            {
                "question": asked.id,
                "prompt": asked.prompt,
                "listener": listener,
                "seconds_played": seconds,
                "player": player[0] if player else None,
                "judgements": len(written),
                "pairs": presented,
                **dict(sidecar or {}),
            },
            session_id=run_id,
            store_path=store_path,
        )
        print_fn(message)
        print_fn(f"Session sidecar → {sidecar_path}")
        return written

    for index, pair in enumerate(pairs, start=1):
        sequence = "context, A, then B" if asked.type == PAIRWISE_CONTEXT else "A then B"
        print_fn(f"\n[{index}/{len(pairs)}] listening — {sequence}")
        play_pair(pair)
        while True:
            answer = parse_verdict(input_fn(prompt))
            while answer is None:
                answer = parse_verdict(input_fn(f"  please answer {controls}: "))
            if answer == "replay":
                play_pair(pair)
                continue
            if answer == "context":
                if asked.type == PAIRWISE_CONTEXT:
                    print_fn("  replaying context ...")
                    play(pair.context_reel_path, None)
                continue
            if answer == "quit":
                return finish(f"Stopped after {len(written)} judgements.")
            if answer == "skip":
                outcomes[pair.pair_id] = "skipped"
                presented.append(_presented_row(pair, "skipped"))
                break
            record = PreferenceRecord(
                pair_id=pair.pair_id,
                path_a=pair.first.path,
                path_b=pair.second.path,
                winner=answer,
                listener=listener,
                question=asked.id,
                session_id=run_id,
                seconds_played=seconds,
                context={
                    "group_a": pair.first.group,
                    "group_b": pair.second.group,
                    "label_a": pair.first.label,
                    "label_b": pair.second.label,
                    **(context or {}),
                    **pair_context_metadata(pair),
                },
                context_paths=list(pair.context_paths),
            )
            append_preference(record, path=store_path)
            written.append(record)
            outcomes[pair.pair_id] = answer
            presented.append(_presented_row(pair, answer))
            break
    return finish(f"\nRecorded {len(written)} judgements → {store_path}")


def run_ab_tui(*args: Any, **kwargs: Any) -> list[PreferenceRecord]:
    """Run the blind session through the single-keystroke terminal interface."""
    from .ab_tui import run_ab_tui as _run_ab_tui

    return _run_ab_tui(*args, **kwargs)
