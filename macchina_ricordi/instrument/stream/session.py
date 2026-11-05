from __future__ import annotations

import fcntl
import json
import os
import random
import shutil
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ...models.specs import clamp_prompt_seconds, resolve_segment_duration
from ...musicgen.backend import GenerationRequest, resolve_generation_checkpoint
from ...musicgen.shared import utc_now_iso, write_text_atomic
from ..controls import BRING_BACK, BRING_BACK_LIGHTLY, BRING_BACK_STRONGLY
from ..prompt_transform import (
    RATE_MODES,
    PromptTilt,
    clamp_prompt_rate,
    clamp_tilt_db,
    render_prompt_transform,
)
from ..provenance import resolve_generation_profile
from ..run_paths import utc_compact_now
from .playback import NullPlaybackWorker, PlaybackWorker
from .worker import PersistentStreamGenerator, StreamGenerator, StreamModelSource

VALID_COMMANDS = ("continue", "reroll", "pin", "inject", "inject_light", "inject_heavy", "stop")
SEED_MODES = ("uncond", "excerpt")
DEFAULT_STREAMS_DIR = Path("artifacts/streams")
DEFAULT_STREAM_MODEL_ID = "facebook/musicgen-small"


@dataclass(frozen=True)
class StreamSessionPaths:
    run_id: str
    session_dir: Path
    rounds_dir: Path
    control_path: Path
    performance_path: Path
    manifest_path: Path
    events_path: Path


MAX_STREAM_CANDIDATES = 8
MIN_COMPLETION_DURATION_SECONDS = 1.0
DEFAULT_COMPLETION_DURATION_SECONDS = 10.0
MAX_COMPLETION_DURATION_SECONDS = 60.0
MAX_ROUND_DELAY_SECONDS = 3.0
PINNED_INJECTION_WEIGHT = BRING_BACK.weight
PINNED_INJECTION_LIGHT_WEIGHT = BRING_BACK_LIGHTLY.weight
PINNED_INJECTION_HEAVY_WEIGHT = BRING_BACK_STRONGLY.weight


@dataclass(frozen=True)
class StreamControl:
    command: Literal["continue", "reroll", "pin", "inject", "inject_light", "inject_heavy", "stop"] = "continue"
    # Set by the TUI at keypress time so a buffered session pins what was
    # playing then, rather than whatever round the generator reaches later.
    pin_target: str | None = None
    tau_seconds: float = 5.0
    completion_duration_seconds: float = DEFAULT_COMPLETION_DURATION_SECONDS
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    # min-p sampling threshold (replaces top-k/top-p when > 0): keeps tokens
    # with p >= min_p * p_max after temperature. 0.05-0.1 is the useful range.
    min_p: float = 0.0
    cfg_coef: float = 3.0
    extend_stride: float = 8.0
    model_xp: str | None = None
    # Best-of-N: generate N candidate rounds in one batched call (near-free on
    # GPU; the AR loop is launch-bound) and keep the best by silence+arc score.
    candidates: int = 1
    # Anti-déjà-vu: when > 0 and candidates > 1, candidate ranking blends arc
    # with novelty (distance from the last few rounds' MFCC fingerprints):
    # (1-w)*arc + w*novelty, both minmax-scaled within the candidate batch.
    novelty_weight: float = 0.0
    # Drone guard (see `drone_readings`). `stuck_weight` ranks candidates whose
    # tail barely moves below the others; inert unless candidates > 1.
    stuck_weight: float = 0.0
    # When the last `stuck_kick_rounds` rounds are all drones (drone_score at or
    # above the kick level), the next round alone runs hotter:
    # temperature + `stuck_kick_temperature`, top_k * `stuck_kick_top_k_factor`.
    # 0 = off.
    stuck_kick_rounds: int = 0
    stuck_kick_temperature: float = 0.4
    stuck_kick_top_k_factor: float = 2.0
    # Recapitulation: every `motif_period` rounds the continuation prompt is a
    # motif_mix blend of the current tail with a pinned motif's tail. Motifs
    # are captured live with the one-shot `pin` command (or dropped into
    # motif_dir by hand); default bank is <session>/motifs.
    motif_dir: str | None = None
    motif_period: int = 0
    motif_mix: float = 0.5
    # Tail rescue: when the window that would become the next prompt is mostly
    # silence, re-prompt from the latest earlier window in the same round that
    # still has audio. Trades an audible jump cut for not fading into nothing.
    tail_rescue: bool = True
    # Prompt-tilt steering: three-band EQ (dB) on the continuation prompt, so
    # the model hears a bass- or treble-heavy recent past and continues from it.
    # Compounds while held — each round's output is the next round's prompt.
    prompt_tilt_low_db: float = 0.0
    prompt_tilt_mid_db: float = 0.0
    prompt_tilt_high_db: float = 0.0
    # Tape-style varispeed of the prompt window only: `rate * tau` seconds of
    # tail are resampled down to exactly tau, so token count, the horizon guard
    # and the buffer arithmetic are untouched. Only what is fed back moves, not
    # what is played, so the drift arrives as the music genuinely migrating.
    # "tape" couples pitch and tempo; "tempo" phase-vocodes, keeping pitch and
    # smearing transients.
    prompt_rate: float = 1.0
    prompt_rate_mode: Literal["tape", "tempo"] = "tape"
    # Playback throttle, live. The generator pauses once the buffer is this far
    # ahead of the playback head and resumes at `resume_buffer_ahead_seconds`
    # (default: max - 10). Both are re-read *inside* the wait, so shrinking the
    # buffer mid-throttle lands immediately instead of after the current
    # drain — this is the responsiveness knob, since the buffer is how long
    # every other control takes to become audible. None = no throttle.
    max_buffer_ahead_seconds: float | None = None
    resume_buffer_ahead_seconds: float | None = None
    # Live on/off for the throttle above, kept separate from the cap so opening
    # it does not destroy the session's own thresholds: switching back restores
    # them rather than a default. Off lets the buffer run ahead unbounded, which
    # is how you bank a cushion before a model swap — the reload stalls
    # generation for 10-60s and only already-generated audio covers that.
    throttle_enabled: bool = True
    # Rest this long before each round, so the knobs can be set for it; cut
    # short when the buffer would otherwise run dry. 0 = generate back-to-back.
    round_delay_seconds: float = 0.0


@dataclass(frozen=True)
class RoundRecord:
    round_idx: int
    take_idx: int
    timestamp: str
    command_consumed: str
    active_control: dict[str, Any]
    seed_used: int | None
    round_wav: str
    duration_seconds: float
    elapsed_seconds: float
    score: dict[str, Any] | None = None
    prompt_start_seconds: float | None = None
    prompt_duration_seconds: float | None = None
    prompt_audio: str | None = None
    motif_source: str | None = None
    motif_injection_weight: float | None = None
    # Set to the tail start the rescue rejected, when tail rescue fired.
    tail_rescued_from_seconds: float | None = None
    # The prompt transforms as actually applied. `prompt_rate` can come back
    # below what was asked when the source could not reach far enough back.
    prompt_rate: float | None = None
    prompt_rate_mode: str | None = None
    prompt_tilt_db: dict[str, float] | None = None
    model_xp: str | None = None
    playback_seconds: float | None = None
    buffer_seconds_ahead: float | None = None
    # The temperature/top_k this round actually ran at, when the drone guard
    # kicked it; `active_control` keeps the player's own values.
    stuck_kick: dict[str, Any] | None = None


@dataclass
class StreamSessionResult:
    paths: StreamSessionPaths
    rounds_written: int
    stopped_by: Literal["command", "max_rounds", "interrupt"] = "max_rounds"
    extras: dict[str, Any] = field(default_factory=dict)


def build_stream_session_paths(output_dir: str | Path, *, run_id: str, name: str | None = None) -> StreamSessionPaths:
    root = Path(output_dir).expanduser().resolve()
    session_dir = root / (name or run_id)
    rounds_dir = session_dir / "rounds"
    return StreamSessionPaths(
        run_id=run_id,
        session_dir=session_dir,
        rounds_dir=rounds_dir,
        control_path=session_dir / "control.json",
        performance_path=session_dir / "performance.jsonl",
        manifest_path=session_dir / "manifest.json",
        events_path=session_dir / "events.jsonl",
    )


@contextmanager
def control_file_lock(path: Path) -> Iterator[None]:
    """Hold ``control.json`` for one read-modify-write, against every other writer of it.

    The loop, the TUI and the GUI each rewrite the whole file, so two of them interleaving would silently drop
    one's edit. The lock is a file of its own because ``write_control`` replaces ``control.json`` by rename, and a
    lock on the old file would not guard the new one. ``flock`` goes with the handle, so a writer that dies holding
    it cannot wedge the others.
    """
    with path.with_name(path.name + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def write_control(path: Path, control: StreamControl) -> None:
    payload = asdict(control)
    write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def read_control(path: Path) -> StreamControl:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Control file does not exist: {path}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Control file is not valid JSON: {path}: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Control file must be a JSON object, got {type(payload).__name__}: {path}")
    if "round_duration_seconds" in payload:
        raise ValueError(
            "Control file uses removed field 'round_duration_seconds'; replace it with 'completion_duration_seconds'."
        )

    defaults = StreamControl()
    command = payload.get("command", defaults.command)
    if command not in VALID_COMMANDS:
        raise ValueError(f"Control file has invalid command '{command}'. Expected one of: {VALID_COMMANDS}")

    rate_mode = payload.get("prompt_rate_mode", defaults.prompt_rate_mode)
    if rate_mode not in RATE_MODES:
        raise ValueError(f"Control file has invalid prompt_rate_mode '{rate_mode}'. Expected one of: {RATE_MODES}")

    def _unit(name: str, default: float) -> float:
        return min(1.0, max(0.0, float(payload.get(name, default))))

    def _optional_seconds(name: str, default: float | None) -> float | None:
        value = payload.get(name, default)
        return None if value is None else max(0.0, float(value))

    completion_duration = float(payload.get("completion_duration_seconds", defaults.completion_duration_seconds))
    if not MIN_COMPLETION_DURATION_SECONDS <= completion_duration <= MAX_COMPLETION_DURATION_SECONDS:
        raise ValueError(
            "Control field 'completion_duration_seconds' must be between "
            f"{MIN_COMPLETION_DURATION_SECONDS:g} and {MAX_COMPLETION_DURATION_SECONDS:g} seconds."
        )

    return StreamControl(
        command=command,
        pin_target=payload.get("pin_target", defaults.pin_target),
        tau_seconds=float(payload.get("tau_seconds", defaults.tau_seconds)),
        completion_duration_seconds=completion_duration,
        temperature=float(payload.get("temperature", defaults.temperature)),
        top_k=int(payload.get("top_k", defaults.top_k)),
        top_p=float(payload.get("top_p", defaults.top_p)),
        min_p=_unit("min_p", defaults.min_p),
        cfg_coef=float(payload.get("cfg_coef", defaults.cfg_coef)),
        extend_stride=float(payload.get("extend_stride", defaults.extend_stride)),
        model_xp=payload.get("model_xp", defaults.model_xp),
        candidates=max(
            1,
            min(MAX_STREAM_CANDIDATES, int(payload.get("candidates", defaults.candidates))),
        ),
        novelty_weight=_unit("novelty_weight", defaults.novelty_weight),
        stuck_weight=_unit("stuck_weight", defaults.stuck_weight),
        stuck_kick_rounds=max(0, int(payload.get("stuck_kick_rounds", defaults.stuck_kick_rounds))),
        stuck_kick_temperature=max(0.0, float(payload.get("stuck_kick_temperature", defaults.stuck_kick_temperature))),
        stuck_kick_top_k_factor=max(
            1.0, float(payload.get("stuck_kick_top_k_factor", defaults.stuck_kick_top_k_factor))
        ),
        motif_dir=payload.get("motif_dir", defaults.motif_dir),
        motif_period=max(0, int(payload.get("motif_period", defaults.motif_period))),
        motif_mix=_unit("motif_mix", defaults.motif_mix),
        tail_rescue=bool(payload.get("tail_rescue", defaults.tail_rescue)),
        prompt_tilt_low_db=clamp_tilt_db(payload.get("prompt_tilt_low_db", defaults.prompt_tilt_low_db)),
        prompt_tilt_mid_db=clamp_tilt_db(payload.get("prompt_tilt_mid_db", defaults.prompt_tilt_mid_db)),
        prompt_tilt_high_db=clamp_tilt_db(payload.get("prompt_tilt_high_db", defaults.prompt_tilt_high_db)),
        prompt_rate=clamp_prompt_rate(payload.get("prompt_rate", defaults.prompt_rate)),
        prompt_rate_mode=rate_mode,
        max_buffer_ahead_seconds=_optional_seconds("max_buffer_ahead_seconds", defaults.max_buffer_ahead_seconds),
        resume_buffer_ahead_seconds=_optional_seconds(
            "resume_buffer_ahead_seconds", defaults.resume_buffer_ahead_seconds
        ),
        throttle_enabled=bool(payload.get("throttle_enabled", defaults.throttle_enabled)),
        # Rounded so 0.1-step key presses land back on exactly 0 (= off).
        round_delay_seconds=round(
            min(
                MAX_ROUND_DELAY_SECONDS,
                max(0.0, float(payload.get("round_delay_seconds", defaults.round_delay_seconds))),
            ),
            2,
        ),
    )


def consume_command(path: Path, control: StreamControl) -> StreamControl:
    """If `command` is a one-shot, reset it to `continue` on disk.

    Resets the command in the file as it is now rather than writing ``control`` back: that copy was read when the
    round started, so a knob moved since (or a newer one-shot) would otherwise be silently undone.
    """
    if control.command == "continue":
        return control
    reset = replace(control, command="continue", pin_target=None)
    with control_file_lock(path):
        try:
            current = read_control(path)
        except (FileNotFoundError, ValueError):
            current = control  # unreadable mid hand-edit: put the round's own copy back, as before
        if (current.command, current.pin_target) == (control.command, control.pin_target):
            write_control(path, replace(current, command="continue", pin_target=None))
    return reset


def append_performance_row(path: Path, record: RoundRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), default=str)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class StreamEventLog:
    """Print a session event and append it to the ``events.jsonl`` sidecar.

    Throttle waits, model swaps and playback errors previously existed only as
    stdout lines, so nothing reading a session afterwards could see them —
    ``performance.jsonl`` has one row per round and no notion of an event. This
    writes both.

    Appends are single ``write`` calls on an ``O_APPEND`` handle, taken under a
    lock, because the playback worker emits from its own thread. Writing is
    best-effort: a failed sidecar write must never interrupt a performance, so
    it degrades to printing only.
    """

    def __init__(self, path: Path | None, *, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        self.round_idx: int | None = None
        self._lock = threading.Lock()
        self._disabled = False

    def emit(
        self,
        kind: str,
        message: str,
        *,
        level: Literal["info", "warning", "error"] = "info",
        round_idx: int | None = None,
        **data: Any,
    ) -> None:
        if self.echo:
            print(f"{kind}: {message}")
        if self.path is None or self._disabled:
            return
        row = {
            "timestamp": _now_iso(),
            "round_idx": self.round_idx if round_idx is None else round_idx,
            "kind": kind,
            "level": level,
            "message": message,
        }
        if data:
            row["data"] = data
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
        except OSError as exc:  # noqa: BLE001 — the log is a convenience, the session is not
            self._disabled = True
            print(f"event_log_disabled: {self.path}: {exc}")


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in WAV header: {path}")
    return frames / float(sample_rate)


def extract_tail(source_wav: Path, tau_seconds: float) -> tuple[float, float]:
    """Compute (prompt_start_seconds, prompt_duration_seconds) for the tail of `source_wav`."""
    if tau_seconds <= 0:
        raise ValueError(f"tau_seconds must be positive, got {tau_seconds}")
    duration = _wav_duration_seconds(source_wav)
    if duration <= 0:
        raise ValueError(f"Could not determine non-zero duration of {source_wav}")
    tau = min(tau_seconds, duration)
    prompt_start = max(0.0, duration - tau)
    return prompt_start, tau


# Tail rescue. The prompt window is the whole of what the next round is
# conditioned on, so once it is mostly dead air the model is being asked to
# continue a fade-out — and it obliges, round after round. Past this ceiling we
# back off to the latest earlier window in the same round that still has audio
# in it. Costs continuity (an audible jump cut) to buy the stream's life, so
# the ceiling is set high enough that only a genuinely dying tail trips it.
_PROMPT_SILENCE_CEILING = 0.60
_PROMPT_BACKOFF_HOP_SECONDS = 0.5


def _rescue_prompt_start(
    source_wav: Path,
    *,
    tau_seconds: float,
    min_start_seconds: float = 0.0,
    silence_ceiling: float = _PROMPT_SILENCE_CEILING,
) -> float | None:
    """Find an earlier prompt window when the tail of ``source_wav`` is dead.

    Scans backwards from the tail in `_PROMPT_BACKOFF_HOP_SECONDS` steps for the
    latest tau-length window at or after ``min_start_seconds`` whose silence
    fraction clears ``silence_ceiling``. ``min_start_seconds`` should be the end
    of the round's own prompt echo, so a rescue never re-prompts with audio that
    was already a prompt. Returns the new start, or ``None`` when the tail is
    fine, no earlier window is better, or the audio could not be read — the
    caller then keeps the plain tail.
    """
    try:
        import librosa

        from ..duet import silence_fraction

        waveform, sr = librosa.load(source_wav, sr=16_000, mono=True, res_type="soxr_qq")
    except Exception:  # noqa: BLE001 — rescue is a preference, never fatal
        return None
    if waveform.size == 0 or sr <= 0:
        return None

    window = min(int(round(tau_seconds * sr)), waveform.size)
    if window <= 0:
        return None
    try:
        tail_start = waveform.size - window
        if silence_fraction(waveform[tail_start:]) <= silence_ceiling:
            return None  # the tail still has audio in it; nothing to rescue

        earliest = max(0, int(round(min_start_seconds * sr)))
        hop = max(1, int(round(_PROMPT_BACKOFF_HOP_SECONDS * sr)))
        start = tail_start - hop
        while start >= earliest:
            if silence_fraction(waveform[start : start + window]) <= silence_ceiling:
                return start / float(sr)
            start -= hop
    except Exception:  # noqa: BLE001
        return None
    return None


def _round_filename(round_idx: int, take_idx: int) -> str:
    suffix = "" if take_idx == 0 else f"_take_{take_idx}"
    return f"round_{round_idx:03d}{suffix}.wav"


def _score_round(wav_path: Path, *, prompt_duration_seconds: float | None = None) -> dict[str, Any] | None:
    """Per-round scoring: only the readings the instrument actually uses.

    This used to run the full arrhythmic scorer (29 fields) plus the full arc
    scorer, ~850 ms per round *and per best-of-N candidate* — more than the
    generation it was scoring. `macchina_ricordi.scoring.live` computes the nine numbers
    the TUI draws and candidate selection ranks by, on identical geometry, so
    every field is bit-for-bit the measurement it was before; the rest are gone.
    See that module for the measurements and for what the thinner sidecar costs.

    ``prompt_duration_seconds`` trims the re-prompt echo before the arc metrics so
    they reflect only the freshly generated audio (the seed round passes ``None``).
    """
    try:
        from macchina_ricordi.scoring.live import score_live_round

        return score_live_round(wav_path, prompt_duration_seconds=prompt_duration_seconds).to_dict()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"scoring failed: {exc!r}"}


_CANDIDATE_SILENCE_CEILING = 0.30

# Trailing-silence gate. A round's tail becomes the next round's continuation
# prompt, so silence at the *end* is self-reinforcing in a way mid-round
# silence is not: the model is conditioned on a fade-out and obliges. Rank
# candidates whose final `WINDOW_FRACTION` of fresh audio is quieter than
# `TAIL_SILENCE_CEILING` below the ones that are not.
_CANDIDATE_TAIL_WINDOW_FRACTION = 0.10
_CANDIDATE_TAIL_SILENCE_CEILING = 0.35

# Anti-déjà-vu memory: fingerprints of the last N selected round tails.
NOVELTY_MEMORY_ROUNDS = 8


def _candidate_feature_vector(wav_path: Path, *, prompt_duration_seconds: float | None = None) -> Any | None:
    """MFCC-stats fingerprint of a round's freshly generated tail (16 kHz mono).

    Same vector `timbral_coherence`/`mfcc_cosine` are built on, so "novel"
    means the same thing here as everywhere else in the project. Returns
    ``None`` on any failure — novelty is an optional preference, never fatal.
    """
    try:
        import librosa

        from macchina_ricordi.scoring.similarity import _mfcc_stats_vector

        offset = float(prompt_duration_seconds) if prompt_duration_seconds else 0.0
        waveform, sr = librosa.load(wav_path, sr=16_000, mono=True, offset=offset, res_type="soxr_qq")
        if waveform.size < int(sr) // 4:
            return None
        return _mfcc_stats_vector(waveform, sr=int(sr))
    except Exception:  # noqa: BLE001
        return None


def _tail_silence_of(wav_path: Path, *, prompt_duration_seconds: float | None = None) -> float | None:
    """Silence fraction of the final slice of a round's freshly generated audio.

    The window is `_CANDIDATE_TAIL_WINDOW_FRACTION` of the post-prompt region —
    the part that will be re-prompted from — measured with the arrhythmic
    scorer's silence notion (RMS frames below -35 dBFS). Returns ``None`` on any
    failure; like novelty, this is a preference and never fatal.
    """
    try:
        import librosa

        from ..duet import silence_fraction

        offset = float(prompt_duration_seconds) if prompt_duration_seconds else 0.0
        waveform, sr = librosa.load(wav_path, sr=16_000, mono=True, offset=offset, res_type="soxr_qq")
        if waveform.size < int(sr) // 4:
            return None
        window = max(1, int(round(waveform.size * _CANDIDATE_TAIL_WINDOW_FRACTION)))
        return silence_fraction(waveform[-window:])
    except Exception:  # noqa: BLE001
        return None


# Drone guard. A held note in the tail is self-reinforcing the same way silence
# is: it becomes the prompt and the model continues it. Read on the last tau of
# fresh audio. In the author's historical check of 57 blind clips across 9 models,
# drones scored 0.60-0.70 and nothing else went above 0.26, so the kick level sits
# mid-gap. This is an empirical threshold, not a guarantee for every model.
# Stationarity alone also flags near-drones, so it does the ranking.
_DRONE_KICK_LEVEL = 0.4
# The window the level was validated on; a longer one reads less stationary.
_DRONE_WINDOW_SECONDS = 5.0


def drone_readings(waveform: Any, *, sr: int = 16_000) -> dict[str, float] | None:
    """How stuck a window is: ``stationarity``, ``chroma_top`` and their product ``drone_score``.

    ``stationarity`` is the mean cosine between each audible frame's mel spectrum
    and the window's mean one (1 = the spectrum never moves); ``chroma_top`` is the
    share of chroma energy on the loudest pitch class. Frames more than 30 dB under
    the window's loudest are ignored, so a fade does not dilute either.
    """
    import librosa
    import numpy as np

    if waveform.size < sr:
        return None
    S = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=256))
    rms_db = 20 * np.log10(np.sqrt((S**2).mean(axis=0)) + 1e-9)
    alive = rms_db > rms_db.max() - 30
    if alive.sum() < 8:
        return None
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=64)[:, alive] ** 0.5
    unit = mel / (np.linalg.norm(mel, axis=0, keepdims=True) + 1e-12)
    mean = unit.mean(axis=1)
    mean /= np.linalg.norm(mean) + 1e-12
    stationarity = float((unit.T @ mean).mean())
    chroma = librosa.feature.chroma_stft(S=S**2, sr=sr)[:, alive].sum(axis=1)
    chroma_top = float(chroma.max() / (chroma.sum() + 1e-12))
    return {"stationarity": stationarity, "chroma_top": chroma_top, "drone_score": stationarity * chroma_top}


def _drone_readings_of(wav_path: Path, *, prompt_duration_seconds: float | None = None) -> dict[str, float] | None:
    """`drone_readings` of the last 5 s of fresh audio — the end of the next prompt."""
    try:
        import librosa

        offset = float(prompt_duration_seconds) if prompt_duration_seconds else 0.0
        waveform, sr = librosa.load(wav_path, sr=16_000, mono=True, offset=offset, res_type="soxr_qq")
        return drone_readings(waveform[-int(_DRONE_WINDOW_SECONDS * sr) :], sr=int(sr))
    except Exception:  # noqa: BLE001 — a preference, never fatal
        return None


def stuck_kick(
    control: StreamControl, recent_scores: Sequence[float | None], rounds_since_kick: int
) -> tuple[float, int] | None:
    """(temperature, top_k) for a kicked round, or ``None`` when no kick is due.

    Due when the last ``stuck_kick_rounds`` rounds all have a drone_score at or
    above the kick level, and as many rounds have passed since the previous
    kick: a drone that survives a kick gets the next one k rounds later, never
    on consecutive rounds.
    """
    k = control.stuck_kick_rounds
    if k <= 0 or rounds_since_kick < k or len(recent_scores) < k:
        return None
    if not all(s is not None and s >= _DRONE_KICK_LEVEL for s in list(recent_scores)[-k:]):
        return None
    temperature = min(3.0, control.temperature + control.stuck_kick_temperature)
    # top_k 0 already means every token.
    top_k = control.top_k and min(2048, round(control.top_k * control.stuck_kick_top_k_factor))
    return temperature, top_k


DEFAULT_NOVELTY_CALIBRATION = Path("research/analysis-synthesis-2026-05/novelty_calibration.json")


@dataclass(frozen=True)
class NoveltyCalibration:
    """Per-dimension mean/std used to whiten MFCC fingerprints before the cosine.

    Raw cosine over `_mfcc_stats_vector` is pinned near 1 — MFCC[0] is
    log-energy and the top dimensions dominate the norm — so novelty had almost
    no dynamic range. Whitening first restores it.

    The corpus behind these numbers is what "novel" gets measured against, and
    is a deliberate choice rather than a detail. It is deliberately
    *frozen and shared across models*: with a per-model calibration the reading
    would jump discontinuously at a mid-session ``model_xp`` swap, which is
    exactly where you most want to read it.
    """

    means: Any
    stds: Any
    source: str
    path: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"path": self.path, "source": self.source, "dims": int(len(self.means))}


def load_novelty_calibration(path: str | Path) -> NoveltyCalibration | None:
    """Read the ``mfcc`` block of a `score sim-calibrate` JSON. ``None`` if unusable."""
    resolved = Path(path).expanduser()
    try:
        import numpy as np

        payload = json.loads(resolved.read_text(encoding="utf-8"))
        block = payload["calibration"]["mfcc"]
        means = np.asarray(block["means"], dtype=float)
        stds = np.asarray(block["stds"], dtype=float)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if means.size == 0 or means.shape != stds.shape:
        return None
    return NoveltyCalibration(
        means=means,
        stds=stds,
        source=str(payload.get("source", "unknown")),
        path=str(resolved),
    )


def _novelty_of(
    vector: Any | None,
    recent_vectors: Sequence[Any],
    *,
    calibration: NoveltyCalibration | None = None,
) -> float | None:
    """Distance from the recent-round fingerprints: 1 - max cosine similarity.

    With a ``calibration``, both sides are z-scored per dimension first and the
    result is halved, since whitened cosine spans [-1, 1] rather than sitting
    just under 1. Both variants therefore land on [0, 1] and can be logged and
    compared side by side.
    """
    if vector is None or not len(recent_vectors):
        return None
    try:
        from macchina_ricordi.scoring.similarity import _cosine_similarity, _zscore_normalize

        if calibration is None:
            best = max(_cosine_similarity(vector, recent) for recent in recent_vectors)
            return max(0.0, 1.0 - float(best))

        whitened = _zscore_normalize(vector, calibration.means, calibration.stds)
        best = max(
            _cosine_similarity(whitened, _zscore_normalize(recent, calibration.means, calibration.stds))
            for recent in recent_vectors
        )
        return max(0.0, min(1.0, (1.0 - float(best)) / 2.0))
    except Exception:  # noqa: BLE001
        return None


def novelty_readings(
    vector: Any | None,
    recent_vectors: Sequence[Any],
    *,
    calibration: NoveltyCalibration | None = None,
) -> dict[str, float | None]:
    """Raw and whitened novelty side by side.

    Logged for every round regardless of `candidates` or `novelty_weight` —
    novelty is an instrument reading first and a control second, and the two
    variants are only comparable when measured on identical rounds.

    There used to be a third field, ``novelty_vs_memory``, holding whichever of
    the two was available. It was a literal alias of ``novelty_whitened`` in
    every calibrated session and silently meant the raw cosine otherwise, which
    made it ambiguous on top of redundant. Readers take ``novelty_whitened``
    with a ``novelty_raw`` fallback and know which one they got.
    """
    raw = _novelty_of(vector, recent_vectors)
    whitened = _novelty_of(vector, recent_vectors, calibration=calibration) if calibration else None
    return {
        "novelty_raw": raw,
        "novelty_whitened": whitened,
    }


def _minmax_scale(values: Sequence[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _select_best_candidate(
    candidate_paths: Sequence[Path],
    *,
    prompt_duration_seconds: float | None,
    scorer: Callable[..., dict[str, Any] | None] = _score_round,
    novelty_weight: float = 0.0,
    recent_vectors: Sequence[Any] = (),
    calibration: NoveltyCalibration | None = None,
    feature_extractor: Callable[..., Any | None] = _candidate_feature_vector,
    tail_silence_extractor: Callable[..., float | None] = _tail_silence_of,
    stuck_weight: float = 0.0,
    drone_extractor: Callable[..., dict[str, float] | None] = _drone_readings_of,
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]], Any | None]:
    """Pick the best round among batched candidates.

    Gates, in order: candidates that clear the overall silence ceiling, then —
    among those — the ones that do not end in silence (see
    ``_CANDIDATE_TAIL_SILENCE_CEILING``), since a silent tail is what the next
    round gets re-prompted with. The surviving pool is ranked by
    arc_coherence_score, blended with novelty (distance from the recent rounds'
    fingerprints) when ``novelty_weight`` > 0, then blended the same way with
    (1 - tail stationarity) by ``stuck_weight``; every term is minmax-scaled in
    the pool. Each gate is skipped rather than enforced when it would eliminate everything: if every candidate ends in
    silence the pool is left untouched, and if every candidate is too silent
    overall the least silent one wins outright. Returns ``(winner_index,
    winner_score, summary, winner_vector)`` where ``summary`` has one row per
    candidate for the performance log and ``winner_vector`` is the winner's
    fingerprint (``None`` unless novelty was computed).
    """
    scores: list[dict[str, Any] | None] = [
        scorer(path, prompt_duration_seconds=prompt_duration_seconds) for path in candidate_paths
    ]

    def silence_of(payload: dict[str, Any] | None) -> float:
        if payload is None:
            return 1.0
        value = payload.get("silence_fraction")
        return float(value) if value is not None else 1.0

    def arc_of(payload: dict[str, Any] | None) -> float:
        if payload is None:
            return -1.0
        value = payload.get("arc_coherence_score")
        return float(value) if value is not None else -1.0

    novelty_active = novelty_weight > 0.0 and len(recent_vectors) > 0
    vectors: list[Any | None] = [None] * len(scores)
    novelties: list[float | None] = [None] * len(scores)
    if novelty_active:
        vectors = [feature_extractor(path, prompt_duration_seconds=prompt_duration_seconds) for path in candidate_paths]
        novelties = [_novelty_of(vector, recent_vectors, calibration=calibration) for vector in vectors]

    eligible = [i for i in range(len(scores)) if silence_of(scores[i]) <= _CANDIDATE_SILENCE_CEILING]
    tail_silences: list[float | None] = [None] * len(scores)
    drones: list[dict[str, float] | None] = [None] * len(scores)
    if eligible:
        # Only the eligible candidates are worth decoding a second time.
        for i in eligible:
            tail_silences[i] = tail_silence_extractor(
                candidate_paths[i], prompt_duration_seconds=prompt_duration_seconds
            )
        fresh_ending = [
            i for i in eligible if tail_silences[i] is None or tail_silences[i] <= _CANDIDATE_TAIL_SILENCE_CEILING
        ]
        pool = fresh_ending or eligible
        if stuck_weight > 0.0:
            for i in pool:
                drones[i] = drone_extractor(candidate_paths[i], prompt_duration_seconds=prompt_duration_seconds)
        use_novelty = novelty_active and any(novelties[i] is not None for i in pool)
        stationarity = {i: d["stationarity"] for i in pool if (d := drones[i]) is not None}
        if len(pool) > 1 and (use_novelty or stationarity):
            blended = dict(zip(pool, _minmax_scale([arc_of(scores[i]) for i in pool]), strict=True))
            if use_novelty:
                nov_scaled = _minmax_scale([novelties[i] or 0.0 for i in pool])
                for pos, i in enumerate(pool):
                    blended[i] = (1.0 - novelty_weight) * blended[i] + novelty_weight * nov_scaled[pos]
            if stationarity:
                # An unreadable tail ranks as the most stuck one, not the least.
                worst = max(stationarity.values())
                stuck_scaled = _minmax_scale([stationarity.get(i, worst) for i in pool])
                for pos, i in enumerate(pool):
                    blended[i] = (1.0 - stuck_weight) * blended[i] + stuck_weight * (1.0 - stuck_scaled[pos])
            winner = max(pool, key=lambda i: (blended[i], arc_of(scores[i])))
        else:
            winner = max(pool, key=lambda i: arc_of(scores[i]))
    else:
        winner = min(range(len(scores)), key=lambda i: silence_of(scores[i]))

    summary = [
        {
            "candidate": i,
            "selected": i == winner,
            "silence_fraction": silence_of(scores[i]) if scores[i] is not None else None,
            "arc_coherence_score": arc_of(scores[i]) if scores[i] is not None else None,
            "tail_silence_fraction": tail_silences[i],
            "novelty": novelties[i],
            **(drones[i] or {}),
        }
        for i in range(len(scores))
    ]
    return winner, scores[winner], summary, vectors[winner]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_event(kind: str, message: str, **_data: Any) -> None:
    """Fallback emitter, for call sites reachable without a session event log."""
    print(f"{kind}: {message}")


_PHASE_LABELS = {
    "generating": "generating",
    "scoring": "scoring",
    "throttled": "throttled (buffer full)",
    "resting": "resting before the round",
    "waiting": "starting round",
    "stopped": "stopped",
}


class StreamStatus:
    """What the round loop is doing right now, for the part disk cannot show.

    Everything else a dashboard needs is on disk; this covers only the part that is in flight — which round is
    being worked on, and what the loop is doing with it. Written by the generator thread, read by the TUI and by
    `LiveStatusWriter` — a lock rather than an atomic swap because the three fields must be read as one
    consistent tuple.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = "starting"
        self._round_idx: int | None = None
        self._take_idx = 0
        self._since = time.monotonic()

    def set_phase(self, phase: str, *, round_idx: int | None = None, take_idx: int = 0) -> None:
        with self._lock:
            self._phase = phase
            self._round_idx = round_idx
            self._take_idx = take_idx
            self._since = time.monotonic()

    def snapshot(self) -> tuple[str, int | None, int, float]:
        with self._lock:
            return self._phase, self._round_idx, self._take_idx, time.monotonic() - self._since

    def describe(self) -> str:
        phase, round_idx, take_idx, elapsed = self.snapshot()
        label = _PHASE_LABELS.get(phase, phase)
        where = f" round {round_idx}" if round_idx is not None else ""
        if take_idx:
            where += f" take {take_idx}"
        return f"{label}{where} · {elapsed:.1f}s"


LIVE_STATUS_FILENAME = "live_status.json"
# The `--playlist` models, in digit-key order, for a front end in another process to offer the same slots.
PLAYLIST_FILENAME = "playlist.json"


class LiveStatusWriter:
    """Keep ``live_status.json`` current with what only this process knows.

    The phase of the round in flight and the file the speakers are playing live in memory (`StreamStatus`, the
    playback worker), so a front end in another process — the GUI — reads them here; everything else it shows is
    already on disk. Rewritten every ``interval`` seconds, which makes ``updated_at`` a heartbeat: with ``pid`` it
    tells a live session from one killed before it could write its manifest. Best-effort, like the event log: a
    write that fails is reported once and never reaches the performance.
    """

    def __init__(
        self,
        path: Path,
        *,
        status: StreamStatus,
        player: PlaybackWorker | NullPlaybackWorker,
        emit: Callable[..., Any],
        interval: float = 0.5,
    ) -> None:
        self.path = path
        self.status = status
        self.player = player
        self.emit = emit
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed = False

    def write(self) -> None:
        try:
            phase, round_idx, take_idx, elapsed = self.status.snapshot()
            playing = self.player.current_path()
            payload = {
                "pid": os.getpid(),
                "updated_at": utc_now_iso(),
                "phase": phase,
                "round_idx": round_idx,
                "take_idx": take_idx,
                "phase_seconds": elapsed,
                # What the TUI pins on `p`; another front end hands it back as `pin_target` the same way.
                "playing": str(playing) if playing is not None else None,
                "buffer_seconds_ahead": self.player.buffer_seconds_ahead(),
            }
            write_text_atomic(self.path, json.dumps(payload, indent=2) + "\n")
        except Exception as exc:  # noqa: BLE001 — the status is a convenience, the session is not
            if not self._failed:
                self._failed = True
                self.emit("live_status_failed", f"{self.path}: {exc}", level="warning")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()

    def start(self) -> None:
        self.write()
        self._thread = threading.Thread(target=self._run, name="macchina_ricordi-stream-status", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.write()


def _print_round_status(
    *,
    label: str,
    elapsed: float,
    budget_seconds: float,
    buffer_ahead: float | None,
    quiet: bool = False,
) -> None:
    """One-line per-round diagnostic: generation time vs playback budget vs buffer headroom.

    ``quiet`` is for a UI that draws this itself from ``performance.jsonl`` and
    cannot have anything printing over its frame.
    """
    if quiet:
        return
    if buffer_ahead is None:
        buffer_part = "ahead=n/a (no playback)"
    else:
        buffer_part = f"ahead={buffer_ahead:+.1f}s"
    print(f"{label}: gen={elapsed:.1f}s budget={budget_seconds:.1f}s {buffer_part}")


def _effective_resume_threshold(
    max_buffer_ahead_seconds: float | None,
    resume_buffer_ahead_seconds: float | None,
    *,
    default_gap: float = 10.0,
) -> float | None:
    """Resolve the resume threshold relative to the throttle threshold.

    When the user passes only ``--max-buffer-ahead-seconds``, the resume
    threshold defaults to ``max - default_gap`` (clamped to ``[0, max]``).
    """
    if max_buffer_ahead_seconds is None:
        return None
    if resume_buffer_ahead_seconds is None:
        resume = max(0.0, float(max_buffer_ahead_seconds) - default_gap)
    else:
        resume = float(resume_buffer_ahead_seconds)
    return min(max(0.0, resume), float(max_buffer_ahead_seconds))


def _wait_for_buffer_room(
    *,
    player: PlaybackWorker | NullPlaybackWorker,
    control_path: Path,
    max_ahead: float,
    resume_ahead: float,
    poll_interval: float = 1.0,
    events: StreamEventLog | None = None,
) -> StreamControl | None:
    """Pause the generator while the playback buffer is at/above ``max_ahead``.

    The thresholds are re-read from ``control_path`` on every poll rather than
    frozen at entry: the buffer is what stands between a control edit and
    hearing it, so lowering the cap has to be able to land *during* the wait it
    is trying to shorten. Setting ``max_buffer_ahead_seconds`` to null resumes
    at once.

    Returns the latest ``StreamControl`` if a ``command: stop`` was observed
    during the wait, so the caller can break out immediately instead of
    spending another round of generation first. Returns ``None`` otherwise.
    """
    ahead = player.buffer_seconds_ahead()
    if ahead is None or ahead < max_ahead:
        return None

    emit = events.emit if events is not None else _print_event
    waited_from = time.monotonic()
    emit(
        "throttle",
        f"ahead={ahead:.1f}s >= max={max_ahead:.1f}s, waiting (resume at {resume_ahead:.1f}s)...",
        seconds_ahead=ahead,
        max_ahead=max_ahead,
        resume_ahead=resume_ahead,
    )
    while True:
        time.sleep(poll_interval)
        try:
            control = read_control(control_path)
        except (FileNotFoundError, ValueError):
            control = None
        if control is not None:
            if control.command == "stop":
                emit("throttle", "stop received during wait")
                return control
            new_max = control.max_buffer_ahead_seconds
            new_resume = _effective_resume_threshold(new_max, control.resume_buffer_ahead_seconds)
            if new_max is None or not control.throttle_enabled:
                # Opening the throttle mid-wait is the whole gesture for banking
                # a cushion before a model swap: it has to release the loop now,
                # not at the end of a drain that could be a minute long.
                emit(
                    "throttle",
                    "throttle switched off during wait; resuming",
                    seconds_ahead=player.buffer_seconds_ahead(),
                )
                return None
            if (float(new_max), new_resume) != (max_ahead, resume_ahead):
                emit(
                    "throttle_retuned",
                    f"max {max_ahead:.1f}s -> {new_max:.1f}s, resume {resume_ahead:.1f}s -> {new_resume:.1f}s",
                    max_ahead=float(new_max),
                    resume_ahead=new_resume,
                )
                max_ahead, resume_ahead = float(new_max), float(new_resume or 0.0)
        ahead = player.buffer_seconds_ahead()
        if ahead is None or ahead <= resume_ahead:
            ahead_str = f"{ahead:.1f}s" if ahead is not None else "n/a"
            emit(
                "throttle",
                f"resumed (ahead={ahead_str})",
                seconds_ahead=ahead,
                waited_seconds=round(time.monotonic() - waited_from, 1),
            )
            return None


def _rest_before_round(
    *,
    player: PlaybackWorker | NullPlaybackWorker,
    control_path: Path,
    delay: float,
    refill_seconds: float,
    poll_interval: float = 0.1,
    events: StreamEventLog | None = None,
) -> StreamControl | None:
    """Sleep ``delay`` seconds before a round, unless the buffer would run dry.

    The delay is re-read on every poll, so shortening it ends a rest in
    progress. The rest is cut once the buffer holds no more than
    ``refill_seconds`` (what the last round took to make): waiting longer would
    leave the player silent before the next round arrives.

    Returns the latest ``StreamControl`` if a ``command: stop`` arrived, like
    ``_wait_for_buffer_room``; ``None`` otherwise.
    """
    started = time.monotonic()
    while (remaining := delay - (time.monotonic() - started)) > 0:
        ahead = player.buffer_seconds_ahead()
        if ahead is not None and ahead <= refill_seconds:
            emit = events.emit if events is not None else _print_event
            emit(
                "rest_cut",
                f"buffer {ahead:.1f}s ≤ refill {refill_seconds:.1f}s; generating now",
                seconds_ahead=ahead,
                refill_seconds=round(refill_seconds, 2),
            )
            return None
        time.sleep(min(poll_interval, remaining))
        try:
            control = read_control(control_path)
        except (FileNotFoundError, ValueError):
            continue
        if control.command == "stop":
            return control
        delay = control.round_delay_seconds
    return None


def _seed_for(base_seed: int | None, round_idx: int, take_idx: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + 1009 * round_idx + 17 * take_idx


def _motif_dir_for(control: StreamControl, paths: StreamSessionPaths) -> Path:
    if control.motif_dir:
        return Path(control.motif_dir).expanduser()
    return paths.session_dir / "motifs"


def _pin_source_path(
    control: StreamControl,
    *,
    player: PlaybackWorker | NullPlaybackWorker,
    fallback: Path,
    session_dir: Path,
) -> Path:
    """Resolve the audio round captured by a one-shot pin command.

    The TUI records the playback path when ``p`` is pressed. Hand-edited
    commands do not have that metadata, so they query the playback worker when
    the loop observes the command and finally fall back to the continuation
    source when playback is disabled.
    """
    if control.pin_target:
        target = Path(control.pin_target).expanduser()
        if not target.is_absolute():
            target = session_dir / target
        if target.is_file():
            return target

    current = player.current_path()
    if current is not None and current.is_file():
        return current
    return fallback


def _list_motif_files(motif_dir: Path) -> list[Path]:
    if not motif_dir.is_dir():
        return []
    return sorted(motif_dir.glob("*.wav"))


def _latest_motif_file(motif_dir: Path) -> Path | None:
    candidates: list[tuple[int, str, Path]] = []
    for path in _list_motif_files(motif_dir):
        try:
            candidates.append((path.stat().st_mtime_ns, path.name, path))
        except OSError:
            continue
    return max(candidates, default=None)[2] if candidates else None


def _render_mixed_prompt(
    prev_path: Path,
    motif_path: Path,
    *,
    tau_seconds: float,
    mix: float,
    out_path: Path,
    prev_start_seconds: float | None = None,
) -> tuple[float, float]:
    """Blend the current tail with a motif's tail into a prompt WAV.

    Both tails are end-aligned (the model continues from the *end* of the
    prompt). A shorter motif tail is zero-padded at the front.
    ``prev_start_seconds`` overrides where the current-audio window is taken
    from, so a rescued window survives a recap round. Returns
    ``(prompt_start_seconds, prompt_duration_seconds)`` for ``out_path``.
    """
    import librosa
    import numpy as np

    prev_start, prev_tau = extract_tail(prev_path, tau_seconds)
    if prev_start_seconds is not None:
        prev_start = float(prev_start_seconds)
    current, sr = librosa.load(prev_path, sr=None, mono=True, offset=prev_start, duration=prev_tau)
    sr = int(sr)
    if current.size == 0:
        raise ValueError(f"Empty continuation tail from {prev_path}")

    motif, _ = librosa.load(motif_path, sr=sr, mono=True)
    if motif.size == 0:
        raise ValueError(f"Empty motif audio: {motif_path}")
    motif_tail = motif[-current.size :]
    if motif_tail.size < current.size:
        motif_tail = np.concatenate([np.zeros(current.size - motif_tail.size, dtype=motif_tail.dtype), motif_tail])

    blend = (1.0 - mix) * current + mix * motif_tail
    peak = float(np.max(np.abs(blend))) if blend.size else 0.0
    if peak > 0.99:
        blend = blend * (0.99 / peak)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(blend, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.tobytes())
    return 0.0, blend.size / float(sr)


def _build_model_source(
    *,
    model_id: str | None,
    device: str | None,
    profile_name: str | None,
    xp_path: str | Path | None,
    checkpoint: str | None,
    checkpoint_path: str | Path | None,
) -> StreamModelSource:
    resolved_checkpoint = resolve_generation_checkpoint(
        xp_path=xp_path,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    resolved_profile = resolve_generation_profile(
        profile_name=profile_name,
        xp_path=xp_path,
        checkpoint_path=resolved_checkpoint,
    )
    resolved_xp = Path(xp_path).expanduser().resolve() if xp_path is not None else None
    return StreamModelSource(
        model_id=model_id or DEFAULT_STREAM_MODEL_ID,
        device=device,
        profile_name=resolved_profile,
        xp_path=str(resolved_xp) if resolved_xp is not None else None,
        checkpoint_path=str(resolved_checkpoint) if resolved_checkpoint is not None else None,
    )


def normalize_model_xp_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _segment_duration_for(model: str | None) -> float | None:
    """The trained segment length for a model source, bundle-aware.

    A bundle carries its own trained ``segment_duration`` in its manifest; a
    Dora XP records it in ``.argv.json`` (or the stats corpus). Falls back to
    the shared resolver for everything else.
    """
    if model in (None, ""):
        return None
    try:
        from macchina_ricordi.models.bundle_runtime import bundle_segment_duration

        bundled = bundle_segment_duration(model)
    except Exception:  # noqa: BLE001 - a bundle that cannot be read is not a model source here
        bundled = None
    if bundled is not None:
        return float(bundled)
    return resolve_segment_duration(model)


def _write_manifest(
    paths: StreamSessionPaths,
    *,
    initial_control: StreamControl,
    backend_config: dict[str, Any],
    seed_mode: str,
    seed_audio_source: str | None,
    base_seed: int | None,
    started_at: str,
    finished_at: str,
    rounds_written: int,
    stopped_by: str,
    score_rounds: bool,
    max_rounds: int,
    final_xp_path: str | None = None,
    swap_count: int = 0,
    max_buffer_ahead_seconds: float | None = None,
    resume_buffer_ahead_seconds: float | None = None,
    throttle_enabled: bool = True,
    novelty_calibration: dict[str, Any] | None = None,
    segment_duration: float | None = None,
) -> None:
    manifest = {
        "schema": "macchina_ricordi.stream_session.v1",
        "run_id": paths.run_id,
        "session_dir": str(paths.session_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "rounds_written": rounds_written,
        "stopped_by": stopped_by,
        "seed_mode": seed_mode,
        "seed_audio_source": seed_audio_source,
        "base_seed": base_seed,
        "max_rounds": max_rounds,
        "score_rounds": score_rounds,
        "initial_control": asdict(initial_control),
        "backend": backend_config,
        "final_xp_path": final_xp_path,
        # The final model's trained horizon: which durations this session's
        # rounds were within, and which overran, is not recoverable without it.
        "segment_duration": segment_duration,
        "swap_count": int(swap_count),
        # The thresholds as the session *ended* on them — they are live controls,
        # so these can differ from `initial_control`.
        "throttle": {
            "max_buffer_ahead_seconds": max_buffer_ahead_seconds,
            "resume_buffer_ahead_seconds": resume_buffer_ahead_seconds,
            # False means the session ended with the throttle held open, which
            # is what a runaway buffer trace in `performance.jsonl` means.
            "enabled": bool(throttle_enabled),
        },
        # Which population novelty was measured against. Without this a
        # session's novelty numbers are not interpretable later.
        "novelty_calibration": novelty_calibration,
        "performance_log": paths.performance_path.name,
        "events_log": paths.events_path.name,
        "rounds_dir": paths.rounds_dir.name,
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_stream_session(
    *,
    paths: StreamSessionPaths,
    initial_control: StreamControl,
    seed_mode: Literal["uncond", "excerpt"],
    seed_audio: Path | None,
    model_id: str | None,
    device: str | None,
    profile_name: str | None,
    xp_path: str | Path | None,
    checkpoint: str | None,
    checkpoint_path: str | Path | None,
    base_seed: int | None,
    max_rounds: int,
    score_rounds: bool,
    player: PlaybackWorker | NullPlaybackWorker,
    generator: StreamGenerator | None = None,
    max_buffer_ahead_seconds: float | None = None,
    resume_buffer_ahead_seconds: float | None = None,
    throttle_poll_interval: float = 1.0,
    status: StreamStatus | None = None,
    quiet: bool = False,
    novelty_calibration: NoveltyCalibration | None = None,
) -> StreamSessionResult:
    if seed_mode not in SEED_MODES:
        raise ValueError(f"Unsupported seed_mode '{seed_mode}'. Expected one of: {SEED_MODES}")
    if seed_mode == "excerpt" and seed_audio is None:
        raise ValueError("seed_mode='excerpt' requires seed_audio.")

    # The throttle thresholds live in the control file so they can be retuned
    # mid-performance; these arguments are only their initial value, and lose to
    # a control that already carries them.
    if initial_control.max_buffer_ahead_seconds is None and max_buffer_ahead_seconds is not None:
        initial_control = replace(
            initial_control,
            max_buffer_ahead_seconds=float(max_buffer_ahead_seconds),
            resume_buffer_ahead_seconds=_effective_resume_threshold(
                max_buffer_ahead_seconds, resume_buffer_ahead_seconds
            ),
        )

    paths.session_dir.mkdir(parents=True, exist_ok=True)
    paths.rounds_dir.mkdir(parents=True, exist_ok=True)
    write_control(paths.control_path, initial_control)

    # A TUI owns the terminal, so nothing may print behind its back; everything
    # that would have been a line of stdout is already an event.
    events = StreamEventLog(paths.events_path, echo=not quiet)
    player.set_event_sink(events.emit)
    tracker = status if status is not None else StreamStatus()
    live_status = LiveStatusWriter(
        paths.session_dir / LIVE_STATUS_FILENAME, status=tracker, player=player, emit=events.emit
    )

    def consume(current: StreamControl) -> StreamControl:
        return consume_command(paths.control_path, current)

    started_at = _now_iso()

    active_xp_path = normalize_model_xp_path(xp_path)
    active_profile: str | None = profile_name
    active_checkpoint: str | None = checkpoint
    active_checkpoint_path: str | Path | None = checkpoint_path
    swap_count = 0
    throttle_max = initial_control.max_buffer_ahead_seconds
    throttle_resume = _effective_resume_threshold(
        throttle_max,
        initial_control.resume_buffer_ahead_seconds,
    )
    throttle_enabled = initial_control.throttle_enabled
    active_source = _build_model_source(
        model_id=model_id,
        device=device,
        profile_name=active_profile,
        xp_path=active_xp_path,
        checkpoint=active_checkpoint,
        checkpoint_path=active_checkpoint_path,
    )
    segment_duration = _segment_duration_for(active_xp_path)

    backend_config = {
        "model_id": model_id,
        "device": device,
        "profile_name": profile_name,
        "xp_path": active_xp_path,
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    owns_generator = generator is None
    if generator is None:
        generator = PersistentStreamGenerator(stderr_path=paths.session_dir / "worker.log" if quiet else None)

    novelty_memory: deque[Any] = deque(maxlen=NOVELTY_MEMORY_ROUNDS)
    drone_memory: deque[float | None] = deque(maxlen=8)
    rounds_since_kick = drone_memory.maxlen

    live_status.start()
    try:
        # ---- Round 0 (seed) -----------------------------------------------------
        seed_path = paths.rounds_dir / _round_filename(0, 0)
        tracker.set_phase("generating", round_idx=0)
        seed_t0 = time.monotonic()
        seed_score: dict[str, Any] | None = None
        seed_audio_source: str | None = None

        if seed_mode == "excerpt":
            assert seed_audio is not None
            seed_audio_source = str(seed_audio)
            shutil.copyfile(seed_audio, seed_path)
        else:
            request = GenerationRequest(
                prompts=[""],
                batch_size=1,
                mode="unconditional",
                duration_seconds=initial_control.completion_duration_seconds,
                seed=_seed_for(base_seed, 0, 0),
                use_sampling=True,
                top_k=initial_control.top_k,
                top_p=initial_control.top_p,
                min_p=initial_control.min_p,
                temperature=initial_control.temperature,
                cfg_coef=initial_control.cfg_coef,
                extend_stride=initial_control.extend_stride,
            )
            generator.generate(request, output_path=seed_path, source=active_source)

        # Fingerprint every round, not only when the blend is on: novelty is a
        # reading before it is a control, and the memory must be unbroken.
        seed_vector = _candidate_feature_vector(seed_path)
        seed_novelty = novelty_readings(seed_vector, list(novelty_memory), calibration=novelty_calibration)
        if seed_vector is not None:
            novelty_memory.append(seed_vector)

        seed_elapsed = time.monotonic() - seed_t0
        try:
            seed_duration = _wav_duration_seconds(seed_path)
        except (wave.Error, OSError, ValueError):
            seed_duration = initial_control.completion_duration_seconds

        if score_rounds:
            seed_score = _score_round(seed_path)
        seed_drone = _drone_readings_of(seed_path) or {}
        drone_memory.append(seed_drone.get("drone_score"))
        seed_score = {**(seed_score or {}), **seed_novelty, **seed_drone}

        seed_playback_seconds = float(seed_duration)
        player.enqueue(seed_path, skip_seconds=0.0, playback_seconds=seed_playback_seconds)
        seed_buffer_ahead = player.buffer_seconds_ahead()
        _print_round_status(
            quiet=quiet,
            label="seed",
            elapsed=seed_elapsed,
            budget_seconds=initial_control.completion_duration_seconds,
            buffer_ahead=seed_buffer_ahead,
        )
        append_performance_row(
            paths.performance_path,
            RoundRecord(
                round_idx=0,
                take_idx=0,
                timestamp=_now_iso(),
                command_consumed="seed",
                active_control=asdict(initial_control),
                seed_used=_seed_for(base_seed, 0, 0),
                round_wav=seed_path.relative_to(paths.session_dir).as_posix(),
                duration_seconds=seed_duration,
                elapsed_seconds=seed_elapsed,
                score=seed_score,
                model_xp=active_xp_path,
                playback_seconds=seed_playback_seconds,
                buffer_seconds_ahead=seed_buffer_ahead,
            ),
        )

        events.emit(
            "novelty_calibration",
            (
                f"whitened against {novelty_calibration.source}"
                if novelty_calibration is not None
                else "none loaded — novelty is raw cosine, which has almost no dynamic range"
            ),
            level="info" if novelty_calibration is not None else "warning",
            round_idx=0,
            **(novelty_calibration.describe() if novelty_calibration is not None else {}),
        )
        events.emit(
            "session_start",
            f"{paths.run_id} seed_mode={seed_mode} model_xp={active_xp_path}",
            round_idx=0,
            run_id=paths.run_id,
            seed_mode=seed_mode,
            model_xp=active_xp_path,
        )

        rounds_written = 1
        stopped_by: Literal["command", "max_rounds", "interrupt"] = "max_rounds"

        # ---- Rounds 1..N (continuation) ----------------------------------------
        prev_path = seed_path
        prev_prompt_duration = 0.0  # the seed round is all fresh audio
        round_idx = 1
        take_idx = 0
        # Last tau the horizon guard clamped to, so a tau parked above the
        # ceiling reports once instead of filling the events box every round.
        horizon_clamped_to: float | None = None
        rate_clamped_to: float | None = None
        refill_seconds = seed_elapsed

        try:
            while max_rounds <= 0 or round_idx <= max_rounds:
                events.round_idx = round_idx
                tracker.set_phase("waiting", round_idx=round_idx, take_idx=take_idx)
                control = read_control(paths.control_path)
                command_consumed = control.command
                if control.command == "stop":
                    consume(control)
                    stopped_by = "command"
                    break

                throttle_max = control.max_buffer_ahead_seconds
                throttle_resume = _effective_resume_threshold(throttle_max, control.resume_buffer_ahead_seconds)
                throttle_enabled = control.throttle_enabled
                if throttle_enabled and throttle_max is not None and throttle_resume is not None:
                    tracker.set_phase("throttled", round_idx=round_idx, take_idx=take_idx)
                    throttle_stop = _wait_for_buffer_room(
                        player=player,
                        control_path=paths.control_path,
                        max_ahead=float(throttle_max),
                        resume_ahead=throttle_resume,
                        poll_interval=throttle_poll_interval,
                        events=events,
                    )
                    if throttle_stop is not None:
                        consume(throttle_stop)
                        stopped_by = "command"
                        break
                    # User may have edited other control fields during the wait,
                    # the thresholds among them.
                    control = read_control(paths.control_path)
                    throttle_max = control.max_buffer_ahead_seconds
                    throttle_resume = _effective_resume_threshold(throttle_max, control.resume_buffer_ahead_seconds)
                    throttle_enabled = control.throttle_enabled

                if control.round_delay_seconds > 0:
                    tracker.set_phase("resting", round_idx=round_idx, take_idx=take_idx)
                    rest_stop = _rest_before_round(
                        player=player,
                        control_path=paths.control_path,
                        delay=control.round_delay_seconds,
                        refill_seconds=refill_seconds,
                        events=events,
                    )
                    if rest_stop is not None:
                        consume(rest_stop)
                        stopped_by = "command"
                        break
                    # The rest is for setting this round's knobs, so read them again.
                    control = read_control(paths.control_path)

                # Live model swap: reload the persistent worker only when the source changes.
                requested_xp_path = normalize_model_xp_path(control.model_xp)
                if requested_xp_path and requested_xp_path != active_xp_path:
                    requested_xp = Path(requested_xp_path)
                    if not requested_xp.exists() or not requested_xp.is_dir():
                        events.emit(
                            "model_swap_skipped",
                            f"{requested_xp} does not exist; keeping {active_xp_path}",
                            level="warning",
                            requested=str(requested_xp),
                        )
                    else:
                        events.emit(
                            "model_swap",
                            f"{active_xp_path} -> {requested_xp}",
                            previous=active_xp_path,
                            current=str(requested_xp),
                        )
                        active_xp_path = requested_xp_path
                        active_profile = None  # force re-detect from the new XP
                        active_checkpoint = "latest"
                        active_checkpoint_path = None
                        active_source = _build_model_source(
                            model_id=model_id,
                            device=device,
                            profile_name=active_profile,
                            xp_path=active_xp_path,
                            checkpoint=active_checkpoint,
                            checkpoint_path=active_checkpoint_path,
                        )
                        segment_duration = _segment_duration_for(active_xp_path)
                        swap_count += 1

                # Live motif capture: copy the round the listener just heard
                # into the motif bank, then carry on generating as usual.
                if control.command == "pin":
                    pin_source = _pin_source_path(
                        control,
                        player=player,
                        fallback=prev_path,
                        session_dir=paths.session_dir,
                    )
                    pin_dir = _motif_dir_for(control, paths)
                    pin_dir.mkdir(parents=True, exist_ok=True)
                    pin_dest = pin_dir / f"motif_{paths.run_id}_{pin_source.stem}.wav"
                    shutil.copyfile(pin_source, pin_dest)
                    events.emit(
                        "pinned",
                        f"{pin_source.name} -> {pin_dest}",
                        source=pin_source.name,
                        destination=str(pin_dest),
                    )
                    control = consume(control)

                prompt_source_path = prev_path
                motif_source: str | None = None
                motif_injection_weight: float | None = None
                applied_rate: float | None = None
                applied_tilt: dict[str, float] | None = None
                # Hard horizon guard: a prompt that reaches the model's trained
                # segment length aborts the round inside AudioCraft, and tau is a
                # live keyboard control. Clamp before anything reads it.
                tau_seconds = clamp_prompt_seconds(control.tau_seconds, segment_duration=segment_duration)
                if tau_seconds < control.tau_seconds and tau_seconds != horizon_clamped_to:
                    events.emit(
                        "horizon_guard",
                        f"tau {control.tau_seconds:.1f}s would reach this model's "
                        f"{segment_duration:g}s horizon; prompting from {tau_seconds:.1f}s instead",
                        level="warning",
                        requested_tau=round(float(control.tau_seconds), 2),
                        clamped_tau=round(float(tau_seconds), 2),
                        segment_duration=segment_duration,
                    )
                horizon_clamped_to = tau_seconds if tau_seconds < control.tau_seconds else None
                prompt_start, prompt_duration = extract_tail(prev_path, tau_seconds)
                rescued_from: float | None = None
                if control.tail_rescue:
                    # Never back off into the previous round's own prompt echo:
                    # that audio was already a prompt, and re-using it is the
                    # déjà-vu loop novelty_weight exists to fight.
                    rescued_start = _rescue_prompt_start(
                        prev_path,
                        tau_seconds=tau_seconds,
                        min_start_seconds=prev_prompt_duration,
                    )
                    if rescued_start is not None:
                        events.emit(
                            "tail_rescue",
                            f"{prev_path.name} tail is silent; "
                            f"re-prompting from {rescued_start:.1f}s (was {prompt_start:.1f}s)",
                            level="warning",
                            rescued_to=round(rescued_start, 2),
                            rescued_from=round(prompt_start, 2),
                        )
                        rescued_from = prompt_start
                        prompt_start = rescued_start
                if control.command in ("inject", "inject_light", "inject_heavy"):
                    injection_command = control.command
                    injection_weight = {
                        "inject": PINNED_INJECTION_WEIGHT,
                        "inject_light": PINNED_INJECTION_LIGHT_WEIGHT,
                        "inject_heavy": PINNED_INJECTION_HEAVY_WEIGHT,
                    }[injection_command]
                    motif_path = _latest_motif_file(_motif_dir_for(control, paths))
                    control = consume(control)
                    if motif_path is None:
                        events.emit(
                            "inject_skipped",
                            "no pinned motif is available",
                            level="warning",
                            weight=injection_weight,
                        )
                    else:
                        mixed_path = paths.session_dir / "prompts" / f"prompt_{round_idx:03d}_{injection_command}.wav"
                        try:
                            prompt_start, prompt_duration = _render_mixed_prompt(
                                prev_path,
                                motif_path,
                                tau_seconds=tau_seconds,
                                mix=injection_weight,
                                out_path=mixed_path,
                                prev_start_seconds=prompt_start,
                            )
                        except Exception as exc:  # noqa: BLE001 — injection is optional, never fatal
                            events.emit(
                                "inject_skipped",
                                f"{type(exc).__name__}: {exc}",
                                level="warning",
                                motif=str(motif_path),
                                weight=injection_weight,
                            )
                        else:
                            prompt_source_path = mixed_path
                            motif_source = str(motif_path)
                            motif_injection_weight = injection_weight
                            events.emit(
                                "pinned_injection",
                                f"{motif_path.name} mix={injection_weight:.2f} for next round only",
                                motif=motif_path.name,
                                mix=injection_weight,
                            )
                elif control.motif_period > 0 and take_idx == 0 and round_idx % control.motif_period == 0:
                    motif_files = _list_motif_files(_motif_dir_for(control, paths))
                    if motif_files:
                        rng = random.Random(f"{base_seed}:{round_idx}")
                        motif_path = rng.choice(motif_files)
                        mixed_path = paths.session_dir / "prompts" / f"prompt_{round_idx:03d}.wav"
                        try:
                            prompt_start, prompt_duration = _render_mixed_prompt(
                                prev_path,
                                motif_path,
                                tau_seconds=tau_seconds,
                                mix=control.motif_mix,
                                out_path=mixed_path,
                                prev_start_seconds=prompt_start,
                            )
                        except Exception as exc:  # noqa: BLE001 — recap is optional, never fatal
                            events.emit(
                                "motif_skipped",
                                f"{type(exc).__name__}: {exc}",
                                level="warning",
                                motif=str(motif_path),
                            )
                        else:
                            prompt_source_path = mixed_path
                            motif_source = str(motif_path)
                            events.emit(
                                "motif_recap",
                                f"{motif_path.name} mix={control.motif_mix:.2f}",
                                motif=motif_path.name,
                                mix=control.motif_mix,
                            )

                # Prompt transforms, applied last: whatever the prompt turned
                # out to be — plain tail, rescued window or motif blend — is
                # what gets varisped and tilted, so what the model hears is what
                # the keys say. Never fatal; a failure keeps the plain window.
                tilt = PromptTilt(
                    low_db=control.prompt_tilt_low_db,
                    mid_db=control.prompt_tilt_mid_db,
                    high_db=control.prompt_tilt_high_db,
                )
                rate = clamp_prompt_rate(control.prompt_rate)
                if not tilt.is_flat or abs(rate - 1.0) > 1e-9:
                    shaped_path = paths.session_dir / "prompts" / f"prompt_{round_idx:03d}_{take_idx}_shaped.wav"
                    try:
                        transform = render_prompt_transform(
                            prompt_source_path,
                            shaped_path,
                            start_seconds=prompt_start,
                            duration_seconds=prompt_duration,
                            rate=rate,
                            mode=control.prompt_rate_mode,
                            tilt=tilt,
                        )
                    except Exception as exc:  # noqa: BLE001 — steering is optional, never fatal
                        events.emit(
                            "prompt_transform_skipped",
                            f"{type(exc).__name__}: {exc}",
                            level="warning",
                            rate=rate,
                            tilt=tilt.to_dict(),
                        )
                    else:
                        prompt_source_path = shaped_path
                        prompt_start = 0.0
                        prompt_duration = transform.duration_seconds
                        applied_rate = transform.rate_applied
                        applied_tilt = None if tilt.is_flat else tilt.to_dict()
                        # A rate above 1 needs more tail than the plain window;
                        # when the round could not reach that far back the model
                        # is being driven at a rate nobody asked for, so say so
                        # — once, since a parked rate would fill the events box.
                        if transform.rate_was_clamped and applied_rate != rate_clamped_to:
                            events.emit(
                                "prompt_rate_clamped",
                                f"prompt_rate {rate:.3f} needs "
                                f"{rate * prompt_duration:.1f}s of tail; "
                                f"only {applied_rate * prompt_duration:.1f}s exists, "
                                f"so this round runs at {applied_rate:.3f}",
                                level="warning",
                                requested=round(rate, 4),
                                applied=round(applied_rate, 4),
                            )
                        rate_clamped_to = applied_rate if transform.rate_was_clamped else None

                temperature, top_k = control.temperature, control.top_k
                kick = stuck_kick(control, drone_memory, rounds_since_kick)
                if kick is not None:
                    temperature, top_k = kick
                    rounds_since_kick = 0
                    events.emit(
                        "stuck_kick",
                        f"{control.stuck_kick_rounds} rounds stuck on one note; this round runs at "
                        f"temperature {temperature:.2f}, top_k {top_k}",
                        level="warning",
                        temperature=round(temperature, 3),
                        top_k=top_k,
                        drone_score=[round(d, 2) for d in list(drone_memory)[-control.stuck_kick_rounds :]],
                    )

                seed_value = _seed_for(base_seed, round_idx, take_idx)
                num_candidates = max(1, min(MAX_STREAM_CANDIDATES, int(control.candidates)))
                request = GenerationRequest(
                    prompts=[""],
                    batch_size=num_candidates,
                    mode="continuation",
                    duration_seconds=prompt_duration + control.completion_duration_seconds,
                    seed=seed_value,
                    prompt_audio_path=str(prompt_source_path),
                    prompt_start_seconds=prompt_start,
                    prompt_duration_seconds=prompt_duration,
                    use_sampling=True,
                    top_k=top_k,
                    top_p=control.top_p,
                    min_p=control.min_p,
                    temperature=temperature,
                    cfg_coef=control.cfg_coef,
                    extend_stride=control.extend_stride,
                )

                tracker.set_phase("generating", round_idx=round_idx, take_idx=take_idx)
                t0 = time.monotonic()
                round_path = paths.rounds_dir / _round_filename(round_idx, take_idx)
                generation = generator.generate(request, output_path=round_path, source=active_source)
                elapsed = time.monotonic() - t0

                tracker.set_phase("scoring", round_idx=round_idx, take_idx=take_idx)
                score_payload: dict[str, Any] | None = None
                winner_vector: Any | None = None
                winner_drone: dict[str, float] | None = None
                candidate_paths = [Path(p) for p in generation.candidate_paths]
                if len(candidate_paths) > 1:
                    winner, score_payload, candidate_summary, winner_vector = _select_best_candidate(
                        candidate_paths,
                        prompt_duration_seconds=prompt_duration,
                        novelty_weight=control.novelty_weight,
                        recent_vectors=list(novelty_memory),
                        calibration=novelty_calibration,
                        stuck_weight=control.stuck_weight,
                    )
                    winner_drone = {
                        key: candidate_summary[winner][key]
                        for key in ("stationarity", "chroma_top", "drone_score")
                        if key in candidate_summary[winner]
                    } or None
                    if candidate_paths[winner] != round_path:
                        os.replace(candidate_paths[winner], round_path)
                    for i, candidate in enumerate(candidate_paths):
                        if i != winner and candidate != round_path:
                            candidate.unlink(missing_ok=True)
                    score_payload = dict(score_payload or {})
                    score_payload["candidates"] = candidate_summary

                if winner_vector is None:
                    winner_vector = _candidate_feature_vector(round_path, prompt_duration_seconds=prompt_duration)
                round_novelty = novelty_readings(winner_vector, list(novelty_memory), calibration=novelty_calibration)
                if winner_vector is not None:
                    novelty_memory.append(winner_vector)
                # Every round, so the kick sees consecutive rounds whatever candidates is.
                if winner_drone is None:
                    winner_drone = _drone_readings_of(round_path, prompt_duration_seconds=prompt_duration)
                winner_drone = winner_drone or {}
                drone_memory.append(winner_drone.get("drone_score"))
                round_novelty = {**round_novelty, **winner_drone}

                try:
                    sample_duration = _wav_duration_seconds(round_path)
                except (wave.Error, OSError, ValueError):
                    sample_duration = prompt_duration + control.completion_duration_seconds

                completion_duration = control.completion_duration_seconds
                if completion_duration > 0 and elapsed > completion_duration:
                    events.emit(
                        "generation_slow_warning",
                        f"round {round_idx} took {elapsed:.1f}s but completion duration is "
                        f"{completion_duration:.1f}s. The audio buffer will starve. "
                        "Increase --completion-duration or decrease --tau.",
                        level="warning",
                        elapsed_seconds=round(elapsed, 2),
                        completion_duration=round(completion_duration, 2),
                    )

                if score_rounds and score_payload is None:
                    score_payload = _score_round(round_path, prompt_duration_seconds=prompt_duration)
                score_payload = {**(score_payload or {}), **round_novelty}

                playback_seconds = max(0.0, float(sample_duration) - float(prompt_duration))
                player.enqueue(round_path, skip_seconds=prompt_duration, playback_seconds=playback_seconds)
                refill_seconds = time.monotonic() - t0
                buffer_ahead = player.buffer_seconds_ahead()
                _print_round_status(
                    quiet=quiet,
                    label=f"round {round_idx}" + (f" take {take_idx}" if take_idx else ""),
                    elapsed=elapsed,
                    budget_seconds=completion_duration,
                    buffer_ahead=buffer_ahead,
                )
                append_performance_row(
                    paths.performance_path,
                    RoundRecord(
                        round_idx=round_idx,
                        take_idx=take_idx,
                        timestamp=_now_iso(),
                        command_consumed=command_consumed,
                        active_control=asdict(control),
                        seed_used=seed_value,
                        round_wav=round_path.relative_to(paths.session_dir).as_posix(),
                        duration_seconds=sample_duration,
                        elapsed_seconds=elapsed,
                        score=score_payload,
                        prompt_start_seconds=prompt_start,
                        prompt_duration_seconds=prompt_duration,
                        prompt_audio=prompt_source_path.relative_to(paths.session_dir).as_posix(),
                        motif_source=motif_source,
                        motif_injection_weight=motif_injection_weight,
                        tail_rescued_from_seconds=rescued_from,
                        prompt_rate=applied_rate,
                        prompt_rate_mode=control.prompt_rate_mode if applied_rate is not None else None,
                        prompt_tilt_db=applied_tilt,
                        model_xp=active_xp_path,
                        playback_seconds=playback_seconds,
                        buffer_seconds_ahead=buffer_ahead,
                        stuck_kick=({"temperature": temperature, "top_k": top_k} if kick is not None else None),
                    ),
                )
                rounds_written += 1
                rounds_since_kick += 1

                consume(control)
                if control.command == "reroll":
                    # Same prev_path, fresh take.
                    take_idx += 1
                    continue

                prev_path = round_path
                prev_prompt_duration = float(prompt_duration)
                round_idx += 1
                take_idx = 0
        except KeyboardInterrupt:
            stopped_by = "interrupt"

        tracker.set_phase("stopped")
        events.emit(
            "session_end",
            f"stopped_by={stopped_by} rounds_written={rounds_written}",
            stopped_by=stopped_by,
            rounds_written=rounds_written,
        )
        player.enqueue(None)
        player.join(timeout=10.0)

        _write_manifest(
            paths,
            initial_control=initial_control,
            backend_config=backend_config,
            seed_mode=seed_mode,
            seed_audio_source=seed_audio_source,
            base_seed=base_seed,
            started_at=started_at,
            finished_at=_now_iso(),
            rounds_written=rounds_written,
            stopped_by=stopped_by,
            score_rounds=score_rounds,
            max_rounds=max_rounds,
            final_xp_path=active_xp_path,
            swap_count=swap_count,
            max_buffer_ahead_seconds=throttle_max,
            resume_buffer_ahead_seconds=throttle_resume,
            throttle_enabled=throttle_enabled,
            novelty_calibration=(novelty_calibration.describe() if novelty_calibration is not None else None),
            segment_duration=segment_duration,
        )

        return StreamSessionResult(
            paths=paths,
            rounds_written=rounds_written,
            stopped_by=stopped_by,
        )
    finally:
        live_status.stop()
        if owns_generator:
            generator.close()


def fresh_run_id() -> str:
    return utc_compact_now()
