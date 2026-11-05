"""Single-pane terminal UI for a live `play stream` session.

The generator process draws its own dashboard and takes keyboard input, so one
terminal both shows the state and changes it — no second window, no editing JSON
mid-performance.

Three things shape the design:

- **The frame is read from disk.** Everything drawn comes from
  ``stream_monitor.read_snapshot`` / ``render_frame`` on the session's own
  files, so there is exactly one renderer for the TUI and for the read-only
  ``play monitor``. Only the in-flight round — which the loop has not
  written down yet — comes from memory, via `StreamStatus`.
- **`control.json` stays authoritative.** Keys do a read-modify-write of the
  same file an external editor would touch, so the round loop's control
  handling is untouched and there is no second source of truth.
- **A performance must survive the UI.** Terminal state is restored from a
  ``finally``, from ``atexit`` and from a signal handler; every draw and every
  keystroke is wrapped so a rendering bug cannot take the audio down with it.
"""

from __future__ import annotations

import atexit
import os
import select
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, TextIO

from ..controls import TAU, TEMPERATURE, TOP_K
from ..prompt_transform import MAX_PROMPT_RATE, MAX_TILT_DB, MIN_PROMPT_RATE
from .monitor import FOLLOW_GENERATED, FOLLOW_PLAYING, read_snapshot, render_frame
from .session import (
    MAX_COMPLETION_DURATION_SECONDS,
    MAX_ROUND_DELAY_SECONDS,
    MAX_STREAM_CANDIDATES,
    MIN_COMPLETION_DURATION_SECONDS,
    PINNED_INJECTION_HEAVY_WEIGHT,
    PINNED_INJECTION_LIGHT_WEIGHT,
    StreamControl,
    StreamStatus,
    control_file_lock,
    normalize_model_xp_path,
    read_control,
    write_control,
)

# Escape sequences. Alt-screen so the session's scrollback comes back on exit,
# unlike the monitor, which is attached from a shell the user still wants.
_ENTER_TUI = "\033[?1049h\033[?25l"
_EXIT_TUI = "\033[?25h\033[?1049l"
_HOME_CLEAR = "\033[H\033[J"

# ── in-flight status ──────────────────────────────────────────────────


class PlaybackTarget(Protocol):
    def current_path(self) -> Path | None: ...


# ── key decoding ──────────────────────────────────────────────────────

_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}


def _unfinished_escape_start(data: str) -> int:
    """Where a trailing escape sequence that the next read may still finish begins; ``len(data)`` if none.

    One read returns at most 64 bytes, so a sequence can straddle two reads. A tail longer than 16
    characters is junk, not a sequence, so it is dropped rather than held.
    """
    start = data.rfind("\033")
    if start < 0 or len(data) - start > 16:
        return len(data)
    tail = data[start:]
    if tail in ("\033", "\033O"):
        return start
    if tail.startswith("\033[") and not any("@" <= char <= "~" for char in tail[2:]):
        return start
    return len(data)


def decode_keys(data: str) -> list[str]:
    """Split a raw read from the terminal into symbolic key names.

    Every key in the buffer is returned, not just the first: holding an arrow
    down is how you sweep a knob, and the terminal delivers those repeats in
    bursts, so dropping all but one would make a held key feel broken.

    Arrow keys arrive as ``ESC [ A`` (or ``ESC O A`` in application mode). A
    bare ``ESC``, or any other escape sequence, is dropped whole rather than
    waited on — a stray escape can never wedge the input loop, nor leak its tail.
    """
    keys: list[str] = []
    index = 0
    while index < len(data):
        char = data[index]
        if char != "\033":
            keys.append(char)
            index += 1
            continue
        if data[index + 1 : index + 2] not in ("[", "O"):
            index += 1
            continue
        end = index + 2
        if data[index + 1] == "[":
            while end < len(data) and not "@" <= data[end] <= "~":
                end += 1
        if end == index + 2 and data[end : end + 1] in _ARROWS:
            keys.append(_ARROWS[data[end]])
        index = end + 1
    return keys


# ── key bindings ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class KeyResult:
    control: StreamControl
    message: str | None = None
    handled: bool = False


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _nudge(
    control: StreamControl,
    field: str,
    delta: float,
    *,
    low: float,
    high: float,
    fmt: str = ".2f",
    integer: bool = False,
) -> KeyResult:
    before = getattr(control, field)
    after = _clamp(float(before) + delta, low, high)
    if integer:
        after = int(round(after))
    if after == before:
        return KeyResult(control, f"{field} {format(before, fmt)} (at limit)", handled=True)
    return KeyResult(
        replace(control, **{field: after}),
        f"{field} {format(before, fmt)} → {format(after, fmt)}",
        handled=True,
    )


@dataclass(frozen=True)
class Knob:
    """A control the keys nudge: ``cut`` steps it down by ``step``, ``boost`` up, within ``low``–``high``."""

    field: str
    cut: str
    boost: str
    step: float
    low: float
    high: float
    fmt: str = ".2f"
    integer: bool = False


# Prompt steering. Both of these transform the tail *before* it is re-prompted,
# so they compound while held — each round's output is the next round's prompt —
# and stop compounding, leaving the stream where it drifted, the moment they
# come back to neutral. `0` returns both to neutral in one press, because
# counting keypresses back to flat mid-performance is not playing.
TILT_STEP_DB = 1.5
PROMPT_RATE_STEP = 0.02

# Every nudged control as data, so the GUI's sliders keep these steps and limits. Unshifted keys cut and shifted
# ones boost; the tilt bands sit low to high across three adjacent keys.
KNOBS: tuple[Knob, ...] = (
    Knob("temperature", "down", "up", 0.05, *TEMPERATURE.limits),
    Knob("tau_seconds", "left", "right", 0.5, TAU.limits[0], 60.0, ".1f"),
    Knob(
        "completion_duration_seconds",
        "[",
        "]",
        1.0,
        MIN_COMPLETION_DURATION_SECONDS,
        MAX_COMPLETION_DURATION_SECONDS,
        ".1f",
    ),
    Knob("top_k", ",", ".", 10, *TOP_K.limits, "d", integer=True),
    # Best-of-N. Batching is near-free on GPU (the AR loop is launch-bound), and
    # every ranking nudge — novelty included — is inert until this is above 1.
    Knob("candidates", "c", "C", 1, 1, MAX_STREAM_CANDIDATES, "d", integer=True),
    Knob("novelty_weight", "n", "N", 0.05, 0.0, 1.0),
    Knob("stuck_weight", "d", "D", 0.05, 0.0, 1.0),
    Knob("stuck_kick_rounds", "g", "G", 1, 0, 8, "d", integer=True),
    Knob("prompt_tilt_low_db", "j", "J", TILT_STEP_DB, -MAX_TILT_DB, MAX_TILT_DB, "+.1f"),
    Knob("prompt_tilt_mid_db", "k", "K", TILT_STEP_DB, -MAX_TILT_DB, MAX_TILT_DB, "+.1f"),
    Knob("prompt_tilt_high_db", "l", "L", TILT_STEP_DB, -MAX_TILT_DB, MAX_TILT_DB, "+.1f"),
    Knob("prompt_rate", "<", ">", PROMPT_RATE_STEP, MIN_PROMPT_RATE, MAX_PROMPT_RATE, ".3f"),
    Knob("round_delay_seconds", "w", "W", 0.1, 0.0, MAX_ROUND_DELAY_SECONDS, ".1f"),
)


def _reset_prompt_steering(control: StreamControl) -> KeyResult:
    neutral = replace(
        control,
        prompt_tilt_low_db=0.0,
        prompt_tilt_mid_db=0.0,
        prompt_tilt_high_db=0.0,
        prompt_rate=1.0,
    )
    if neutral == control:
        return KeyResult(control, "prompt steering already flat", handled=True)
    return KeyResult(neutral, "prompt steering flat — tilt 0 dB, rate 1.000", handled=True)


# Throttle nudging. The buffer is how long every other control takes to become
# audible, so this is really the responsiveness knob; the floor keeps a keyboard
# sweep from starving the player, and the first press on an unthrottled session
# switches the throttle on rather than doing nothing visible.
BUFFER_AHEAD_STEP_SECONDS = 5.0
BUFFER_AHEAD_DEFAULT_SECONDS = 30.0
BUFFER_AHEAD_FLOOR_SECONDS = 5.0
BUFFER_AHEAD_CEILING_SECONDS = 300.0
_DEFAULT_RESUME_GAP_SECONDS = 10.0
FLOOR_RATIO_STEP = 0.1


def _floor_ratio(control: StreamControl) -> float:
    """Where the resume threshold sits, as a fraction of the cap (unset = cap - 10s)."""
    cap = float(control.max_buffer_ahead_seconds or 0.0)
    resume = control.resume_buffer_ahead_seconds
    if resume is None:
        resume = cap - _DEFAULT_RESUME_GAP_SECONDS
    return _clamp(float(resume) / cap, 0.0, 1.0) if cap > 0 else 0.0


def _nudge_buffer(control: StreamControl, delta: float) -> KeyResult:
    """Retune the playback throttle, keeping the floor at the same fraction of the cap.

    Moving the cap without the resume threshold would silently widen or collapse
    the band the loop resumes in, so the floor follows the cap proportionally;
    a session with no explicit resume starts from the standard 10s gap.
    """
    before = control.max_buffer_ahead_seconds
    if before is None:
        after = BUFFER_AHEAD_DEFAULT_SECONDS
        message = f"throttle on, max_buffer_ahead {after:.0f}s"
        ratio = 1.0 - _DEFAULT_RESUME_GAP_SECONDS / after
    else:
        after = _clamp(float(before) + delta, BUFFER_AHEAD_FLOOR_SECONDS, BUFFER_AHEAD_CEILING_SECONDS)
        ratio = _floor_ratio(control)
        if after == float(before):
            return KeyResult(control, f"max_buffer_ahead {before:.0f}s (at limit)", handled=True)
        message = f"max_buffer_ahead {before:.0f}s → {after:.0f}s"

    resume = after * ratio
    # Nudging the cap deliberately does not re-arm a throttle that is held open:
    # one key per concern, and silently re-throttling mid-cushion would be the
    # worst possible moment for it. Say so instead of looking like a dead key.
    parked = "" if control.throttle_enabled else " — throttle still off"
    return KeyResult(
        replace(control, max_buffer_ahead_seconds=after, resume_buffer_ahead_seconds=resume),
        f"{message} (floor {resume:.0f}s){parked}",
        handled=True,
    )


def _nudge_floor(control: StreamControl, delta: float) -> KeyResult:
    """Move the resume threshold as a fraction of the cap: 1.0 tops up round by round, 0 drains it dry."""
    cap = control.max_buffer_ahead_seconds
    if cap is None:
        return KeyResult(control, "no throttle configured — - / + set a cap", handled=True)
    before = _floor_ratio(control)
    after = _clamp(round(before + delta, 1), 0.0, 1.0)
    if after == before:
        return KeyResult(control, f"buffer floor {before:.1f} × cap (at limit)", handled=True)
    resume = float(cap) * after
    return KeyResult(
        replace(control, resume_buffer_ahead_seconds=resume),
        f"buffer floor {before:.1f} → {after:.1f} × {float(cap):.0f}s = {resume:.0f}s",
        handled=True,
    )


def _toggle_throttle(control: StreamControl) -> KeyResult:
    """Hold the throttle open so the buffer can run ahead, keeping the cap.

    The cushion this banks is what carries playback across a model swap: the
    reload stalls generation for 10-60s, and only audio already generated keeps
    the player fed through it. Held open the buffer grows unbounded, so this is
    a gesture you undo — the frame keeps saying so until you do.

    A separate field from `max_buffer_ahead_seconds` rather than nulling the
    cap, so switching back restores this session's own thresholds instead of a
    default.
    """
    if control.max_buffer_ahead_seconds is None:
        return KeyResult(control, "no throttle configured — - / + set a cap", handled=True)
    if control.throttle_enabled:
        return KeyResult(
            replace(control, throttle_enabled=False),
            f"throttle off — buffer runs ahead (cap {control.max_buffer_ahead_seconds:.0f}s parked)",
            handled=True,
        )
    resume = control.resume_buffer_ahead_seconds
    return KeyResult(
        replace(control, throttle_enabled=True),
        f"throttle on (max {control.max_buffer_ahead_seconds:.0f}s"
        + (f", resume {resume:.0f}s)" if resume is not None else ")"),
        handled=True,
    )


def _select_model(control: StreamControl, playlist: Sequence[str], index: int) -> KeyResult:
    """Point ``model_xp`` at one playlist slot; the loop swaps on the next round."""
    target = playlist[index]
    where = f"{index + 1}/{len(playlist)} {Path(target).name}"
    if normalize_model_xp_path(control.model_xp) == normalize_model_xp_path(target):
        return KeyResult(control, f"model {where} (already loaded)", handled=True)
    return KeyResult(replace(control, model_xp=target), f"model → {where}", handled=True)


def _cycle_model(control: StreamControl, playlist: Sequence[str]) -> KeyResult:
    current = normalize_model_xp_path(control.model_xp)
    position = next(
        (i for i, entry in enumerate(playlist) if normalize_model_xp_path(entry) == current),
        None,
    )
    return _select_model(control, playlist, 0 if position is None else (position + 1) % len(playlist))


def apply_key(key: str | None, control: StreamControl, *, playlist: Sequence[str] = ()) -> KeyResult:
    """Pure keymap: a key plus the current control gives the next control.

    ``playlist`` is the ordered list of XP folders bound to the digit keys; it
    is empty unless the session was started with ``--playlist``, and the digits
    stay unbound in that case.

    Unknown keys return ``handled=False`` and the control untouched, so the
    caller can leave the file alone rather than rewriting it identically.
    """
    if key is None:
        return KeyResult(control)

    if playlist:
        if key == "\t":
            return _cycle_model(control, playlist)
        if len(key) == 1 and key in "123456789":
            index = int(key) - 1
            if index >= len(playlist):
                return KeyResult(control, f"playlist has {len(playlist)} models", handled=True)
            return _select_model(control, playlist, index)

    for knob in KNOBS:
        if key in (knob.cut, knob.boost):
            step = knob.step if key == knob.boost else -knob.step
            return _nudge(control, knob.field, step, low=knob.low, high=knob.high, fmt=knob.fmt, integer=knob.integer)
    if key == "+":
        return _nudge_buffer(control, BUFFER_AHEAD_STEP_SECONDS)
    if key == "-":
        return _nudge_buffer(control, -BUFFER_AHEAD_STEP_SECONDS)
    if key == "{":
        return _nudge_floor(control, -FLOOR_RATIO_STEP)
    if key == "}":
        return _nudge_floor(control, FLOOR_RATIO_STEP)
    if key == "t":
        return _toggle_throttle(control)
    if key == "0":
        return _reset_prompt_steering(control)

    if key == "r":
        return KeyResult(replace(control, command="reroll"), "reroll queued", handled=True)
    if key == "p":
        return KeyResult(replace(control, command="pin"), "pin queued", handled=True)
    if key == "i":
        return KeyResult(
            replace(control, command="inject_light"),
            f"inject latest pinned motif next round at {PINNED_INJECTION_LIGHT_WEIGHT:.2f}",
            handled=True,
        )
    if key == "I":
        return KeyResult(
            replace(control, command="inject_heavy"),
            f"inject latest pinned motif next round at {PINNED_INJECTION_HEAVY_WEIGHT:.2f}",
            handled=True,
        )
    if key == "Q":
        return KeyResult(replace(control, command="stop"), "stop queued", handled=True)
    return KeyResult(control)


HELP_LINES = (
    "keys   ↑/↓ temperature    ←/→ tau    [ / ] completion duration",
    "       , / . top_k        n/N novelty_weight    c/C candidates (best-of-N)",
    "       d/D stuck_weight (rank held-note tails down)   g/G stuck_kick rounds (0 = off)",
    "       - / + buffer cap (smaller = every knob lands sooner)   { / } buffer floor, × cap (0–1)",
    "       w/W rest before each round ∓/± 0.1s (0–3, cut short before the buffer runs dry)",
    "       t  throttle on/off — hold it open to bank a cushion before a model swap",
    "       j/J k/K l/L  prompt tilt low/mid/high ∓/± 1.5dB — EQs the tail the model re-reads",
    "       < / > prompt_rate (varispeed of the prompt only)    0  both back to flat",
    "       r reroll   p pin   i/I inject pinned motif (0.20 / 0.60)   Q stop   ? help   Ctrl-C interrupt",
    "       f  metrics follow last-generated / now-playing round (display only)",
    "       b  blind: hide every per-round reading (still recorded)",
    "       everything else stays editable in control.json",
)

PLAYLIST_HELP_HEADER = "       1-9 / tab  swap model (--playlist)"


def playlist_help_lines(playlist: Sequence[str]) -> list[str]:
    """The digit bindings, named, so a scene change is a keystroke you can find."""
    if not playlist:
        return []
    lines = [PLAYLIST_HELP_HEADER]
    lines.extend(f"            {i + 1}  {Path(entry).name}" for i, entry in enumerate(playlist[:9]))
    return lines


# ── terminal ──────────────────────────────────────────────────────────


class TerminalSession:
    """Put the terminal in cbreak + alt-screen, and always put it back.

    ``cbreak`` rather than ``raw``: it leaves ``ISIG`` on, so Ctrl-C still
    raises ``KeyboardInterrupt`` and the session's existing interrupt path keeps
    working. Restoration is registered with ``atexit`` and with SIGTERM/SIGINT
    as well as running in ``__exit__``, because a terminal left in cbreak is a
    genuinely hostile thing to hand back to someone mid-performance.

    A no-op when stdin is not a TTY (piped, ``nohup``, pytest), which is also
    what makes the surrounding code testable.
    """

    def __init__(self, *, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.active = False
        self._saved: Any = None
        self._fd: int | None = None
        self._pending = ""  # an escape sequence that the next read may still finish
        self._previous_handlers: dict[int, Any] = {}

    def is_interactive(self) -> bool:
        try:
            return bool(sys.stdin.isatty() and self.stream.isatty())
        except (AttributeError, ValueError):
            return False

    def __enter__(self) -> TerminalSession:
        if not self.is_interactive():
            return self
        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:  # noqa: BLE001 — no TUI is better than a broken terminal
            self._saved = None
            self._fd = None
            return self

        self.active = True
        self._write(_ENTER_TUI)
        atexit.register(self.restore)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            try:
                self._previous_handlers[signum] = signal.signal(signum, self._on_signal)
            except (ValueError, OSError):  # not the main thread, or unsupported
                pass
        return self

    def __exit__(self, *_exc: object) -> None:
        self.restore()

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.restore()
        previous = self._previous_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
        else:
            raise SystemExit(128 + signum)

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):
            pass

    def restore(self) -> None:
        if not self.active:
            return
        self.active = False
        self._write(_EXIT_TUI)
        if self._fd is not None and self._saved is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:  # noqa: BLE001, S110
                pass
        for signum, previous in self._previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):  # noqa: PERF203
                pass
        self._previous_handlers.clear()
        try:
            atexit.unregister(self.restore)
        except Exception:  # noqa: BLE001, S110
            pass

    def read_keys(self, timeout: float) -> list[str]:
        """Wait up to ``timeout`` for input; every key in the burst, oldest first."""
        if not self.active or self._fd is None:
            time.sleep(timeout)
            return []
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
        except (OSError, ValueError):
            time.sleep(timeout)
            return []
        if not ready:
            self._pending = ""  # input went quiet: a lone Esc or a cut-off sequence will not be finished
            return []
        try:
            data = self._pending + os.read(self._fd, 64).decode("utf-8", errors="ignore")
        except (OSError, ValueError):
            return []
        cut = _unfinished_escape_start(data)
        self._pending = data[cut:]
        return decode_keys(data[:cut])

    def read_line(self, prompt: str, *, max_length: int = 240) -> str:
        """Read a short line while cbreak is active, for an occasional TUI note."""
        if not self.active or self._fd is None:
            return ""
        self._write(prompt)
        chars: list[str] = []
        while True:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], None)
                if not ready:
                    continue
                data = os.read(self._fd, 1).decode("utf-8", errors="ignore")
            except (OSError, ValueError):
                return ""
            if not data:
                return ""
            char = data[0]
            if char in "\r\n":
                self._write("\n")
                return "".join(chars)
            if char in ("\x03", "\x04"):
                raise KeyboardInterrupt
            if char in ("\x08", "\x7f"):
                if chars:
                    chars.pop()
                    self._write("\b \b")
                continue
            if char.isprintable() and len(chars) < max_length:
                chars.append(char)


# ── the TUI ───────────────────────────────────────────────────────────


class StreamTui:
    """Draws the session and turns keystrokes into `control.json` edits.

    Runs on its own thread so it keeps redrawing while the round loop is blocked
    on the worker — which is most of the time, and exactly when you want to see
    the buffer draining.
    """

    def __init__(
        self,
        session_dir: str | Path,
        control_path: str | Path,
        *,
        status: StreamStatus | None = None,
        interval: float = 0.25,
        history: int = 60,
        events_height: int = 8,
        follow: str = FOLLOW_GENERATED,
        blind: bool = False,
        playlist: Sequence[str] = (),
        stream: TextIO | None = None,
        terminal: TerminalSession | None = None,
        playback: PlaybackTarget | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.control_path = Path(control_path)
        self.status = status if status is not None else StreamStatus()
        self.interval = max(0.05, float(interval))
        self.history = history
        self.events_height = events_height
        self.follow = follow
        self.blind = blind
        self.playlist = tuple(playlist)
        self.stream = stream if stream is not None else sys.stdout
        self.terminal = terminal if terminal is not None else TerminalSession(stream=self.stream)
        self.playback = playback
        self.flash: str | None = None
        self.show_help = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- frame ---------------------------------------------------------

    def status_lines(self) -> list[str]:
        lines = [self.status.describe()]
        if self.flash:
            lines.append(self.flash)
        if self.show_help:
            lines.extend(HELP_LINES)
            lines.extend(playlist_help_lines(self.playlist))
        return lines

    def frame(self, *, width: int) -> str:
        snapshot = read_snapshot(self.session_dir, history=self.history, events=max(0, self.events_height))
        return render_frame(
            snapshot,
            width=width,
            events_height=self.events_height,
            status_lines=self.status_lines(),
            title=f"stream — {self.session_dir.name}   [? for keys]",
            follow=self.follow,
            show_metrics=not self.blind,
        )

    def draw(self, *, width: int) -> None:
        try:
            frame = self.frame(width=width)
        except Exception as exc:  # noqa: BLE001 — a render bug must not stop the music
            frame = f"render_error: {exc!r}\n"
        try:
            self.stream.write(_HOME_CLEAR + frame)
            self.stream.flush()
        except (OSError, ValueError):
            pass

    # -- input ---------------------------------------------------------

    def handle_key(self, key: str | None) -> bool:
        """Apply one key. Returns True when the frame should be redrawn."""
        if key is None:
            return False
        if key == "?":
            self.show_help = not self.show_help
            return True
        if key == "f":
            # Display-only: never touches control.json, so it cannot affect the
            # performance. The metrics header always names the active mode.
            self.follow = FOLLOW_PLAYING if self.follow == FOLLOW_GENERATED else FOLLOW_GENERATED
            self.flash = f"metrics now follow the {self.follow} round"
            return True
        if key == "b":
            # Blind mode. Also display-only — the readings are still computed
            # and still logged, they just stop being visible while you play.
            # Every session in the statistics corpus was played with the numbers
            # on screen, which is exactly what makes the novelty reflex finding
            # circular; this is how that gets tested.
            self.blind = not self.blind
            self.flash = "playing blind — metrics hidden" if self.blind else "metrics visible again"
            return True
        try:
            with control_file_lock(self.control_path):
                control = read_control(self.control_path)
                result = apply_key(key, control, playlist=self.playlist)
                if result.handled and key == "p":
                    current_path = self.playback.current_path() if self.playback is not None else None
                    result = KeyResult(
                        replace(result.control, pin_target=str(current_path) if current_path else None),
                        result.message,
                        handled=True,
                    )
                if result.handled:
                    write_control(self.control_path, result.control)
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.flash = f"control edit failed: {exc}"
            return True
        if not result.handled:
            return False
        self.flash = result.message
        return True

    # -- loop ----------------------------------------------------------

    def _width(self) -> int:
        try:
            return min(110, max(64, os.get_terminal_size().columns - 1))
        except OSError:
            return 84

    def run(self) -> None:
        width = self._width()
        while not self._stop.is_set():
            self.draw(width=width)
            keys = self.terminal.read_keys(self.interval)
            for key in keys:
                self.handle_key(key)
            if keys:
                width = self._width()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self.run, name="macchina_ricordi-stream-tui", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
