"""Live dashboard for a running stream session.

Reads the session's ``performance.jsonl`` and ``control.json`` from disk and
redraws a terminal frame. It is a pure reader — it never writes to the session,
so it is safe to attach to a session that is mid-performance, several can watch
the same one, and it works on finished sessions too.

What it shows is the *last generated round*, not the audio currently coming out
of the speakers: with a throttle buffer the two can be a minute apart, and the
thing ``control.json`` steers is the generator. The header carries both heads
plus the round at which an edit made right now would become audible.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ...models.specs import Advisory, advisories
from ...sessions.record import read_json
from .session import LIVE_STATUS_FILENAME

DEFAULT_STREAMS_DIR = Path("artifacts/streams")

# Which round the metrics block describes.
#
# `generated` — the round the generator just finished. Correct for steering:
#   it is what your next control edit builds on.
# `playing`   — the round you are hearing right now, which with a buffer is
#   several rounds behind. Wrong for steering, right for *listening*: it lets
#   you correlate a number with what is actually in your ears, which is how a
#   metric gets validated. The frame always says which mode is active.
FOLLOW_GENERATED = "generated"
FOLLOW_PLAYING = "playing"
FOLLOW_MODES = (FOLLOW_GENERATED, FOLLOW_PLAYING)
_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_BAR_FULL = "█"
_BAR_EMPTY = "·"
_TAIL_BYTES = 512 * 1024

# `timbral_coherence` as an alarm rather than a score: 63% of rounds sit above
# this line, where the number carries almost nothing, and its usable range lives
# entirely in the low tail — the timbre lurching around. Above the line the panel
# says "steady" and stops pretending to have three decimals of meaning; below it,
# the number and a bar scaled to the tail. The raw value is untouched in the
# sidecar. This is a display experiment (2026-08-06): if the alarm reads worse
# than the number while playing, put the row back in `_METRIC_ROWS` and delete
# `_timbral_line`.
TIMBRAL_ALARM_THRESHOLD = 0.985

# Display ranges for the meters. Empirical, from streamed sessions — these only
# affect how long a bar is drawn, never a decision.
_METER_RANGES: dict[str, tuple[float, float]] = {
    "arc_coherence_score": (0.0, 1.5),
    # Only ever drawn below the alarm line — see `TIMBRAL_ALARM_THRESHOLD`.
    "timbral_coherence": (0.9, TIMBRAL_ALARM_THRESHOLD),
    "silence_fraction": (0.0, 1.0),
    "tail_silence_fraction": (0.0, 1.0),
    "novelty": (0.0, 0.6),
    "onset_density": (0.0, 12.0),
    "percussive_energy_ratio": (0.0, 1.0),
    "drone_score": (0.0, 1.0),
}

# Six rows, one per axis the corpus actually found, rather than ten labels over
# four-and-a-bit. Dropped, with the pooled Spearman that justifies each:
# `chroma_drift` (+0.75 with arc — both are chroma movement), `novelty_raw`
# (+0.64 with whitened, and it lives near 0.004 with no dynamic range),
# `tempo_instab` (−0.74 with onset density — the rate axis drawn twice with the
# sign flipped), and the arrhythmic `score`, which is a weighted function of the
# rows below it and which every phase script already excludes as "not an
# independent observation". `arc_tv4` stays in the sidecar and off the panel at
# ρ = 0.9994 with arc. The stream no longer computes any of them — see
# `macchina_ricordi.scoring.live`.
_METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("arc", "arc_coherence_score"),
    ("timbral_coh", "timbral_coherence"),
    ("novelty", "novelty"),
    ("silence", "silence_fraction"),
    ("onset_density", "onset_density"),
    ("percussive", "percussive_energy_ratio"),
    # The kick level is 0.4; drones heard blind sat at 0.6-0.7.
    ("drone", "drone_score"),
)

_CONTROL_ROW: tuple[tuple[str, str, str], ...] = (
    ("temp", "temperature", "4.2f"),
    ("tau", "tau_seconds", "4.1f"),
    ("completion", "completion_duration_seconds", "4.1f"),
    ("cands", "candidates", "d"),
    ("novelty", "novelty_weight", "4.2f"),
    ("stuck", "stuck_weight", "4.2f"),
    ("kick", "stuck_kick_rounds", "d"),
    ("min_p", "min_p", "4.2f"),
    ("top_k", "top_k", "d"),
    ("rate", "prompt_rate", "5.3f"),
    ("delay", "round_delay_seconds", "4.1f"),
    ("max_ahead", "max_buffer_ahead_seconds", "5.1f"),
)

# Values that mean "this control is off". Hidden, so the row carries what is
# actually engaged rather than a wall of neutral zeros — ten knobs no longer fit
# an 80-column terminal, and the ones that fell off the end were the newest.
_CONTROL_ROW_NEUTRAL: dict[str, float] = {
    "novelty_weight": 0.0,
    "stuck_weight": 0.0,
    "stuck_kick_rounds": 0,
    "min_p": 0.0,
    "prompt_rate": 1.0,
    "round_delay_seconds": 0.0,
}

_TILT_BANDS = ("low", "mid", "high")


def _tilt_part(control: dict[str, Any]) -> str | None:
    """The three-band prompt tilt as one compact part, or None when flat."""
    bands = [_control_float(control, f"prompt_tilt_{band}_db") or 0.0 for band in _TILT_BANDS]
    if all(abs(value) < 1e-9 for value in bands):
        return None
    return "tilt " + "/".join(f"{value:+.1f}" for value in bands)


def _wrap_control_parts(parts: list[str], *, width: int) -> list[str]:
    """The control row, wrapped rather than clipped.

    Silently losing the tail of the row is worse than a second line, and the
    tail is exactly where newly added controls land.
    """
    head = "  control   "
    indent = " " * len(head)
    lines: list[str] = []
    current = head
    for part in parts:
        fresh = current in (head, indent)
        separator = "" if fresh else "  "
        if not fresh and len(current) + len(separator) + len(part) > width:
            lines.append(current)
            current = indent + part
        else:
            current += separator + part
    lines.append(current)
    return lines


def _completion_duration_from_control(control: dict[str, Any]) -> float | None:
    """Read the new duration field, deriving it for old read-only sessions."""
    try:
        value = control.get("completion_duration_seconds")
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass

    try:
        legacy_total = float(control["round_duration_seconds"])
        tau = float(control.get("tau_seconds", 5.0))
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, legacy_total - tau)


@dataclass(frozen=True)
class MonitorSnapshot:
    """Everything one frame is rendered from."""

    session_dir: Path
    rows: list[dict[str, Any]]
    control: dict[str, Any] | None
    read_at: float
    events: list[dict[str, Any]] = field(default_factory=list)
    events_derived: bool = False
    live: dict[str, Any] | None = None  # live_status.json as last written, alive or not


@dataclass(frozen=True)
class PlaybackHead:
    """Where playback is, relative to where generation is.

    ``audible_round`` is the round most likely coming out of the speakers now,
    ``rounds_behind`` how far that is behind the generator, and ``edit_lands_at``
    the first round an edit made right now could affect (the generator re-reads
    the control file at the start of every round, so that is ``latest + 1``).
    ``drained`` means the buffer has emptied since the last logged round — the
    generator has stopped, or is throttled/held, and nothing is playing.
    """

    audible_round: int | None
    rounds_behind: int | None
    edit_lands_at: int | None
    seconds_ahead_now: float | None
    drained: bool = False


def latest_session_dir(streams_dir: str | Path = DEFAULT_STREAMS_DIR) -> Path | None:
    """Newest session folder holding a performance log, by folder name (UTC-compact, so sortable)."""
    root = Path(streams_dir).expanduser()
    if not root.is_dir():
        return None
    candidates = [d for d in root.iterdir() if d.is_dir() and (d / "performance.jsonl").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


def _read_jsonl_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    """Last ``limit`` well-formed JSON objects from a JSONL file.

    Seeks to the final `_TAIL_BYTES` rather than reading the whole log, and drops
    the first (possibly truncated) line when it did seek. A row half-written by
    the generator simply fails to parse and is skipped — it will be complete by
    the next frame.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            seeked = size > _TAIL_BYTES
            if seeked:
                fh.seek(size - _TAIL_BYTES)
            chunk = fh.read()
    except OSError:
        return []
    lines = chunk.decode("utf-8", errors="replace").splitlines()
    if seeked and lines:
        lines = lines[1:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows[-limit:]


def read_snapshot(session_dir: str | Path, *, history: int = 60, events: int = 8) -> MonitorSnapshot:
    session = Path(session_dir).expanduser()
    control = read_json(session / "control.json") or None
    rows = _read_jsonl_tail(session / "performance.jsonl", limit=history)
    event_rows = _read_jsonl_tail(session / "events.jsonl", limit=events)
    derived = False
    if not event_rows:
        # Sessions recorded before the sidecar existed still carry the facts in
        # their round rows; reconstruct what can be reconstructed.
        event_rows = derive_events(rows)[-events:]
        derived = bool(event_rows)

    return MonitorSnapshot(
        session_dir=session,
        rows=rows,
        control=control,
        read_at=time.time(),
        events=event_rows,
        events_derived=derived,
        live=read_json(session / LIVE_STATUS_FILENAME) or None,
    )


def derive_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct events from round rows, for sessions with no ``events.jsonl``.

    Only covers what a round row actually records — rescues, recaps, model
    swaps, one-shot commands and slow rounds. Throttle waits and playback
    errors are not in there at all, which is exactly why the sidecar exists.
    """
    out: list[dict[str, Any]] = []
    previous_xp: Any = None
    for index, row in enumerate(rows):

        def add(kind: str, message: str, level: str = "info", row: dict[str, Any] = row) -> None:
            out.append(
                {
                    "timestamp": row.get("timestamp"),
                    "round_idx": row.get("round_idx"),
                    "kind": kind,
                    "level": level,
                    "message": message,
                }
            )

        round_idx = row.get("round_idx")
        xp = row.get("model_xp")
        if index > 0 and xp != previous_xp:
            add("model_swap", f"{previous_xp} -> {xp}")
        previous_xp = xp

        command = row.get("command_consumed")
        injection_command = command in ("inject", "inject_light", "inject_heavy")
        if command in ("pin", "reroll") or injection_command:
            kind = "pinned_injection" if injection_command else ("reroll" if command == "reroll" else "pinned")
            add(kind, f"round {round_idx}")

        rescued = row.get("tail_rescued_from_seconds")
        if rescued is not None:
            add("tail_rescue", f"re-prompted away from a silent tail at {rescued}s", "warning")

        if row.get("stuck_kick"):
            add("stuck_kick", "a hotter round after a drone", "warning")

        motif = row.get("motif_source")
        if motif and not injection_command:
            add("motif_recap", Path(str(motif)).name)

        try:
            elapsed = float(row.get("elapsed_seconds"))
            active_control = row.get("active_control")
            if isinstance(active_control, dict):
                budget = _completion_duration_from_control(active_control)
            else:
                budget = None
            if budget is None:
                budget = float(row.get("playback_seconds"))
            if budget > 0 and elapsed > budget:
                add("generation_slow_warning", f"took {elapsed:.1f}s against a {budget:.1f}s budget", "warning")
        except (TypeError, ValueError):
            pass
    return out


# ── derived readings ──────────────────────────────────────────────────


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def playback_head(rows: list[dict[str, Any]], *, now: float) -> PlaybackHead:
    """Estimate which round is audible right now, from the last row's buffer reading.

    ``buffer_seconds_ahead`` is sampled when a round is enqueued; wall-clock has
    drained it since, so the live figure is that value minus the age of the row.
    Walking backwards through ``playback_seconds`` until the remaining buffer is
    consumed identifies the round the player is inside. Inherits every
    approximation in ``PlaybackWorker.buffer_seconds_ahead`` (it undercounts
    after an underrun), so this is an instrument, not a clock.
    """
    if not rows:
        return PlaybackHead(None, None, None, None)
    last = rows[-1]
    latest_round = last.get("round_idx")
    edit_lands_at = latest_round + 1 if isinstance(latest_round, int) else None

    ahead = last.get("buffer_seconds_ahead")
    stamped = _parse_timestamp(last.get("timestamp"))
    if ahead is None or stamped is None:
        return PlaybackHead(None, None, edit_lands_at, None)
    remaining = float(ahead) - max(0.0, now - stamped)
    if remaining <= 0.0:
        return PlaybackHead(latest_round, 0, edit_lands_at, 0.0, drained=True)

    for row in reversed(rows):
        span = row.get("playback_seconds")
        try:
            span = float(span) if span is not None else 0.0
        except (TypeError, ValueError):
            span = 0.0
        remaining -= span
        if remaining <= 0.0:
            audible = row.get("round_idx")
            behind = latest_round - audible if isinstance(audible, int) and isinstance(latest_round, int) else None
            return PlaybackHead(audible, behind, edit_lands_at, float(ahead))
    # Buffer reaches past the oldest row we loaded.
    oldest = rows[0].get("round_idx")
    return PlaybackHead(oldest if isinstance(oldest, int) else None, None, edit_lands_at, float(ahead))


def _selected_candidate(score: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in score.get("candidates") or []:
        if isinstance(candidate, dict) and candidate.get("selected"):
            return candidate
    return None


def _round_novelty(score: dict[str, Any]) -> float | None:
    """Novelty of the round that was kept, wherever it got recorded.

    Whitened first: raw cosine over MFCC fingerprints is pinned near 1 and has
    almost no dynamic range, so it is only the fallback for an uncalibrated
    session.
    """
    value = score.get("novelty_whitened")
    if value is None:
        value = score.get("novelty_raw")
    if value is None:
        selected = _selected_candidate(score)
        value = selected.get("novelty") if selected else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _history(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    out: list[float | None] = []
    for row in rows:
        score = row.get("score")
        if not isinstance(score, dict):
            out.append(None)
            continue
        value = _round_novelty(score) if key == "novelty" else score.get(key)
        try:
            out.append(float(value) if value is not None else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


# ── drawing primitives ────────────────────────────────────────────────


def _downsample(values: list[float | None], *, width: int) -> list[float | None]:
    """Bucket-average ``values`` down to ``width`` points, so a long history still fits.

    A bucket with no readings stays ``None`` rather than borrowing a neighbour's.
    """
    if len(values) <= width or width <= 0:
        return values
    out: list[float | None] = []
    for i in range(width):
        chunk = values[len(values) * i // width : len(values) * (i + 1) // width]
        present = [v for v in chunk if v is not None]
        out.append(sum(present) / len(present) if present else None)
    return out


def sparkline(values: list[float | None], *, width: int = 40) -> str:
    """Fixed-width sparkline, self-scaled. Longer histories are bucket-averaged; gaps are spaces."""
    tail = _downsample(values, width=width)
    present = [v for v in tail if v is not None]
    if not present:
        return " " * len(tail)
    lo, hi = min(present), max(present)
    span = hi - lo
    out = []
    for value in tail:
        if value is None:
            out.append(" ")
            continue
        level = 0 if span < 1e-12 else int(round((value - lo) / span * (len(_SPARK_CHARS) - 1)))
        out.append(_SPARK_CHARS[max(0, min(len(_SPARK_CHARS) - 1, level))])
    return "".join(out)


def meter(value: float | None, *, lo: float, hi: float, width: int = 14) -> str:
    if value is None:
        return " " * width
    span = hi - lo
    filled = 0 if span <= 0 else int(round((float(value) - lo) / span * width))
    return _BAR_FULL * max(0, min(width, filled)) + _BAR_EMPTY * (width - max(0, min(width, filled)))


def _num(value: Any, spec: str, *, dash: str = "—") -> str:
    """Format a number, right-aligning ``dash`` in the same field when it is missing."""
    pad = spec.split(".")[0] or "0"
    try:
        width = int(pad)
    except ValueError:
        width = 0
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return f"{dash:>{width}}"


def _elapsed_str(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.1f}m"


def _age(timestamp: Any, *, now: float) -> str:
    stamped = _parse_timestamp(timestamp)
    if stamped is None:
        return "—"
    return _elapsed_str(max(0.0, now - stamped)) + " ago"


def _metric_line(label: str, payload: dict[str, Any], key: str, *, width: int = 16) -> str:
    raw = payload.get(key)
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        value = None
    lo, hi = _METER_RANGES.get(key, (0.0, 1.0))
    return f"  {label:<14}{_num(value, '8.3f')}  {meter(value, lo=lo, hi=hi, width=width)}"


def _timbral_line(label: str, payload: dict[str, Any], *, width: int = 16) -> str:
    """The timbral-coherence row: an alarm below the threshold, a word above it."""
    raw = payload.get("timbral_coherence")
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        value = None
    if value is None:
        return f"  {label:<14}{_num(None, '8.3f')}"
    if value >= TIMBRAL_ALARM_THRESHOLD:
        return f"  {label:<14}{'steady':>8}  (≥{TIMBRAL_ALARM_THRESHOLD:g})"
    lo, hi = _METER_RANGES["timbral_coherence"]
    return f"  {label:<14}{_num(value, '8.3f')}  ⚠ lurching {meter(value, lo=lo, hi=hi, width=width)}"


def _focus_row(rows: list[dict[str, Any]], *, head: PlaybackHead, follow: str) -> tuple[dict[str, Any], str]:
    """The round the metrics block describes, and a label naming it out loud.

    Falls back to the newest round — saying so — when playback-follow is asked
    for but the audible round is unknown or has scrolled out of the loaded
    history, so the panel never silently shows something other than its label.
    """
    last = rows[-1]
    if follow != FOLLOW_PLAYING:
        return last, f"round {last.get('round_idx')} · last generated"
    if head.drained or head.audible_round is None:
        return last, f"round {last.get('round_idx')} · last generated (nothing playing)"
    match = next((row for row in reversed(rows) if row.get("round_idx") == head.audible_round), None)
    if match is None:
        return last, f"round {last.get('round_idx')} · last generated (round {head.audible_round} not in history)"
    if head.rounds_behind is None:
        # The buffer reaches past the oldest row loaded, so playback_head could
        # only clamp. Saying "now playing" would overstate what is known.
        return match, f"round {match.get('round_idx')} or earlier · ▶ playing — buffer predates --history"
    behind = f", {head.rounds_behind} behind" if head.rounds_behind else ""
    return match, f"round {match.get('round_idx')} · ▶ NOW PLAYING{behind} — lags the generator"


def _control_float(control: dict[str, Any], key: str) -> float | None:
    try:
        value = control.get(key)
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _advisories_for(model: Any, control: dict[str, Any]) -> tuple[Advisory, ...]:
    top_k = _control_float(control, "top_k")
    return advisories(
        str(model) if model else None,
        tau_seconds=_control_float(control, "tau_seconds"),
        completion_duration_seconds=_control_float(control, "completion_duration_seconds"),
        temperature=_control_float(control, "temperature"),
        top_k=int(top_k) if top_k is not None else None,
    )


def render_advisories(model: Any, control: dict[str, Any]) -> list[str]:
    """The guidance block: what the corpus says about these settings on this model.

    Advisory only, and deliberately so — overrunning the horizon is sometimes
    the sound being sought, and no knob here is ever turned automatically. See
    ``model_specs.advisories``.
    """
    return [
        f"  {'guidance' if index == 0 else '':<10}{'⚠ ' if a.level == 'warning' else '· '}{a.message}"
        for index, a in enumerate(_advisories_for(model, control))
    ]


def _focus_flags(focus: dict[str, Any]) -> list[str]:
    """How this round's prompt was made unusual: a rescue, a recap, a varispeed, a tilt."""
    flags = []
    if focus.get("tail_rescued_from_seconds") is not None:
        flags.append("tail_rescue")
    if focus.get("motif_source"):
        flags.append(f"motif {Path(str(focus['motif_source'])).name}")
    rate = focus.get("prompt_rate")
    if rate is not None and abs(float(rate) - 1.0) > 1e-9:
        mode = focus.get("prompt_rate_mode") or "tape"
        flags.append(f"prompt_rate {float(rate):.3f} ({mode})")
    tilt = focus.get("prompt_tilt_db")
    if isinstance(tilt, dict):
        flags.append(
            "tilt " + "/".join(f"{float(tilt.get(band, 0.0)):+.1f}" for band in ("low_db", "mid_db", "high_db")) + "dB"
        )
    return flags


_LEVEL_MARKS = {"info": " ", "warning": "!", "error": "✗"}


def render_events(events: list[dict[str, Any]], *, height: int, width: int, derived: bool = False) -> list[str]:
    """The events box: newest last, always exactly ``height`` lines tall.

    Padding to a fixed height is the point — the box scrolls internally while
    the frame around it never changes size, so nothing below it shifts as
    events arrive.
    """
    title = "  events" + ("  (derived from round rows — no events.jsonl)" if derived else "")
    lines = [title]
    body: list[str] = []
    for event in events[-height:]:
        stamp = _parse_timestamp(event.get("timestamp"))
        clock = datetime.fromtimestamp(stamp, UTC).strftime("%H:%M:%S") if stamp else "--:--:--"
        mark = _LEVEL_MARKS.get(str(event.get("level", "info")), " ")
        round_idx = event.get("round_idx")
        where = f"r{round_idx}" if round_idx is not None else "  "
        line = f"  {mark} {clock} {where:>5}  {event.get('kind', '?')}: {event.get('message', '')}"
        body.append(line[:width])
    body += [""] * max(0, height - len(body))
    return lines + body


def render_frame(
    snapshot: MonitorSnapshot,
    *,
    width: int = 84,
    events_height: int = 8,
    status_lines: Sequence[str] = (),
    title: str | None = None,
    follow: str = FOLLOW_GENERATED,
    show_metrics: bool = True,
) -> str:
    """Render one dashboard frame as plain text.

    ``status_lines`` are placed directly under the header, for a caller that
    knows something the session files cannot say yet — the in-flight round, in
    the TUI's case. ``title`` overrides the header text for the same reason.

    ``follow`` picks which round the metrics and candidate table describe — see
    `FOLLOW_PLAYING`. The choice is always spelled out on the metrics header, so
    a lagging reading can never be mistaken for a live one.

    ``show_metrics=False`` is *blind mode*: every per-round reading leaves the
    frame — the metrics block, the candidate table and the arc/novelty/silence
    sparklines. What stays is everything that is not a judgement about the
    music: buffer health, the control row, the guidance block and the events.
    The point is that the readings were on screen for all 3,523 rounds of the
    statistics corpus, which is what makes the strongest result in it circular
    — the player reaching for the novelty knob when whitened novelty was low
    may have been reacting to the number. Playing blind is the only way to find
    out, and it is the prerequisite for the blind A/B.
    """
    rule = "─" * width
    rows = snapshot.rows
    title = title or f"stream monitor — {snapshot.session_dir.name}"
    if not rows:
        head = [title, rule, *(f"  {line}" for line in status_lines), "  waiting for the first round..."]
        return "\n".join(line[:width] for line in head) + "\n"

    last = rows[-1]
    control = dict(snapshot.control or last.get("active_control") or {})
    completion_duration = _completion_duration_from_control(control)
    if completion_duration is not None:
        control["completion_duration_seconds"] = completion_duration
    take = last.get("take_idx") or 0
    head = playback_head(rows, now=snapshot.read_at)

    focus, focus_label = _focus_row(rows, head=head, follow=follow)
    score_raw = focus.get("score")
    score: dict[str, Any] = score_raw if isinstance(score_raw, dict) else {}

    lines: list[str] = []
    right = f"round {last.get('round_idx')}" + (f" take {take}" if take else "")
    right += f"   {_age(last.get('timestamp'), now=snapshot.read_at)}"
    lines.append(f"{title}{right:>{max(1, width - len(title))}}")
    lines.append(rule)
    lines.extend(f"  {line}" for line in status_lines)

    stamped = _parse_timestamp(last.get("timestamp"))
    age = max(0.0, snapshot.read_at - stamped) if stamped is not None else None
    try:
        idle_after = 2.0 * float(completion_duration) if completion_duration is not None else 20.0
    except (TypeError, ValueError):
        idle_after = 20.0
    if head.drained:
        gap = _elapsed_str(age) if age is not None else "a while"
        lines.append(f"  playback drained — nothing new for {gap}; generator stopped, throttled or held")
    elif head.audible_round is not None:
        behind = f" ({head.rounds_behind} behind)" if head.rounds_behind else ""
        lands = f" · an edit now lands at round {head.edit_lands_at}" if head.edit_lands_at else ""
        lines.append(f"  hearing round {head.audible_round}{behind}{lands}")
    if age is not None and age > max(30.0, idle_after) and not head.drained:
        lines.append(f"  ⚠ readings below are {_elapsed_str(age)} old")

    elapsed = last.get("elapsed_seconds")
    playback = last.get("playback_seconds")
    note = ""
    try:
        if elapsed is not None and playback:
            ratio = float(elapsed) / float(playback)
            note = f"  {ratio:.2f}x rt" + (" ⚠ starving" if ratio > 1.0 else "")
    except (TypeError, ValueError, ZeroDivisionError):
        note = ""
    lines.append(
        f"  last sample  {Path(str(last.get('round_wav', '?'))).name}  "
        f"gen {_num(elapsed, '.1f')}s / fresh {_num(playback, '.1f')}s{note}"
    )

    ahead_line = f"  buffer ahead {_num(head.seconds_ahead_now, '7.1f')}s"
    max_ahead = control.get("max_buffer_ahead_seconds")
    # A throttle held open has no ceiling to meter against, and is a state you
    # meant to undo — so it says so on the line you already watch the buffer on,
    # for as long as it lasts, rather than only in the flash that set it.
    throttle_open = max_ahead and not bool(control.get("throttle_enabled", True))
    try:
        if head.seconds_ahead_now is not None and max_ahead and not throttle_open:
            ahead_line += "  " + meter(head.seconds_ahead_now, lo=0.0, hi=float(max_ahead), width=20)
            ahead_line += f"  max {float(max_ahead):.0f}s"
            resume = control.get("resume_buffer_ahead_seconds")
            if resume is not None:
                ahead_line += f" · floor {float(resume):.0f}s ({float(resume) / float(max_ahead):.1f})"
        elif throttle_open:
            ahead_line += f"  ⚠ throttle off — running ahead (cap {float(max_ahead):.0f}s parked)"
    except (TypeError, ValueError):
        pass
    lines.append(ahead_line)

    flags = _focus_flags(focus)
    if flags:
        lines.append("  flags         " + ", ".join(flags))

    lines.append(rule)
    if show_metrics:
        lines.append(f"  metrics · {focus_label}")
        metrics = dict(score)
        metrics["novelty"] = _round_novelty(score)
        lines.extend(
            _timbral_line(label, metrics) if key == "timbral_coherence" else _metric_line(label, metrics, key)
            for label, key in _METRIC_ROWS
        )

        candidates = [c for c in (score.get("candidates") or []) if isinstance(c, dict)]
        if candidates:
            lines.append(rule)
            lines.append("  candidates          arc   tail-sil    novelty")
            for candidate in candidates:
                mark = "◀" if candidate.get("selected") else " "
                lines.append(
                    f"   {mark} #{candidate.get('candidate')}       "
                    f"{_num(candidate.get('arc_coherence_score'), '9.3f')}"
                    f"{_num(candidate.get('tail_silence_fraction'), '11.3f')}"
                    f"{_num(candidate.get('novelty'), '11.4f')}"
                )
    else:
        lines.append("  metrics hidden — playing blind; the numbers are still recorded")

    lines.append(rule)
    spark_width = max(10, width - 18)
    if show_metrics:
        for label, key in (("arc", "arc_coherence_score"), ("novelty", "novelty"), ("silence", "silence_fraction")):
            lines.append(f"  {label:<12}  {sparkline(_history(rows, key), width=spark_width)}")
    buffer_history: list[float | None] = []
    for row in rows:
        value = row.get("buffer_seconds_ahead")
        try:
            buffer_history.append(float(value) if value is not None else None)
        except (TypeError, ValueError):
            buffer_history.append(None)
    lines.append(f"  {'buffer':<12}  {sparkline(buffer_history, width=spark_width)}")
    binned = "" if len(rows) <= spark_width else f", {len(rows) / spark_width:.1f} per column"
    lines.append(f"  {'':<12}  last {len(rows)} rounds{binned}")

    lines.append(rule)
    parts = []
    for label, key, spec in _CONTROL_ROW:
        value = control.get(key)
        if value is None:
            continue
        neutral = _CONTROL_ROW_NEUTRAL.get(key)
        if neutral is not None and _control_float(control, key) == neutral:
            continue
        try:
            parts.append(f"{label} {format(value, spec)}")
        except (TypeError, ValueError):
            parts.append(f"{label} {value}")
    tilt_part = _tilt_part(control)
    if tilt_part is not None:
        parts.append(tilt_part)
    lines.extend(_wrap_control_parts(parts, width=width))
    xp = control.get("model_xp") or last.get("model_xp")
    if xp:
        lines.append(f"  model     {xp}")
    command = control.get("command")
    if command and command != "continue":
        lines.append(f"  pending   command={command}")
    lines.extend(render_advisories(xp, control))

    if events_height > 0:
        lines.append(rule)
        lines.extend(
            render_events(
                snapshot.events,
                height=events_height,
                width=width,
                derived=snapshot.events_derived,
            )
        )

    # Clip rather than let the terminal wrap: a wrapped line pushes the frame
    # taller than the screen and the cursor-home redraw then leaves debris.
    return "\n".join(line[:width] for line in lines) + "\n"


_ROUND_FILE = re.compile(r"round_(\d+)(?:_take_\d+)?\.wav$")


def _live_head(rows: list[dict[str, Any]], live: dict[str, Any]) -> PlaybackHead | None:
    """Where playback is, read off ``live_status.json``: the file the speakers are on, not an estimate.

    ``None`` when the session plays nothing (``--no-play``), which leaves it to the estimate to say so.
    """
    ahead = _control_float(live, "buffer_seconds_ahead")
    if ahead is None:
        return None
    latest = rows[-1].get("round_idx")
    match = _ROUND_FILE.search(str(live.get("playing") or ""))
    audible = int(match.group(1)) if match else None
    known = audible is not None and isinstance(latest, int)
    edit_lands_at = latest + 1 if isinstance(latest, int) else None
    current = live.get("round_idx")
    if isinstance(current, int):
        # The loop reads control.json while waiting, throttled or resting; once it generates, edits wait a round.
        edit_lands_at = current + 1 if live.get("phase") in ("generating", "scoring") else current
    return PlaybackHead(audible, latest - audible if known else None, edit_lands_at, ahead, drained=match is None)


def frame_data(
    snapshot: MonitorSnapshot, *, follow: str = FOLLOW_GENERATED, live: dict[str, Any] | None = None
) -> dict[str, Any]:
    """What `render_frame` draws, as data, for a front end that draws it itself (the GUI's Scène page).

    Pass ``live`` (the session's ``live_status.json``) only while its process is known to be running: the round
    being heard then comes from it exactly, instead of being estimated from the buffer telemetry.
    """
    rows = snapshot.rows
    control = dict(snapshot.control or (rows[-1].get("active_control") if rows else None) or {})
    completion = _completion_duration_from_control(control)
    if completion is not None:
        control["completion_duration_seconds"] = completion
    model = control.get("model_xp") or (rows[-1].get("model_xp") if rows else None)
    data: dict[str, Any] = {
        "control": control,
        "model": model,
        "guidance": [asdict(advisory) for advisory in _advisories_for(model, control)],
        "events": snapshot.events,
        "events_derived": snapshot.events_derived,
        "metric_rows": _METRIC_ROWS,
        "meter_ranges": _METER_RANGES,
        "timbral_alarm": TIMBRAL_ALARM_THRESHOLD,
        "latest": None,
        "head": None,
        "focus": None,
        "history": None,
    }
    if not rows:
        return data

    last = rows[-1]
    head = (_live_head(rows, live) if live else None) or playback_head(rows, now=snapshot.read_at)
    focus, label = _focus_row(rows, head=head, follow=follow)
    score = focus.get("score") if isinstance(focus.get("score"), dict) else {}
    metrics = {key: _control_float(score, key) for _, key in _METRIC_ROWS}
    metrics["novelty"] = _round_novelty(score)
    stamped = _parse_timestamp(last.get("timestamp"))
    age = max(0.0, snapshot.read_at - stamped) if stamped is not None else None
    elapsed, fresh = _control_float(last, "elapsed_seconds"), _control_float(last, "playback_seconds")
    data.update(
        latest={
            "round_idx": last.get("round_idx"),
            "take_idx": last.get("take_idx") or 0,
            "round_wav": Path(str(last.get("round_wav") or "")).name or None,
            "model_xp": last.get("model_xp"),  # the model this round was made with; control.json's may be next
            "age_seconds": age,
            # Same rule as the frame's "readings below are old" line.
            "stale": age is not None and not head.drained and age > max(30.0, 2.0 * (completion or 10.0)),
            "elapsed_seconds": elapsed,
            "playback_seconds": fresh,
            "realtime_ratio": elapsed / fresh if elapsed is not None and fresh else None,
        },
        head=asdict(head),
        focus={
            "round_idx": focus.get("round_idx"),
            "label": label,
            "flags": _focus_flags(focus),
            "metrics": metrics,
            "candidates": [c for c in score.get("candidates") or [] if isinstance(c, dict)],
        },
        history={
            "round_idx": [row.get("round_idx") for row in rows],
            "arc": _history(rows, "arc_coherence_score"),
            "novelty": _history(rows, "novelty"),
            "silence": _history(rows, "silence_fraction"),
            "buffer": [_control_float(row, "buffer_seconds_ahead") for row in rows],
            "model": [row.get("model_xp") for row in rows],
            "playback": [_control_float(row, "playback_seconds") for row in rows],
            "marks": _round_marks(rows),
            "edits": _control_edits(rows),
        },
    )
    return data


def _round_marks(rows: list[dict[str, Any]]) -> list[list[str]]:
    """Per round, the kinds of event its row records (pins, rerolls, rescues, swaps…), for the round ribbon."""
    by_round: dict[Any, list[str]] = {}
    for event in derive_events(rows):
        by_round.setdefault(event["round_idx"], []).append(event["kind"])
    return [by_round.get(row.get("round_idx"), []) for row in rows]


def _control_edits(rows: list[dict[str, Any]]) -> list[list[list[Any]]]:
    """Per round, ``[field, before, after]`` for every control that differs from the round before it."""
    out: list[list[list[Any]]] = []
    previous: dict[str, Any] | None = None
    for row in rows:
        control = row.get("active_control") if isinstance(row.get("active_control"), dict) else None
        out.append(
            [
                [key, previous.get(key), value]
                for key, value in control.items()
                if key not in ("command", "pin_target") and previous.get(key) != value
            ]
            if previous is not None and control is not None
            else []
        )
        previous = control
    return out


def terminal_columns(default: int = 84) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def run_monitor(
    session_dir: str | Path,
    *,
    interval: float = 2.0,
    history: int = 60,
    events_height: int = 8,
    follow: str = FOLLOW_GENERATED,
    once: bool = False,
    width: int | None = None,
    stream: TextIO | None = None,
) -> None:
    """Redraw the dashboard every ``interval`` seconds until interrupted."""
    import sys

    out = stream if stream is not None else sys.stdout
    resolved = width or min(110, max(64, terminal_columns() - 1))
    try:
        while True:
            snapshot = read_snapshot(session_dir, history=history, events=max(0, events_height))
            frame = render_frame(snapshot, width=resolved, events_height=max(0, events_height), follow=follow)
            if once:
                out.write(frame)
                out.flush()
                return
            # Home the cursor and clear forward: no flicker, and no alt-screen to
            # restore if the monitor is killed mid-frame.
            out.write("\033[H\033[J" + frame)
            out.flush()
            time.sleep(max(0.2, float(interval)))
    except KeyboardInterrupt:
        out.write("\n")
        out.flush()
