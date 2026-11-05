"""Single-keystroke blind A/B listening interface."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO

from ..cli.common import clip
from ..instrument.stream.tui import TerminalSession
from ..sessions.takes import knob_text
from .ab import AbPair, NonBlockingClipPlayer, PlaybackClip, matchups, pair_context_metadata, varied_knob
from .preference import (
    PreferenceRecord,
    append_preference,
    load_preferences,
    new_session_id,
    remove_last_preference,
    session_agreement,
    write_session_sidecar,
)
from .questions import PAIRWISE, PAIRWISE_CONTEXT, Question, get_question

_CLEAR = "\033[H\033[J"
_MIN_SECONDS = 1.0
_MAX_SECONDS = 120.0
_SECONDS_STEP = 1.0
_DEFAULT_SHORTENED_SECONDS = 30.0


class AbPlayback(Protocol):
    def play(self, path: str | Path, *, seconds: float | None = None) -> bool: ...

    def play_sequence(
        self,
        paths: Sequence[str | Path | PlaybackClip],
        *,
        seconds: float | None = None,
    ) -> bool: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def _agreement_wins(agreement: float, pairs: int) -> str:
    value = agreement * pairs
    rounded = round(value)
    return str(rounded) if abs(value - rounded) < 1e-9 else f"{value:.1f}"


class AbTui:
    """Manage one blind A/B session while the terminal remains responsive."""

    def __init__(
        self,
        pairs: Sequence[AbPair],
        *,
        store_path: str | Path,
        question: str = "better",
        listener: str = "me",
        seconds: float | None = None,
        session_id: str = "",
        context: Mapping[str, Any] | None = None,
        sidecar: Mapping[str, Any] | None = None,
        metric_lookup: Mapping[str, Mapping[str, float]] | None = None,
        stream: TextIO | None = None,
        terminal: TerminalSession | None = None,
        playback: AbPlayback | None = None,
        note_reader: Callable[[str], str] | None = None,
        print_fn: Callable[..., Any] = print,
        interval: float = 0.08,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.pairs = list(pairs)
        self.store_path = Path(store_path).expanduser()
        self.question: Question = get_question(question)
        if self.question.type not in (PAIRWISE, PAIRWISE_CONTEXT):
            raise ValueError(
                f"question {self.question.id!r} is {self.question.type}, which the pairwise TUI cannot ask honestly"
            )
        if self.question.type == PAIRWISE_CONTEXT:
            missing_reels = [pair.pair_id for pair in self.pairs if not pair.context_reel_path]
            if missing_reels:
                raise ValueError(f"shared-context pairs have no prepared reel: {', '.join(missing_reels)}")
        self.listener = listener
        self.seconds = seconds if seconds is None or seconds > 0 else None
        self.session_id = session_id or new_session_id()
        self.context = dict(context or {})
        self.sidecar = dict(sidecar or {})
        self.metric_lookup = metric_lookup
        self.stream = stream
        self.terminal = terminal if terminal is not None else TerminalSession(stream=stream)
        self.playback = playback if playback is not None else NonBlockingClipPlayer()
        self.note_reader = note_reader
        self.print_fn = print_fn
        self.interval = max(0.03, float(interval))
        self.clock = clock
        self.started_at = clock()
        self.flash: str | None = None
        self.playing: str | None = None
        self.pending_note = ""
        self.show_help = False
        self._index = 0
        self._outcomes = {pair.pair_id: "unreached" for pair in self.pairs}
        self._written: list[PreferenceRecord] = []
        self._done = not self.pairs
        self._finalized = False
        self._auto_played_pair_id: str | None = None
        self.sidecar_path: Path | None = None

    @property
    def records(self) -> list[PreferenceRecord]:
        return list(self._written)

    @property
    def current_pair(self) -> AbPair | None:
        if 0 <= self._index < len(self.pairs):
            return self.pairs[self._index]
        return None

    @property
    def done(self) -> bool:
        return self._done

    def _seconds_text(self) -> str:
        return "whole clip" if self.seconds is None else f"{self.seconds:.0f}s max"

    def _current_text(self) -> str:
        if self.playing is None:
            return "nothing playing"
        return f"▶ {self.playing}"

    def _refresh_playing(self) -> None:
        current_path = getattr(self.playback, "current_path", None)
        if not callable(current_path):
            return
        path = current_path()
        if path is None:
            self.playing = None
            return
        pair = self.current_pair
        if pair is None:
            self.playing = None
        elif pair.context_reel_path and Path(path) == Path(pair.context_reel_path):
            self.playing = "context"
        elif Path(path) == Path(pair.first.playable_path):
            self.playing = "A"
        elif Path(path) == Path(pair.second.playable_path):
            self.playing = "B"

    def status_lines(self) -> list[str]:
        lines = [self._current_text()]
        if self.flash:
            lines.append(self.flash)
        if self.pending_note:
            lines.append("note queued for the next verdict")
        if self.show_help:
            lines.extend(
                (
                    "1/2 play A/B   space replay   c context   a/b/t verdict",
                    "z undo   s skip   m note   ,/. seconds   q quit",
                )
            )
        return lines

    def frame(self, *, width: int = 84) -> str:
        self._refresh_playing()
        width = max(32, int(width))
        total = len(self.pairs)
        shown_index = min(self._index + 1, total) if total else 0
        progress = 1.0 if self._done else (self._index / total if total else 0.0)
        bar_width = max(10, min(48, width - 28))
        filled = int(round(progress * bar_width))
        bar = "█" * filled + "·" * (bar_width - filled)
        elapsed = max(0.0, self.clock() - self.started_at)
        lines = [
            clip(self.question.prompt, width),
            "",
            clip(f"pair {shown_index}/{total}  [{bar}]", width),
            clip(f"elapsed {elapsed:.0f}s  ·  {self._current_text()}  ·  {self._seconds_text()}", width),
        ]
        lines.extend(clip(line, width) for line in self.status_lines()[1:])
        return "\n".join(lines) + "\n"

    def draw(self, *, width: int = 84) -> None:
        try:
            frame = self.frame(width=width)
        except Exception as exc:  # noqa: BLE001 — a frame bug must not end a listening session
            frame = f"render_error: {exc!r}\n"
        output = self.stream
        if output is None:
            return
        try:
            output.write(_CLEAR + frame)
            output.flush()
        except (OSError, ValueError):
            pass

    def _stop_playback(self) -> None:
        try:
            self.playback.stop()
        except (OSError, RuntimeError):
            pass
        self.playing = None

    def _play(self, side: str) -> None:
        pair = self.current_pair
        if pair is None:
            return
        path = pair.first.playable_path if side == "A" else pair.second.playable_path
        self._stop_playback()
        try:
            started = self.playback.play(path, seconds=self.seconds)
        except (OSError, RuntimeError):
            started = False
        self.playing = side if started else None
        self.flash = f"playing {side}" if started else "could not start audio player"

    def _context_paths(self) -> tuple[str, ...]:
        pair = self.current_pair
        if pair is None:
            return ()
        return (pair.context_reel_path,) if pair.context_reel_path else pair.context_paths

    def _replay(self) -> None:
        pair = self.current_pair
        if pair is None:
            return
        if pair.context_reel_path:
            paths: list[str | Path | PlaybackClip] = [
                PlaybackClip(pair.context_reel_path),
                PlaybackClip(pair.first.playable_path, self.seconds),
                PlaybackClip(pair.second.playable_path, self.seconds),
            ]
            shared_seconds = None
        else:
            paths = [*pair.context_paths, pair.first.playable_path, pair.second.playable_path]
            shared_seconds = self.seconds
        self._stop_playback()
        try:
            started = self.playback.play_sequence(paths, seconds=shared_seconds)
        except (OSError, RuntimeError):
            started = False
        self.playing = "context" if started and self._context_paths() else ("A" if started else None)
        self.flash = "replaying sequence" if started else "could not start audio player"

    def _replay_context(self) -> None:
        paths = self._context_paths()
        if not paths:
            self.flash = "no context reel in this session"
            return
        self._stop_playback()
        try:
            if self.current_pair is not None and self.current_pair.context_reel_path:
                started = self.playback.play_sequence([PlaybackClip(paths[0])])
            else:
                started = self.playback.play_sequence(paths, seconds=self.seconds)
        except (OSError, RuntimeError):
            started = False
        self.playing = "context" if started else None
        self.flash = "replaying context" if started else "could not start audio player"

    def _autoplay_current(self) -> None:
        pair = self.current_pair
        if self.question.type != PAIRWISE_CONTEXT or pair is None or pair.pair_id == self._auto_played_pair_id:
            return
        self._auto_played_pair_id = pair.pair_id
        self._replay()

    def _read_note(self) -> str:
        if self.note_reader is not None:
            return self.note_reader("note: ")[:240].strip()
        reader = getattr(self.terminal, "read_line", None)
        if callable(reader):
            return str(reader("note: ", max_length=240)).strip()
        return ""

    def _record_verdict(self, verdict: str) -> None:
        pair = self.current_pair
        if pair is None:
            return
        self._stop_playback()
        record = PreferenceRecord(
            pair_id=pair.pair_id,
            path_a=pair.first.path,
            path_b=pair.second.path,
            winner=verdict,
            listener=self.listener,
            notes=self.pending_note,
            context={
                "group_a": pair.first.group,
                "group_b": pair.second.group,
                "label_a": pair.first.label,
                "label_b": pair.second.label,
                **self.context,
                **pair_context_metadata(pair),
            },
            question=self.question.id,
            context_paths=list(pair.context_paths),
            session_id=self.session_id,
            seconds_played=self.seconds,
        )
        append_preference(record, path=self.store_path)
        self._written.append(record)
        self._outcomes[pair.pair_id] = verdict
        self.pending_note = ""
        self.flash = f"recorded {verdict}; next pair"
        self._index += 1
        if self._index >= len(self.pairs):
            self._done = True
        else:
            self._autoplay_current()

    def _undo(self) -> None:
        if not self._written:
            self.flash = "nothing to undo"
            return
        record = self._written[-1]
        try:
            remove_last_preference(record, path=self.store_path)
        except ValueError as exc:
            self.flash = f"undo failed: {exc}"
            return
        self._written.pop()
        for index, pair in enumerate(self.pairs):
            if pair.pair_id == record.pair_id:
                self._index = index
                self._outcomes[pair.pair_id] = "unreached"
                break
        self._done = False
        self._auto_played_pair_id = None
        self.flash = "last verdict undone"
        self._autoplay_current()

    def _nudge_seconds(self, delta: float) -> None:
        if self.seconds is None:
            if delta < 0:
                self.seconds = _DEFAULT_SHORTENED_SECONDS
                self.flash = f"seconds {self.seconds:.0f}s max"
            else:
                self.flash = "already playing the whole clip"
            return
        updated = min(_MAX_SECONDS, max(_MIN_SECONDS, self.seconds + delta))
        if updated == self.seconds:
            self.flash = f"seconds {self.seconds:.0f}s max (at limit)"
            return
        self.seconds = updated
        self.flash = f"seconds {self.seconds:.0f}s max"

    def handle_key(self, key: str | None) -> bool:
        """Handle one decoded key and return whether the frame needs a redraw."""
        if key is None:
            return False
        if key == "q":
            self._stop_playback()
            self._done = True
            self.flash = "session ended"
            return True
        if key == "?":
            self.show_help = not self.show_help
            return True
        if key == "1":
            self._play("A")
            return True
        if key == "2":
            self._play("B")
            return True
        if key == " ":
            self._replay()
            return True
        if key == "c":
            self._replay_context()
            return True
        if key == ",":
            self._nudge_seconds(-_SECONDS_STEP)
            return True
        if key == ".":
            self._nudge_seconds(_SECONDS_STEP)
            return True
        if key == "m":
            self.pending_note = self._read_note()
            self.flash = "note queued" if self.pending_note else "note cleared"
            return True
        if key == "z":
            self._undo()
            return True
        if key == "s":
            pair = self.current_pair
            if pair is None:
                return False
            self._stop_playback()
            self._outcomes[pair.pair_id] = "skipped"
            self.flash = "pair skipped"
            self._index += 1
            if self._index >= len(self.pairs):
                self._done = True
            else:
                self._autoplay_current()
            return True
        if key in {"a", "b", "t"}:
            try:
                self._record_verdict("tie" if key == "t" else key)
            except (OSError, ValueError) as exc:
                self.flash = f"record failed: {exc}"
            return True
        if key in {"y", "n", "u"}:
            self.flash = "single-clip labels are not enabled in judge ab yet"
            return True
        return False

    def run(self) -> None:
        width = 84
        self._autoplay_current()
        while not self._done:
            self.draw(width=width)
            for key in self.terminal.read_keys(self.interval):
                self.handle_key(key)
                if self._done:
                    break

    def _presented_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "pair_id": pair.pair_id,
                "path_a": pair.first.path,
                "path_b": pair.second.path,
                "group_a": pair.first.group,
                "group_b": pair.second.group,
                "context_paths": list(pair.context_paths),
                "outcome": self._outcomes[pair.pair_id],
                **pair_context_metadata(pair),
            }
            for pair in self.pairs
        ]

    def finalize(self) -> Path | None:
        if self._finalized:
            return self.sidecar_path
        player_name = getattr(self.playback, "player_name", None)
        payload = {
            "question": self.question.id,
            "prompt": self.question.prompt,
            "listener": self.listener,
            "seconds_played": self.seconds,
            "player": player_name,
            "judgements": len(self._written),
            "pairs": self._presented_rows(),
            **self.sidecar,
        }
        self.sidecar_path = write_session_sidecar(payload, session_id=self.session_id, store_path=self.store_path)
        self._finalized = True
        return self.sidecar_path

    def _focus(self) -> list[str]:
        targeted = (self.sidecar.get("target_metric"), *(self.sidecar.get("disagree") or ()))
        return [metric for metric in targeted if metric]

    def _agreement_lines(self) -> list[str]:
        """Every clip in the pool counts for the store column, so earlier sessions on these sources are pooled in."""
        lookup = self.metric_lookup
        if lookup is None:
            lookup = {pair.first.path: pair.first.metrics for pair in self.pairs}
            lookup.update({pair.second.path: pair.second.metrics for pair in self.pairs})
        metrics = self._focus() or sorted({metric for values in lookup.values() for metric in values})
        if not metrics or not self._written:
            return []
        rows = session_agreement(
            self._written,
            load_preferences(self.store_path),
            lookup,
            question=self.question.id,
            metrics=metrics,
            testing=self.context.get("testing"),
        )
        lines = ["metric agreement (available scored pairs):"]
        for item, all_item in rows:
            overall_text = (
                f"{_agreement_wins(all_item.agreement, all_item.pairs)}/{all_item.pairs} overall, {all_item.verdict}"
                if all_item is not None
                else "no overall coverage"
            )
            lines.append(
                f"  {item.metric}: {_agreement_wins(item.agreement, item.pairs)}/{item.pairs} tonight · {overall_text}"
            )
        return lines

    def _side(self, candidate: Any) -> str:
        """A metric or parameter session's sides share one condition, so what tells them apart is the metric or knob."""
        knob = varied_knob(self.context.get("testing") or "")
        if knob:
            return f"{candidate.label} {knob}={knob_text(candidate.controls.get(knob))}"
        if self.context.get("testing") != "metric":
            return candidate.group
        metrics = candidate.metrics
        values = [f"{metric}={metrics[metric]:.3g}" for metric in self._focus() if metric in metrics]
        return " ".join([candidate.label, *values])

    def reveal(self) -> str:
        self.finalize()
        lines = ["", "Blind session reveal", "=" * 20]
        for pair in self.pairs:
            outcome = self._outcomes[pair.pair_id]
            verdict = {"a": "A", "b": "B", "tie": "tie", "skipped": "skip", "unreached": "—"}.get(outcome, outcome)
            lines.append(f"{pair.pair_id}: {self._side(pair.first)} vs {self._side(pair.second)} → {verdict}")
        lines.append(f"judgements: {len(self._written)}")
        knob = varied_knob(self.context.get("testing") or "")
        if knob:
            judged = [(pair, self._outcomes[pair.pair_id]) for pair in self.pairs]
            order = "first wins–ties–second wins" if knob == "model" else "lower first: lower wins–ties–higher wins"
            lines.append(f"{knob} ({order}):")
            for row in matchups(judged, knob):
                lines.append(f"  {row['low']} vs {row['high']}: {row['low_wins']}–{row['ties']}–{row['high_wins']}")
        else:
            lines.extend(self._agreement_lines())
        if self.sidecar_path is not None:
            lines.append(f"Session sidecar → {self.sidecar_path}")
        return "\n".join(lines)


def run_ab_tui(
    pairs: Sequence[AbPair],
    *,
    store_path: str | Path,
    question: str = "better",
    listener: str = "me",
    seconds: float | None = None,
    session_id: str = "",
    context: Mapping[str, Any] | None = None,
    sidecar: Mapping[str, Any] | None = None,
    metric_lookup: Mapping[str, Mapping[str, float]] | None = None,
    stream: TextIO | None = None,
    terminal: TerminalSession | None = None,
    playback: AbPlayback | None = None,
    note_reader: Callable[[str], str] | None = None,
    print_fn: Callable[..., Any] = print,
) -> list[PreferenceRecord]:
    """Run and reveal a TUI session, restoring the terminal on every exit path."""
    tui = AbTui(
        pairs,
        store_path=store_path,
        question=question,
        listener=listener,
        seconds=seconds,
        session_id=session_id,
        context=context,
        sidecar=sidecar,
        metric_lookup=metric_lookup,
        stream=stream,
        terminal=terminal,
        playback=playback,
        note_reader=note_reader,
        print_fn=print_fn,
    )
    if isinstance(tui.playback, NonBlockingClipPlayer) and not tui.playback.available:
        print_fn("No CLI audio player found (tried ffplay, play, paplay, aplay).")
        return []
    try:
        with tui.terminal:
            if tui.terminal.active:
                tui.run()
            else:
                tui.flash = "TUI requires an interactive terminal"
                tui._done = True
    except KeyboardInterrupt:
        tui._done = True
        tui.flash = "session interrupted"
    finally:
        tui._stop_playback()
        try:
            close = getattr(tui.playback, "close", None)
            if callable(close):
                close()
        finally:
            tui.finalize()
    print_fn(tui.reveal())
    return tui.records
