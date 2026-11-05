"""Single-keystroke blind single-clip labeling interface."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from ..cli.common import clip
from ..instrument.stream.tui import TerminalSession
from .ab import NonBlockingClipPlayer
from .label import (
    LabelRecord,
    LabelSelection,
    analyze_detection,
    append_label,
    get_alarm,
    label_context,
    remove_last_label,
    write_label_sidecar,
)
from .preference import new_session_id
from .questions import get_question

_CLEAR = "\033[H\033[J"
_MIN_SECONDS = 1.0
_MAX_SECONDS = 120.0
_DEFAULT_SHORTENED_SECONDS = 30.0


class LabelTui:
    def __init__(
        self,
        selections: Sequence[LabelSelection],
        *,
        store_path: str | Path,
        listener: str = "me",
        seconds: float | None = None,
        session_id: str = "",
        sidecar: Mapping[str, Any] | None = None,
        stream: TextIO | None = None,
        terminal: TerminalSession | None = None,
        playback: Any = None,
        note_reader: Callable[[str], str] | None = None,
        print_fn: Callable[..., Any] = print,
        interval: float = 0.08,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.selections = list(selections)
        if not self.selections:
            raise ValueError("label TUI needs at least one selection")
        self.alarm = get_alarm(self.selections[0].alarm_id)
        if any(selection.alarm_id != self.alarm.id for selection in self.selections):
            raise ValueError("one labeling session must contain exactly one alarm")
        self.question = get_question(self.alarm.question)
        self.store_path = Path(store_path).expanduser()
        self.listener = listener
        self.seconds = seconds if seconds is None or seconds > 0 else None
        self.session_id = session_id or new_session_id()
        self.sidecar = dict(sidecar or {})
        self.stream = stream
        self.terminal = terminal if terminal is not None else TerminalSession(stream=stream)
        self.playback = playback if playback is not None else NonBlockingClipPlayer()
        self.note_reader = note_reader
        self.print_fn = print_fn
        self.interval = max(0.03, interval)
        self.clock = clock
        self.started_at = clock()
        self.flash: str | None = None
        self.playing = False
        self.pending_note = ""
        self.show_help = False
        self._index = 0
        self._outcomes = {index: "unreached" for index in range(len(self.selections))}
        self._written: list[LabelRecord] = []
        self._done = False
        self._finalized = False
        self._auto_played_index: int | None = None
        self.sidecar_path: Path | None = None

    @property
    def current_selection(self) -> LabelSelection | None:
        if 0 <= self._index < len(self.selections):
            return self.selections[self._index]
        return None

    @property
    def records(self) -> list[LabelRecord]:
        return list(self._written)

    @property
    def done(self) -> bool:
        return self._done

    def _stop(self) -> None:
        try:
            self.playback.stop()
        except (OSError, RuntimeError):
            pass
        self.playing = False

    def _play(self) -> None:
        selection = self.current_selection
        if selection is None:
            return
        self._stop()
        try:
            started = self.playback.play(selection.candidate.playable_path, seconds=self.seconds)
        except (OSError, RuntimeError):
            started = False
        self.playing = started
        self.flash = "playing clip" if started else "could not start audio player"

    def _autoplay_current(self) -> None:
        if self.current_selection is None or self._auto_played_index == self._index:
            return
        self._auto_played_index = self._index
        self._play()

    def _refresh_playing(self) -> None:
        current_path = getattr(self.playback, "current_path", None)
        if callable(current_path):
            self.playing = current_path() is not None

    def _seconds_text(self) -> str:
        return "whole clip" if self.seconds is None else f"{self.seconds:.0f}s max"

    def frame(self, *, width: int = 84) -> str:
        self._refresh_playing()
        width = max(32, width)
        total = len(self.selections)
        progress = self._index / total if total else 1.0
        bar_width = max(10, min(48, width - 28))
        filled = round(progress * bar_width)
        elapsed = max(0.0, self.clock() - self.started_at)
        lines = [
            clip(self.question.prompt, width),
            "",
            clip(f"clip {min(self._index + 1, total)}/{total}  [{'█' * filled}{'·' * (bar_width - filled)}]", width),
            clip(
                f"elapsed {elapsed:.0f}s  ·  "
                f"{'▶ clip' if self.playing else 'nothing playing'}  ·  {self._seconds_text()}",
                width,
            ),
        ]
        if self.flash:
            lines.append(clip(self.flash, width))
        if self.pending_note:
            lines.append("note queued for the next answer")
        if self.show_help:
            lines.extend(
                (
                    "space replay   y/n/u answer   z undo   s skip",
                    "m note   ,/. seconds   q quit   ? help",
                )
            )
        return "\n".join(lines) + "\n"

    def draw(self, *, width: int = 84) -> None:
        if self.stream is None:
            return
        try:
            self.stream.write(_CLEAR + self.frame(width=width))
            self.stream.flush()
        except (OSError, ValueError):
            pass

    def _read_note(self) -> str:
        if self.note_reader is not None:
            return self.note_reader("note: ")[:240].strip()
        reader = getattr(self.terminal, "read_line", None)
        return str(reader("note: ", max_length=240)).strip() if callable(reader) else ""

    def _advance(self) -> None:
        self._index += 1
        if self._index >= len(self.selections):
            self._done = True
        else:
            self._autoplay_current()

    def _record(self, answer: str) -> None:
        selection = self.current_selection
        if selection is None:
            return
        self._stop()
        record = LabelRecord(
            path=selection.candidate.path,
            alarm_id=self.alarm.id,
            question=self.question.id,
            answer=answer,
            listener=self.listener,
            notes=self.pending_note,
            session_id=self.session_id,
            seconds_played=self.seconds,
            context=label_context(selection),
        )
        append_label(record, path=self.store_path)
        self._written.append(record)
        self._outcomes[self._index] = answer
        self.pending_note = ""
        self.flash = f"recorded {answer}; next clip"
        self._advance()

    def _undo(self) -> None:
        if not self._written:
            self.flash = "nothing to undo"
            return
        record = self._written[-1]
        try:
            remove_last_label(record, path=self.store_path)
        except ValueError as exc:
            self.flash = f"undo failed: {exc}"
            return
        self._written.pop()
        for index, selection in enumerate(self.selections):
            if selection.candidate.path == record.path and selection.alarm_id == record.alarm_id:
                self._index = index
                self._outcomes[index] = "unreached"
                break
        self._done = False
        self._auto_played_index = None
        self.flash = "last answer undone"
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
        self.seconds = updated
        self.flash = f"seconds {updated:.0f}s max"

    def handle_key(self, key: str | None) -> bool:
        if key is None:
            return False
        if key == "q":
            self._stop()
            self._done = True
        elif key == "?":
            self.show_help = not self.show_help
        elif key == " ":
            self._play()
        elif key in {"y", "n", "u"}:
            try:
                self._record({"y": "yes", "n": "no", "u": "unsure"}[key])
            except (OSError, ValueError) as exc:
                self.flash = f"record failed: {exc}"
        elif key == "z":
            self._undo()
        elif key == "s":
            if self.current_selection is None:
                return False
            self._stop()
            self._outcomes[self._index] = "skipped"
            self.flash = "clip skipped"
            self._advance()
        elif key == "m":
            self.pending_note = self._read_note()
            self.flash = "note queued" if self.pending_note else "note cleared"
        elif key == ",":
            self._nudge_seconds(-1.0)
        elif key == ".":
            self._nudge_seconds(1.0)
        else:
            return False
        return True

    def run(self) -> None:
        self._autoplay_current()
        while not self._done:
            self.draw()
            for key in self.terminal.read_keys(self.interval):
                self.handle_key(key)
                if self._done:
                    break

    def _presented_rows(self) -> list[dict[str, Any]]:
        from .label_session import _presented_row

        return [_presented_row(selection, self._outcomes[index]) for index, selection in enumerate(self.selections)]

    def finalize(self) -> Path | None:
        if self._finalized:
            return self.sidecar_path
        self.sidecar_path = write_label_sidecar(
            {
                "alarm_id": self.alarm.id,
                "question": self.question.id,
                "prompt": self.question.prompt,
                "listener": self.listener,
                "seconds_played": self.seconds,
                "player": getattr(self.playback, "player_name", None),
                "labels": len(self._written),
                "clips": self._presented_rows(),
                **self.sidecar,
            },
            session_id=self.session_id,
            store_path=self.store_path,
        )
        self._finalized = True
        return self.sidecar_path

    def reveal(self) -> str:
        self.finalize()
        lines = ["", "Blind label session reveal", "=" * 26]
        for index, selection in enumerate(self.selections):
            value = selection.candidate.audit_values[self.alarm.predictor]
            lines.append(
                f"{Path(selection.candidate.path).name}: {self._outcomes[index]} · "
                f"{self.alarm.predictor}={value:.6g} · {selection.stratum_label}"
            )
        if self._written:
            report = analyze_detection(
                self._written,
                alarm_id=self.alarm.id,
                min_labels=8,
                session_id=self.session_id,
            )
            frame = report.frames[0] if report.frames else None
            if frame and frame.recommended_threshold is not None:
                lines.append(f"session max-F1 threshold: {frame.recommended_threshold:.6g}")
        lines.append(f"labels: {len(self._written)}")
        if self.sidecar_path:
            lines.append(f"Session sidecar → {self.sidecar_path}")
        return "\n".join(lines)


def run_label_tui(
    selections: Sequence[LabelSelection],
    *,
    store_path: str | Path,
    listener: str = "me",
    seconds: float | None = None,
    session_id: str = "",
    sidecar: Mapping[str, Any] | None = None,
    stream: TextIO | None = None,
    terminal: TerminalSession | None = None,
    playback: Any = None,
    note_reader: Callable[[str], str] | None = None,
    print_fn: Callable[..., Any] = print,
) -> list[LabelRecord]:
    tui = LabelTui(
        selections,
        store_path=store_path,
        listener=listener,
        seconds=seconds,
        session_id=session_id,
        sidecar=sidecar,
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
                tui._done = True
    except KeyboardInterrupt:
        tui._done = True
    finally:
        tui._stop()
        close = getattr(tui.playback, "close", None)
        if callable(close):
            close()
        tui.finalize()
    print_fn(tui.reveal())
    return tui.records
