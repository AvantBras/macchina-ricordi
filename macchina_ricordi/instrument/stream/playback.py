from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# (kind, message, **data) — matches StreamEventLog.emit, so a session can pass
# its event log straight in and have playback errors land in events.jsonl
# instead of scrolling past in the terminal.
EventSink = Callable[..., Any]


def _print_event(kind: str, message: str, **_data: Any) -> None:
    print(f"{kind}: {message}")


class NullPlaybackWorker:
    """No-op stand-in used when playback is disabled or sounddevice is unavailable."""

    def __init__(self, *, crossfade_seconds: float = 0.02) -> None:
        self.crossfade_seconds = float(crossfade_seconds)
        self.enqueued: list[tuple[Path, float]] = []
        self._audio_enqueued_seconds = 0.0
        self._emit: EventSink = _print_event

    def set_event_sink(self, emit: EventSink) -> None:
        self._emit = emit

    def enqueue(
        self,
        path: Path | None,
        skip_seconds: float = 0.0,
        playback_seconds: float | None = None,
    ) -> None:
        if path is None:
            return
        if playback_seconds is not None:
            self._audio_enqueued_seconds += float(playback_seconds)
        self.enqueued.append((path, float(skip_seconds)))

    def buffer_seconds_ahead(self) -> float | None:
        """Null worker doesn't actually play audio, so there's no meaningful buffer."""
        return None

    def current_path(self) -> Path | None:
        """Return no target because this worker does not play audio."""
        return None

    def join(self, timeout: float | None = None) -> None:
        return None


class PlaybackWorker:
    """Background thread that streams round WAVs into a single OutputStream.

    Each round's WAV is trimmed by ``skip_seconds`` (drops the EnCodec echo
    of the previous round's tail) and joined to the previous chunk with a
    short overlap-add crossfade (default 20ms) so the splice is click-free.
    """

    def __init__(self, *, crossfade_seconds: float = 0.02) -> None:
        try:
            import sounddevice
            import soundfile
        except ImportError as exc:
            raise RuntimeError(
                "Live playback requires the optional `instrument` extra. "
                'Install it with `pip install -e ".[instrument]"`.'
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Live playback could not load PortAudio: {exc}. "
                "On Debian/Ubuntu install the system library with "
                "`sudo apt install libportaudio2`."
            ) from exc

        self._sounddevice = sounddevice
        self._soundfile = soundfile
        self._crossfade_seconds = max(0.0, float(crossfade_seconds))
        self._queue: queue.Queue[tuple[Path, float] | None] = queue.Queue()
        self._stream = None
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._fade_n: int = 0
        self._pending_tail: np.ndarray | None = None
        self._short_chunk_warned: bool = False
        self._stream_failed: bool = False
        self._stream_corrupted: bool = False
        self._audio_enqueued_seconds: float = 0.0
        self._started_at_monotonic: float | None = None
        self._emit: EventSink = _print_event
        self._meter_lock = threading.Lock()
        self._current_path: Path | None = None
        self._queued_paths: deque[Path] = deque()
        self._thread = threading.Thread(
            target=self._run,
            name="macchina_ricordi-stream-playback",
            daemon=True,
        )
        self._thread.start()

    def set_event_sink(self, emit: EventSink) -> None:
        """Route playback events to a session log as well as stdout.

        Called from the generator thread before playback starts; the worker
        thread only reads it, so a plain attribute assignment is enough.
        """
        self._emit = emit

    def enqueue(
        self,
        path: Path | None,
        skip_seconds: float = 0.0,
        playback_seconds: float | None = None,
    ) -> None:
        if path is None:
            self._queue.put(None)
            return
        with self._meter_lock:
            if playback_seconds is not None:
                self._audio_enqueued_seconds += float(playback_seconds)
            self._queued_paths.append(path)
        self._queue.put((path, float(skip_seconds)))

    def buffer_seconds_ahead(self) -> float | None:
        """How many seconds of audio are queued (or in PortAudio's buffer) but not yet played.

        Approximation: tracks (sum of enqueued post-trim durations) - (wall-clock since
        the worker started writing). It ignores PortAudio's small internal latency
        (~100ms with latency='high'), and undercounts if playback ever underran
        (because wall-clock kept ticking through the silence). Good enough as a
        steady-state instrument for spotting whether the buffer is growing or
        shrinking across rounds.
        """
        with self._meter_lock:
            enqueued = self._audio_enqueued_seconds
            started_at = self._started_at_monotonic
        if started_at is None:
            return enqueued
        elapsed = time.monotonic() - started_at
        return max(0.0, enqueued - elapsed)

    def current_path(self) -> Path | None:
        """Return the chunk currently being played, or the next queued chunk.

        The generator and TUI use this to capture a pin target at the moment
        the listener presses ``p``. The queue fallback covers the small gap
        between two chunks at a playback boundary.
        """
        with self._meter_lock:
            return self._current_path or (self._queued_paths[0] if self._queued_paths else None)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _ensure_stream(self, sample_rate: int, channels: int) -> bool:
        if self._stream is not None:
            return True
        if self._stream_failed or self._stream_corrupted:
            return False
        try:
            self._stream = self._sounddevice.OutputStream(
                samplerate=int(sample_rate),
                channels=int(channels),
                dtype="float32",
                latency="high",  # ~100ms ALSA buffer; absorbs generation jitter
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            self._emit("playback_error", f"could not open output stream: {exc}", level="error")
            self._stream_failed = True
            self._stream = None
            return False
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._fade_n = int(self._crossfade_seconds * self._sample_rate)
        self._emit(
            "playback_started",
            f"sample_rate={self._sample_rate}Hz channels={self._channels} "
            f"crossfade_ms={self._crossfade_seconds * 1000:.0f}",
            sample_rate=self._sample_rate,
            channels=self._channels,
        )
        return True

    def _safe_write(self, data: np.ndarray) -> bool:
        """Write to the stream, marking it corrupted on any host-level error.

        ALSA underruns surface as ``Unanticipated host error`` exceptions from
        PortAudio. Once the driver is in that state, further writes (and a
        clean ``stop()``) can segfault inside the PortAudio C code — so we
        flip a flag and skip every subsequent stream call, then go through
        the corrupted-path in ``_finalize``.
        """
        if self._stream is None or self._stream_corrupted:
            return False
        try:
            with self._meter_lock:
                if self._started_at_monotonic is None:
                    self._started_at_monotonic = time.monotonic()
            self._stream.write(data)
            return True
        except Exception as exc:  # noqa: BLE001
            self._emit(
                "playback_error",
                f"stream write failed ({exc}); disabling further playback for this "
                "session. This usually means the generator could not keep up with "
                "playback — try a larger --completion-duration or a smaller --tau.",
                level="error",
            )
            self._stream_corrupted = True
            self._pending_tail = None
            return False

    @staticmethod
    def _to_2d(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.reshape(-1, 1)
        return audio

    def _process_chunk(self, path: Path, skip_seconds: float) -> None:
        if self._stream_corrupted:
            return  # Silently drop further chunks; the stream is unusable.
        audio, sample_rate = self._soundfile.read(str(path), dtype="float32", always_2d=False)
        audio = self._to_2d(audio)
        channels = audio.shape[1]
        if not self._ensure_stream(int(sample_rate), channels):
            return
        if int(sample_rate) != self._sample_rate or channels != self._channels:
            self._emit(
                "playback_error",
                f"{path}: sample_rate/channels {sample_rate}/{channels} != stream "
                f"{self._sample_rate}/{self._channels}; skipping chunk",
                level="error",
            )
            return

        skip_samples = max(0, int(skip_seconds * self._sample_rate))
        if skip_samples >= audio.shape[0]:
            self._emit(
                "playback_warning",
                f"{path}: skip_seconds={skip_seconds} exceeds duration; skipping chunk",
                level="warning",
            )
            return
        audio = audio[skip_samples:]

        fade_n = self._fade_n

        if fade_n == 0:
            if self._pending_tail is not None:
                self._safe_write(self._pending_tail)
                self._pending_tail = None
            self._safe_write(audio)
            return

        if audio.shape[0] < 2 * fade_n:
            if not self._short_chunk_warned:
                self._emit(
                    "playback_warning",
                    "chunk shorter than 2*crossfade_seconds; degrading to un-faded "
                    f"write ({audio.shape[0]} < {2 * fade_n} samples)",
                    level="warning",
                )
                self._short_chunk_warned = True
            if self._pending_tail is not None:
                self._safe_write(self._pending_tail)
            self._safe_write(audio)
            self._pending_tail = None
            return

        if self._pending_tail is None:
            self._safe_write(audio[:-fade_n])
        else:
            ramp_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32).reshape(-1, 1)
            ramp_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32).reshape(-1, 1)
            mixed = (self._pending_tail * ramp_out + audio[:fade_n] * ramp_in).astype(np.float32, copy=False)
            if not self._safe_write(mixed):
                return
            body = audio[fade_n:-fade_n]
            if body.shape[0] > 0 and not self._safe_write(body):
                return

        if not self._stream_corrupted:
            self._pending_tail = audio[-fade_n:].copy()

    def _finalize(self) -> None:
        if self._stream is None:
            return

        if self._stream_corrupted:
            # ALSA / PortAudio is in an unusable state. Calling stop() would try
            # to drain the broken buffer, which is what segfaults — go straight
            # to abort() and close() in a defensive swallow.
            try:
                self._stream.abort()
            except Exception as exc:  # noqa: BLE001
                self._emit("playback_warning", f"abort failed: {exc}", level="warning")
            try:
                self._stream.close()
            except Exception as exc:  # noqa: BLE001
                self._emit("playback_warning", f"close failed: {exc}", level="warning")
            self._stream = None
            self._pending_tail = None
            return

        try:
            if self._pending_tail is not None:
                self._safe_write(self._pending_tail)
                self._pending_tail = None
            if not self._stream_corrupted:
                self._stream.stop()
                self._stream.close()
            else:
                # A write during finalize tripped the error path; cleanup
                # falls back to abort().
                try:
                    self._stream.abort()
                except Exception:  # noqa: BLE001, S110
                    pass
                try:
                    self._stream.close()
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception as exc:  # noqa: BLE001
            self._emit("playback_error", f"finalize: {exc}", level="error")
        finally:
            self._stream = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._finalize()
                return
            path, skip = item
            with self._meter_lock:
                if self._queued_paths and self._queued_paths[0] == path:
                    self._queued_paths.popleft()
                else:
                    try:
                        self._queued_paths.remove(path)
                    except ValueError:
                        pass
                self._current_path = path
            try:
                self._process_chunk(path, skip)
            except Exception as exc:  # noqa: BLE001
                # Playback errors must never bring down the generator thread.
                self._emit("playback_error", f"{path}: {exc}", level="error")
            finally:
                with self._meter_lock:
                    if self._current_path == path:
                        self._current_path = None
