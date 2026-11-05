"""Keeping the speakers fed while the model thinks.

Generation is bursty — a round arrives all at once — and playback is not, so the
two need a buffer between them and a thread each. The whole design rests on one
measured number: a round of audio is produced several times faster than it
plays, so the buffer fills faster than it drains and the only real risk is the
*first* round, before anything has been produced at all.

That risk is handled by prefilling before the stream opens rather than by making
the buffer large. A large buffer would hide underruns by adding latency to every
knob change, which for an instrument is the worse trade: turning a knob should be
audible soon, and a gap should be visible when it happens rather than smoothed
over.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .player import Knobs, Player


@dataclass
class Health:
    """What the UI shows and what a session log keeps."""

    rounds: int = 0
    buffered_seconds: float = 0.0
    underruns: int = 0
    last_xrt: float = 0.0
    mean_xrt: float = 0.0
    worst_xrt: float = 0.0
    played_seconds: float = 0.0
    walls: list[float] = field(default_factory=list)
    audio_seconds: float = 0.0

    def note_round(self, info: dict) -> None:
        """Rounds can change length between one and the next, so ratios are per round."""
        self.rounds += 1
        self.walls.append(info["wall_s"])
        self.audio_seconds += info["audio_s"]
        self.last_xrt = info["xrt"]
        self.mean_xrt = self.audio_seconds / max(sum(self.walls), 1e-9)
        self.worst_xrt = min(self.worst_xrt, info["xrt"]) if self.rounds > 1 else info["xrt"]

    def snapshot(self) -> dict:
        return {
            "rounds": self.rounds,
            "buffered_seconds": round(self.buffered_seconds, 2),
            "underruns": self.underruns,
            "last_xrt": round(self.last_xrt, 2),
            "mean_xrt": round(self.mean_xrt, 2),
            "worst_xrt": round(self.worst_xrt, 2),
            "played_seconds": round(self.played_seconds, 1),
        }


class RingBuffer:
    """A deque of chunks with a sample counter. Simple beats clever at this size."""

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._available = 0
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(chunk)
            self._available += len(chunk)

    def available(self) -> int:
        with self._lock:
            return self._available

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._available = 0

    def take(self, n: int) -> tuple[np.ndarray, int]:
        """Pop ``n`` samples, zero-padding on starvation. Returns (audio, shortfall)."""
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                head = self._chunks[0]
                take = min(len(head), n - filled)
                out[filled : filled + take] = head[:take]
                if take == len(head):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = head[take:]
                filled += take
                self._available -= take
        return out, n - filled


class Engine:
    """A player, a buffer, a producer thread, and somewhere to send the result."""

    def __init__(
        self,
        player: Player,
        *,
        knobs: Knobs | None = None,
        prefill_rounds: int = 2,
        max_buffered_seconds: float = 8.0,
        gain: float = 0.9,
    ) -> None:
        self.player = player
        self.buffer = RingBuffer()
        self.health = Health()
        self.gain = gain
        self.prefill_rounds = max(1, prefill_rounds)
        self.max_buffered = max_buffered_seconds
        self._knobs = (knobs or Knobs()).clamped()
        self._knobs_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._listeners: list = []
        # What was produced, by position, so a pin can find what is audible now
        # rather than what was generated last — the buffer runs seconds ahead.
        self._produced: deque[tuple[int, np.ndarray]] = deque()
        self._produced_samples = 0
        self._played_samples = 0
        self._log_lock = threading.Lock()

    # ---- controls -------------------------------------------------------

    @property
    def knobs(self) -> Knobs:
        with self._knobs_lock:
            return self._knobs

    def set_knobs(self, **changes) -> Knobs:
        with self._knobs_lock:
            current = self._knobs.__dict__ | changes
            self._knobs = Knobs(**current).clamped()
            return self._knobs

    def on_round(self, callback) -> None:
        self._listeners.append(callback)

    # ---- production -----------------------------------------------------

    def _deliver(self, audio: np.ndarray, info: dict) -> None:
        """Queue a round for the speakers, and remember it for pinning."""
        sr = self.player.spec.sample_rate
        self.buffer.push(np.clip(audio * self.gain, -1.0, 1.0).astype(np.float32))
        with self._log_lock:
            self._produced.append((self._produced_samples, audio))
            self._produced_samples += len(audio)
            while self._produced and self._produced_samples - self._produced[0][0] > 40 * sr:
                self._produced.popleft()
        self.health.note_round(info)
        self.health.buffered_seconds = self.buffer.available() / sr

    def _produce_forever(self) -> None:
        sr = self.player.spec.sample_rate
        while not self._stop.is_set():
            # Checked before producing, so the buffer overshoots this by at
            # most one round. That ceiling is what bounds how long a knob change
            # takes to become audible, which is why it is not simply large.
            if self.buffer.available() / sr >= self.max_buffered:
                time.sleep(0.05)
                continue
            try:
                audio, info = self.player.round(self.knobs)
            except Exception as exc:  # a dead producer must not be a silent one
                self.health.underruns += 1
                for cb in self._listeners:
                    cb({"error": repr(exc)})
                self._stop.set()
                raise
            self._deliver(audio, info)
            for cb in self._listeners:
                cb(info)

    def prefill(self) -> None:
        """Produce the opening rounds before anything opens the speakers."""
        for _ in range(self.prefill_rounds):
            audio, info = self.player.round(self.knobs)
            self._deliver(audio, info)

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._produce_forever, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=5.0)
        self._worker = None

    # ---- consumption ----------------------------------------------------

    def pull(self, n: int) -> np.ndarray:
        audio, shortfall = self.buffer.take(n)
        self._played_samples += n - shortfall
        if shortfall:
            self.health.underruns += 1
        sr = self.player.spec.sample_rate
        self.health.buffered_seconds = self.buffer.available() / sr
        self.health.played_seconds += n / sr
        return audio

    def play(self, *, device=None, blocksize: int = 2048, seconds: float | None = None) -> None:
        """Open the speakers and keep them fed until interrupted."""
        import sounddevice as sd

        sr = self.player.spec.sample_rate

        def callback(outdata, frames, _time, status):  # noqa: ANN001
            outdata[:, 0] = self.pull(frames)

        self.prefill()
        self.start()
        with sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=device,
            callback=callback,
        ):
            deadline = None if seconds is None else time.monotonic() + seconds
            try:
                while not self._stop.is_set():
                    if deadline and time.monotonic() >= deadline:
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        self.stop()

    def pin(self, *, at_least_seconds: float = 5.0, playing: bool = True) -> int:
        """Pin the round that is audible now, reaching back to at least ``at_least_seconds``.

        With nothing playing (a render), the newest round is what is "now".
        Returns how many moments are pinned.
        """
        sr = self.player.spec.sample_rate
        with self._log_lock:
            chunks = list(self._produced)
            position = self._played_samples if playing else self._produced_samples - 1
        if not chunks:
            raise ValueError("nothing has played yet")
        index = next((i for i, (start, a) in enumerate(chunks) if start <= position < start + len(a)), len(chunks) - 1)
        audio = np.concatenate([a for _, a in chunks[: index + 1]])
        keep = max(len(chunks[index][1]), int(at_least_seconds * sr))
        return self.player.pin(audio[-keep:])

    def render(self, seconds: float) -> np.ndarray:
        """Produce ``seconds`` of audio with no audio device involved."""
        sr = self.player.spec.sample_rate
        needed = int(seconds * sr)
        pieces: list[np.ndarray] = []
        have = 0
        while have < needed:
            audio, info = self.player.round(self.knobs)
            chunk = np.clip(audio * self.gain, -1.0, 1.0).astype(np.float32)
            pieces.append(chunk)
            have += len(chunk)
            with self._log_lock:
                self._produced.append((self._produced_samples, audio))
                self._produced_samples += len(audio)
            self.health.note_round(info)
            for cb in self._listeners:
                cb(info)
        return np.concatenate(pieces)[:needed]
