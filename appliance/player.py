"""Turning a model into sound: the delay pattern, sampling, and a rolling loop.

Written against numpy and a three-method backend, so the same loop runs on the
shipped ONNX runtime and on torch during development — and nothing here can pull
torch into a build by accident.

The delay pattern is what lets one transformer step emit all four codebooks.
Codebook ``k`` is shifted ``delays[k]`` steps later in the sequence, so at any
step the model predicts codebook 0 of frame *t*, codebook 1 of frame *t-1*, and
so on — each conditioned on everything coarser that precedes it. Generating
``T`` frames costs ``T + max(delays)`` steps.

The loop keeps a **fixed** window of recent tokens and re-primes on it each
round. The context never grows, so minute sixty costs what minute one costs.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from macchina_ricordi.instrument.controls import (
    ROUND,
    TAPE_SPEED,
    TAU,
    TEMPERATURE,
    TILT_HIGH,
    TILT_LOW,
    TILT_MID,
    TOP_K,
)

from .spec import ModelSpec
from .steering import INJECT_HEAVY, INJECT_LIGHT, RECAP_MIX, Tilt, apply_tilt, blend, resample_to


class Backend(Protocol):
    """What the loop needs from a runtime, and no more."""

    spec: ModelSpec

    def start(self, batch: int, horizon: int) -> object: ...

    def step(self, tokens: np.ndarray, offset: int, state: object) -> tuple[np.ndarray, object]:
        """``[B, n_q, T]`` tokens at ``offset`` -> ``[B, n_q, T, card]`` logits, new state."""
        ...

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """``[1, n_q, T]`` codes -> mono float32 samples."""
        ...

    # Optional. Backends that can turn audio back into codes set ``can_encode``
    # and provide ``encode(samples) -> [1, n_q, T]``; the prompt-steering controls
    # need it, everything else does not.


MIN_TAU, MAX_TAU = TAU.appliance
MIN_ROUND, MAX_ROUND = ROUND.appliance


@dataclass(frozen=True)
class Knobs:
    """The controls a listener actually turns, read afresh every round.

    ``tau_seconds`` and ``round_seconds`` left at ``None`` mean "whatever the
    player was built with". The steering fields change what the model hears of
    its own past (see ``appliance.steering``); at their defaults they cost nothing.
    """

    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    tau_seconds: float | None = None
    round_seconds: float | None = None
    tilt_low_db: float = 0.0
    tilt_mid_db: float = 0.0
    tilt_high_db: float = 0.0
    prompt_rate: float = 1.0
    recap_every: int = 0

    def clamped(self) -> Knobs:
        def within(value, low, high):
            return max(low, min(high, value))

        # ranges are the shared vocabulary's (macchina_ricordi/instrument/controls.py),
        # the appliance's own where narrower
        return Knobs(
            temperature=TEMPERATURE.clamp(self.temperature, appliance=True),
            top_k=int(TOP_K.clamp(int(self.top_k), appliance=True)),
            top_p=within(self.top_p, 0.0, 1.0),
            tau_seconds=None if self.tau_seconds is None else TAU.clamp(self.tau_seconds, appliance=True),
            round_seconds=None if self.round_seconds is None else ROUND.clamp(self.round_seconds, appliance=True),
            tilt_low_db=TILT_LOW.clamp(self.tilt_low_db, appliance=True),
            tilt_mid_db=TILT_MID.clamp(self.tilt_mid_db, appliance=True),
            tilt_high_db=TILT_HIGH.clamp(self.tilt_high_db, appliance=True),
            prompt_rate=TAPE_SPEED.clamp(self.prompt_rate, appliance=True),
            recap_every=within(int(self.recap_every), 0, 16),
        )

    @property
    def tilt(self) -> Tilt:
        return Tilt(self.tilt_low_db, self.tilt_mid_db, self.tilt_high_db)

    @property
    def steers(self) -> bool:
        return not self.tilt.is_flat or abs(self.prompt_rate - 1.0) > 1e-6


def sample_token(logits: np.ndarray, knobs: Knobs, rng: np.random.Generator | None = None) -> np.ndarray:
    """``[B, n_q, card]`` logits -> ``[B, n_q, 1]`` int64 tokens."""
    if knobs.temperature <= 0:
        return logits.argmax(axis=-1)[..., None].astype(np.int64)
    rng = rng or np.random.default_rng()
    z = logits.astype(np.float64) / knobs.temperature
    z -= z.max(axis=-1, keepdims=True)
    probs = np.exp(z)
    card = probs.shape[-1]
    if knobs.top_p > 0:
        order = np.argsort(-probs, axis=-1)
        ranked = np.take_along_axis(probs, order, axis=-1)
        ranked /= ranked.sum(axis=-1, keepdims=True)
        cumulative = np.cumsum(ranked, axis=-1)
        ranked = np.where(cumulative - ranked > knobs.top_p, 0.0, ranked)
        return _draw(ranked, order, rng)
    if 0 < knobs.top_k < card:
        order = np.argpartition(-probs, knobs.top_k - 1, axis=-1)[..., : knobs.top_k]
        return _draw(np.take_along_axis(probs, order, axis=-1), order, rng)
    return _draw(probs, None, rng)


def _draw(weights: np.ndarray, order: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(weights / weights.sum(axis=-1, keepdims=True), axis=-1)
    u = rng.random(cdf.shape[:-1] + (1,))
    pick = np.minimum((cdf < u).sum(axis=-1, keepdims=True), cdf.shape[-1] - 1)
    if order is not None:
        pick = np.take_along_axis(order, pick, axis=-1)
    return pick.astype(np.int64)


class Player:
    """A backend, made playable: generates tokens, decodes them, rolls forward.

    Rounds overlap on purpose. Each round re-primes on a window of recent tokens
    and decodes with real audio on the left of what it emits, and — after
    ``lookahead_frames`` — real audio on the right too.

    The overlap also makes a crossfade the right tool at the join, the opposite
    of the nano stack's conclusion for its own decoder. There the two sides of a
    seam were different instants; here the same instant is rendered twice, once
    near a window edge and once from the middle, and blending two renderings of
    one moment is what a crossfade is for. Boundary discontinuity over interior
    on ``nano_disco_60ep``: 13.0 raw, 6.0 with lookahead alone, 1.35 with both,
    against a random-interior-point control of about 1.0.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        context_seconds: float = 5.0,
        round_seconds: float = 4.0,
        lookahead_frames: int = 4,
        crossfade_ms: float = 20.0,
        decode_context_seconds: float | None = None,
        silence_floor: float = 3e-3,
        revive_after: int = 2,
        seed: int | None = None,
    ) -> None:
        self.backend = backend
        self.spec = spec = backend.spec
        self.default_tau = float(context_seconds)
        self.default_round = float(round_seconds)
        self.lookahead = max(0, int(lookahead_frames))
        self.samples_per_frame = int(round(spec.sample_rate / spec.frame_rate))
        self.crossfade = int(crossfade_ms * spec.sample_rate / 1000)
        self.decode_context = (
            None if decode_context_seconds is None else max(1, int(round(decode_context_seconds * spec.frame_rate)))
        )
        self.silence_floor = silence_floor
        self.revive_after = revive_after
        self.rng = np.random.default_rng(seed)
        # Pinned moments belong to the performer, not to the stream state, so a
        # reset or a revival does not forget them.
        self.motifs: list[np.ndarray] = []
        self._inject: float | None = None
        self.context_frames, self.round_frames = self.frames_for(Knobs())
        self.reset()

    @property
    def latency_seconds(self) -> float:
        """Audio held back so the decoder always has future context for it."""
        return self.lookahead / self.spec.frame_rate

    @property
    def can_steer(self) -> bool:
        return bool(getattr(self.backend, "can_encode", False))

    @property
    def horizon_frames(self) -> int:
        """Past the trained segment the model's positions were never seen."""
        return int(self.spec.segment_duration * self.spec.frame_rate) - self.spec.max_delay - 1

    def frames_for(self, knobs: Knobs) -> tuple[int, int]:
        """(context, round) in frames for these knobs, kept inside the model's horizon.

        The round gives way before the memory does: the memory is what the music
        is continuing from, the round only how far it runs before listening again.
        """
        rate = self.spec.frame_rate
        tau = self.default_tau if knobs.tau_seconds is None else knobs.tau_seconds
        length = self.default_round if knobs.round_seconds is None else knobs.round_seconds
        context = max(1, int(round(tau * rate)))
        fresh = max(1, int(round(length * rate)))
        room = self.horizon_frames
        if context + fresh > room:
            fresh = max(1, room - context)
            context = max(1, min(context, room - fresh))
        return context, fresh

    def pin(self, audio: np.ndarray, *, keep: int = 16) -> int:
        """Remember a moment. Returns how many are held."""
        self.motifs.append(np.asarray(audio, dtype=np.float32).copy())
        del self.motifs[:-keep]
        return len(self.motifs)

    def bring_back(self, strength: str = "light") -> None:
        """Blend the newest pinned moment into the next round's past, once."""
        self._inject = {"light": INJECT_LIGHT, "heavy": INJECT_HEAVY}[strength]

    def reset(self) -> None:
        self._history: np.ndarray | None = None
        self._prev_decode: np.ndarray | None = None
        self._prev_start = 0
        self._emitted = 0
        self._window_start = 0
        self._silent_rounds = 0
        self.revivals = 0
        self.rounds_played = 0

    # ---- token generation ----------------------------------------------

    def _generate_frames(self, n_frames: int, knobs: Knobs, prompt=...) -> np.ndarray:
        """Sample ``n_frames`` more frames, primed on the rolling history.

        Returns the **whole** window (prompt plus new frames), ``[1, n_q, T]``.
        One take per round: choosing between several would mean scoring them,
        and scoring in real time costs more than a CPU has to spare.
        """
        batch = 1
        spec = self.spec
        K, special, delays = spec.n_q, spec.special_token, spec.delays
        if prompt is ...:
            prompt = self._history
        prompt_len = 0 if prompt is None else prompt.shape[-1]
        total = prompt_len + n_frames

        # One all-special BOS step, then frame t of codebook k at 1 + t + delays[k].
        seq_len = 1 + total + spec.max_delay
        seq = np.full((batch, K, seq_len), special, dtype=np.int64)
        if prompt is not None:
            src = np.broadcast_to(prompt, (batch, K, prompt_len))
            for k in range(K):
                seq[:, k, 1 + delays[k] : 1 + delays[k] + prompt_len] = src[:, k]

        valid = np.zeros((K, seq_len), dtype=bool)
        for k in range(K):
            valid[k, 1 + delays[k] : 1 + delays[k] + total] = True

        state = self.backend.start(batch, seq_len)
        # Everything the prompt fully determines is one batched prime; only the
        # steps after it must run one at a time.
        start = 1 + prompt_len
        offset = 0
        if start > 1:
            _, state = self.backend.step(seq[:, :, : start - 1], 0, state)
            offset = start - 1

        for pos in range(start, seq_len):
            logits, state = self.backend.step(seq[:, :, pos - 1 : pos], offset, state)
            offset += 1
            token = sample_token(logits[:, :, -1], knobs, self.rng)
            token = np.where(valid[:, pos][None, :, None], token, special)
            written = seq[:, :, pos : pos + 1]
            seq[:, :, pos : pos + 1] = np.where(written == special, token, written)

        return np.stack([seq[:, k, 1 + delays[k] : 1 + delays[k] + total] for k in range(K)], axis=1)

    # ---- the loop -------------------------------------------------------

    def round(self, knobs: Knobs) -> tuple[np.ndarray, dict]:
        """One round of audio, plus what it cost."""
        knobs = knobs.clamped()
        t0 = time.perf_counter()
        self.context_frames, self.round_frames = self.frames_for(knobs)
        if self._history is not None and self._history.shape[-1] > self.context_frames:
            # The memory was shortened: forget the oldest part of the past.
            drop = self._history.shape[-1] - self.context_frames
            self._history = self._history[:, :, drop:]
            self._window_start += drop
        played = self._history
        prompt, steering = self._steered_prompt(knobs)
        prompt_len = 0 if played is None else played.shape[-1]
        window = self._generate_frames(self.round_frames, knobs, prompt=played if prompt is None else prompt)
        fresh = window[:, :, prompt_len:]
        # Decode the past as it was played, not as it was steered: the steering
        # changes what the model continues from, never what anyone hears again.
        chosen = fresh if played is None else np.concatenate([played, fresh], axis=-1)
        self._history = chosen[:, :, -self.context_frames :].copy()

        window_end = self._window_start + chosen.shape[-1]
        emit_to = max(window_end - self.lookahead, self._emitted)
        decode_from = self._window_start
        if self.decode_context is not None:
            decode_from = max(self._window_start, self._emitted - self.decode_context)
        full = self.backend.decode(chosen[:, :, decode_from - self._window_start :])
        spf = self.samples_per_frame
        chunk = self._blend(full[(self._emitted - decode_from) * spf : (emit_to - decode_from) * spf])

        self._prev_decode, self._prev_start = full, decode_from
        self._emitted = emit_to
        # The next window opens where the retained history begins, which is
        # shorter than ``context_frames`` until the history has filled.
        self._window_start = window_end - self._history.shape[-1]
        revived = self._watch_for_death(chunk)
        self.rounds_played += 1
        wall = time.perf_counter() - t0
        seconds = len(chunk) / self.spec.sample_rate
        return chunk, {
            "wall_s": wall,
            "audio_s": seconds,
            "xrt": seconds / max(wall, 1e-9),
            "rms": _rms(chunk),
            "revived": revived,
            "tau_s": self.context_frames / self.spec.frame_rate,
            "round_s": self.round_frames / self.spec.frame_rate,
            **steering,
        }

    def _motif_for(self, knobs: Knobs) -> tuple[np.ndarray | None, float, str]:
        """Which pinned moment, if any, this round's past should carry, and how much."""
        if not self.motifs:
            return None, 0.0, ""
        if self._inject is not None:
            return self.motifs[-1], self._inject, "bring back"
        period = knobs.recap_every
        if period and (self.rounds_played + 1) % period == 0:
            which = (self.rounds_played // period) % len(self.motifs)
            return self.motifs[which], RECAP_MIX, "recap"
        return None, 0.0, ""

    def _steered_prompt(self, knobs: Knobs) -> tuple[np.ndarray | None, dict]:
        """The past the model continues from this round, when a control changes it.

        Decodes nothing new: the audio of the recent past is already sitting in
        the previous round's decode. It is blended with a pinned moment, played
        back at tape speed, tilted, and encoded back into codes the same length as
        the history, so nothing about the round's arithmetic moves.
        """
        motif, mix, reason = self._motif_for(knobs)
        info: dict = {"steered": False}
        history, decoded = self._history, self._prev_decode
        if history is None or decoded is None or not (knobs.steers or motif is not None):
            return None, info
        if not self.can_steer:
            return None, {**info, "steer_skipped": "this model file has no encoder"}
        if reason == "bring back":
            self._inject = None
        spf = self.samples_per_frame
        frames = history.shape[-1]
        target = frames * spf
        end = (self._window_start + frames - self._prev_start) * spf
        take = max(1, min(int(round(knobs.prompt_rate * target)), end))
        tail = decoded[end - take : end]
        if motif is not None:
            tail = blend(tail, motif, mix)
        audio = tail if take == target else resample_to(tail, target)
        audio = apply_tilt(audio, self.spec.sample_rate, knobs.tilt)
        t0 = time.perf_counter()
        codes = np.asarray(self.backend.encode(audio), dtype=np.int64)[:, :, -frames:]
        encode_s = time.perf_counter() - t0
        if codes.shape[-1] < frames:
            codes = np.concatenate([history[:, :, : frames - codes.shape[-1]], codes], axis=-1)
        return codes, {
            "steered": True,
            "rate_applied": round(take / target, 4),
            "tilt_db": [knobs.tilt_low_db, knobs.tilt_mid_db, knobs.tilt_high_db],
            "motif": reason or None,
            "motif_mix": mix,
            "encode_s": encode_s,
        }

    def _blend(self, chunk: np.ndarray) -> np.ndarray:
        """Crossfade the join with the previous window's rendering of the same instants."""
        xf = self.crossfade
        if xf <= 0 or self._prev_decode is None or len(chunk) <= xf:
            return chunk
        offset = (self._emitted - self._prev_start) * self.samples_per_frame
        tail = self._prev_decode[offset : offset + xf]
        if len(tail) < xf:
            # The previous window did not reach this far; nothing honest to blend.
            return chunk
        ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
        chunk = chunk.copy()
        chunk[:xf] = tail * (1.0 - ramp) + chunk[:xf] * ramp
        return chunk

    def _watch_for_death(self, chunk: np.ndarray) -> bool:
        """Silence is an absorbing state here; the context is what absorbs.

        One-shot generation recovers from these models' silent attractor because
        every take starts cold. A rolling stream does not: once the history is
        silence the model predicts more silence from it. On ``nano_disco_60ep``
        at temperature 1.0 a five-minute stream went quiet inside the first
        minute and never came back.

        The probed temperature in each box avoids the attractor; this is the
        backstop for when someone turns the knob down anyway.
        """
        if self.revive_after <= 0 or not len(chunk):
            return False
        if _rms(chunk) >= self.silence_floor:
            self._silent_rounds = 0
            return False
        self._silent_rounds += 1
        if self._silent_rounds < self.revive_after:
            return False
        self._history = None
        self._prev_decode = None
        self._window_start = self._emitted
        self._silent_rounds = 0
        self.revivals += 1
        return True

    def stream(self, knobs_source, *, rounds: int | None = None) -> Iterator[tuple[np.ndarray, dict]]:
        """Endless rounds. ``knobs_source`` is called per round so controls stay live."""
        index = 0
        while rounds is None or index < rounds:
            audio, info = self.round(knobs_source())
            info["round"] = index
            yield audio, info
            index += 1


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt((chunk.astype(np.float64) ** 2).mean())) if len(chunk) else 0.0
