"""Temporary audio materialization for shared-context A/B sessions."""

from __future__ import annotations

import tempfile
import wave
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np

from .ab import AbCandidate, AbPair

FADE_SECONDS = 0.020


class ContextAudioError(ValueError):
    """A selected shared-context pair cannot be prepared safely."""


@dataclass(frozen=True)
class PcmFormat:
    channels: int
    sample_width: int
    sample_rate: int

    @property
    def frame_width(self) -> int:
        return self.channels * self.sample_width


@dataclass(frozen=True)
class _PcmWav:
    format: PcmFormat
    frames: bytes
    frame_count: int


def _read_pcm_wav(path: str | Path) -> _PcmWav:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise ContextAudioError(f"{source}: expected uncompressed PCM WAV")
            fmt = PcmFormat(handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            if fmt.channels < 1 or fmt.sample_rate < 1 or fmt.sample_width not in (1, 2, 3, 4):
                raise ContextAudioError(
                    f"{source}: unsupported PCM format "
                    f"({fmt.channels} channels, {fmt.sample_rate} Hz, {fmt.sample_width * 8}-bit)"
                )
            frame_count = handle.getnframes()
            frames = handle.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise ContextAudioError(f"{source}: cannot read PCM WAV ({exc})") from exc
    if len(frames) != frame_count * fmt.frame_width:
        raise ContextAudioError(f"{source}: truncated PCM data")
    return _PcmWav(fmt, frames, frame_count)


def _write_pcm_wav(path: Path, fmt: PcmFormat, frames: bytes) -> None:
    try:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(fmt.channels)
            handle.setsampwidth(fmt.sample_width)
            handle.setframerate(fmt.sample_rate)
            handle.writeframes(frames)
    except (OSError, wave.Error) as exc:
        raise ContextAudioError(f"{path}: cannot write temporary PCM WAV ({exc})") from exc


def _pcm_samples(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return np.frombuffer(frames, dtype=np.uint8).astype(np.int64) - 128
    if sample_width == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.int64)
    if sample_width == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.int64)
    octets = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
    values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
    return np.where(values & 0x800000, values - 0x1000000, values)


def _sample_bytes(samples: np.ndarray, sample_width: int) -> bytes:
    limit = 1 << (sample_width * 8 - 1)
    clipped = np.clip(np.rint(samples), -limit, limit - 1).astype(np.int64)
    if sample_width == 1:
        return (clipped + 128).astype(np.uint8).tobytes()
    if sample_width == 2:
        return clipped.astype("<i2").tobytes()
    if sample_width == 4:
        return clipped.astype("<i4").tobytes()
    unsigned = clipped & 0xFFFFFF
    octets = np.empty((len(unsigned), 3), dtype=np.uint8)
    octets[:, 0] = unsigned & 0xFF
    octets[:, 1] = (unsigned >> 8) & 0xFF
    octets[:, 2] = (unsigned >> 16) & 0xFF
    return octets.tobytes()


def _fade_excerpt(frames: bytes, fmt: PcmFormat, frame_count: int) -> bytes:
    fade_frames = min(round(FADE_SECONDS * fmt.sample_rate), frame_count // 2)
    if fade_frames < 1:
        return frames
    samples = _pcm_samples(frames, fmt.sample_width).reshape(frame_count, fmt.channels)
    gain = np.ones(frame_count, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, fade_frames, endpoint=False)
    gain[:fade_frames] = ramp
    gain[-fade_frames:] = ramp[::-1]
    return _sample_bytes(samples * gain[:, None], fmt.sample_width)


def _validate_output(path: Path, expected_format: PcmFormat, expected_frames: int) -> None:
    prepared = _read_pcm_wav(path)
    if prepared.format != expected_format:
        raise ContextAudioError(f"{path}: temporary WAV format changed during preparation")
    if prepared.frame_count != expected_frames:
        actual = prepared.frame_count / prepared.format.sample_rate
        expected = expected_frames / expected_format.sample_rate
        raise ContextAudioError(f"{path}: temporary WAV duration is {actual:.3f}s; expected {expected:.3f}s")


class FreshAudioPreparer(AbstractContextManager["FreshAudioPreparer"]):
    """Own prompt-trimmed PCM clips for exactly one listening session."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="macchina_ricordi_scoring_audio_")
        self.directory = Path(self._temporary.name)
        self._fresh_cache: dict[tuple[str, float], str] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._temporary.cleanup()

    def prepare_fresh_wav(self, path: str | Path, prompt_duration_seconds: float) -> str:
        key = (str(path), prompt_duration_seconds)
        cached = self._fresh_cache.get(key)
        if cached is not None:
            return cached
        source = _read_pcm_wav(path)
        prompt_frames = round(prompt_duration_seconds * source.format.sample_rate)
        if prompt_frames < 0 or prompt_frames >= source.frame_count:
            duration = source.frame_count / source.format.sample_rate
            raise ContextAudioError(
                f"{path}: prompt duration {prompt_duration_seconds:.3f}s "
                f"leaves no fresh audio in a {duration:.3f}s continuation"
            )
        fresh_frames = source.frame_count - prompt_frames
        output = self.directory / f"fresh_{len(self._fresh_cache):04d}.wav"
        _write_pcm_wav(output, source.format, source.frames[prompt_frames * source.format.frame_width :])
        _validate_output(output, source.format, fresh_frames)
        self._fresh_cache[key] = str(output)
        return str(output)


class ContextAudioPreparer(FreshAudioPreparer):
    """Own fresh-only clips and context reels for exactly one listening session."""

    def _fresh_candidate(self, candidate: AbCandidate) -> AbCandidate:
        output = self.prepare_fresh_wav(candidate.path, candidate.prompt_duration_seconds)
        return replace(candidate, playback_path=output)

    def _context_reel(self, pair: AbPair, index: int) -> str:
        if not pair.context_paths or pair.context_excerpt_seconds is None:
            raise ContextAudioError(f"{pair.pair_id}: shared context is missing")
        if len(pair.context_prompt_durations_seconds) != len(pair.context_paths):
            raise ContextAudioError(f"{pair.pair_id}: shared context prompt metadata is incomplete")
        common_format: PcmFormat | None = None
        excerpts: list[bytes] = []
        excerpt_frames: int | None = None
        for path, prompt_seconds in zip(
            pair.context_paths,
            pair.context_prompt_durations_seconds,
            strict=True,
        ):
            source = _read_pcm_wav(path)
            if common_format is None:
                common_format = source.format
                excerpt_frames = round(pair.context_excerpt_seconds * source.format.sample_rate)
            elif source.format != common_format:
                raise ContextAudioError(
                    f"{pair.pair_id}: context WAV {path} does not match "
                    f"{common_format.sample_rate} Hz/{common_format.channels}ch/"
                    f"{common_format.sample_width * 8}-bit"
                )
            assert excerpt_frames is not None
            prompt_frames = round(prompt_seconds * source.format.sample_rate)
            fresh_frames = source.frame_count - prompt_frames
            if excerpt_frames < 1 or fresh_frames < excerpt_frames:
                fresh_seconds = max(0, fresh_frames) / source.format.sample_rate
                raise ContextAudioError(
                    f"{pair.pair_id}: context WAV {path} has {fresh_seconds:.3f}s fresh audio; "
                    f"needs {pair.context_excerpt_seconds:.3f}s"
                )
            start = (source.frame_count - excerpt_frames) * source.format.frame_width
            tail = source.frames[start:]
            excerpts.append(_fade_excerpt(tail, source.format, excerpt_frames))
        assert common_format is not None and excerpt_frames is not None
        output = self.directory / f"context_{index:04d}.wav"
        _write_pcm_wav(output, common_format, b"".join(excerpts))
        _validate_output(output, common_format, excerpt_frames * len(excerpts))
        return str(output)

    def prepare(self, pairs: Sequence[AbPair]) -> list[AbPair]:
        prepared: list[AbPair] = []
        for index, pair in enumerate(pairs):
            try:
                first = self._fresh_candidate(pair.first)
                second = self._fresh_candidate(pair.second)
                reel = self._context_reel(pair, index)
            except ContextAudioError as exc:
                raise ContextAudioError(f"cannot prepare selected pair {pair.pair_id}: {exc}") from exc
            prepared.append(replace(pair, first=first, second=second, context_reel_path=reel))
        return prepared


def read_pcm_frames(path: str | Path) -> tuple[PcmFormat, bytes, int]:
    """Return prepared PCM data for focused tests and diagnostics."""
    audio = _read_pcm_wav(path)
    return audio.format, audio.frames, audio.frame_count
