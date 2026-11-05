from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_BAR_COUNTS = (1, 2, 4, 8)
DEFAULT_BEATS_PER_BAR = 4
DEFAULT_VOICE_THRESHOLD_DB = 24.0
DEFAULT_MIN_BREATH_SECONDS = 0.08
DEFAULT_PRE_ROLL_SECONDS = 0.03
DEFAULT_FADE_MS = 10.0
DEFAULT_TARGET_SECONDS = 60.0
ANALYSIS_SR = 22_050
ENVELOPE_STEP_SECONDS = 0.01
LAYERS = ("mix", "vocals", "instrumental", "stuck")

_ONSET_HOP = 128
_TEMPO_EXCERPT_SECONDS = 300.0
_REFINE_BEATS = 16
_REFINE_HALF_WINDOW = 16
_TEMPO_PRIOR_BPM = 120.0
_TEMPO_PRIOR_OCTAVES = 1.0
_END_WINDOW_SECONDS = (0.12, 0.01)
_TAIL_MARGIN_SECONDS = 0.1

# Hand-tuned on one song (Hit My Heart). They only order the shortlist; the ear picks the loop.
_VOICE_WEIGHT = 1.5
_BREATH_WEIGHT = 0.8
_BREATH_CAP_SECONDS = 0.6
_END_VOICE_WEIGHT = 0.03
_END_VOICE_FLOOR_DB = -40.0


@dataclass(frozen=True, eq=False)
class VocalActivity:
    envelope_db: np.ndarray
    reference_db: float
    threshold_db: float
    step_seconds: float

    @property
    def active(self) -> np.ndarray:
        return self.envelope_db > self.reference_db - self.threshold_db

    def _frames(self, start_seconds: float, end_seconds: float) -> slice:
        size = self.envelope_db.shape[0]
        first = min(max(0, int(start_seconds / self.step_seconds)), size - 1)
        last = min(max(first + 1, int(end_seconds / self.step_seconds)), size)
        return slice(first, last)

    def level_db(self, start_seconds: float, end_seconds: float) -> float:
        """Mean vocal level over the span, relative to the song's loud-voice reference."""
        return float(np.mean(self.envelope_db[self._frames(start_seconds, end_seconds)]) - self.reference_db)

    def voiced_fraction(self, start_seconds: float, end_seconds: float) -> float:
        return float(np.mean(self.active[self._frames(start_seconds, end_seconds)]))

    def longest_gap_seconds(self, start_seconds: float, end_seconds: float) -> float:
        """Longest stretch without voice inside the span, e.g. an instrumental break inside a loop."""
        quiet = ~self.active[self._frames(start_seconds, end_seconds)]
        if not quiet.any():
            return 0.0
        edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]])))
        return float((edges[1::2] - edges[::2]).max() * self.step_seconds)


@dataclass(frozen=True)
class VocalOnset:
    time_seconds: float
    breath_seconds: float


@dataclass(frozen=True)
class LoopCandidate:
    start_seconds: float
    bars: int
    length_seconds: float
    voice_fraction: float
    end_voice_db: float
    breath_seconds: float
    score: float
    longest_gap_seconds: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, eq=False)
class LoopAnalysis:
    tempo_bpm: float
    bar_seconds: float
    beats_per_bar: int
    bpm_source: str
    duration_seconds: float
    activity: VocalActivity
    onsets: tuple[VocalOnset, ...]
    candidates: tuple[LoopCandidate, ...]


def parse_bar_counts(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse ``"1,2,4,8"`` into unique positive bar counts, order kept."""
    if isinstance(value, str):
        parts: list[object] = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    counts: list[int] = []
    for part in parts:
        try:
            count = int(part)  # type: ignore[call-overload]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bar count: {part!r}") from exc
        if count <= 0:
            raise ValueError(f"Bar counts must be > 0, got {part!r}")
        if count not in counts:
            counts.append(count)
    if not counts:
        raise ValueError("At least one bar count is required, e.g. 1,2,4,8")
    return tuple(counts)


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    if 0 < index < values.shape[0] - 1:
        left, centre, right = values[index - 1 : index + 2]
        curvature = left - 2.0 * centre + right
        if curvature < 0.0:
            return index + 0.5 * (left - right) / curvature
    return float(index)


def estimate_beat_seconds(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    min_bpm: float = 80.0,
    max_bpm: float = 200.0,
) -> float:
    """Beat period from onset-envelope autocorrelation.

    Feed it the instrumental stem: drums repeat exactly while the voice does not. The coarse lag is
    re-measured 16 beats out, which divides the frame quantisation error by 16. A broad log-normal prior
    around 120 BPM (one octave wide, as in librosa's tempo estimator) breaks near-ties between the beat and
    dotted or half-time patterns: Mauro Picotto's Proximus scores its 1.5-beat riff 0.583 against 0.569 for
    the beat. Assumes a steady grid; pass an explicit BPM when the detected one is still off. Only the
    central five minutes are analysed, which keeps long recordings from exhausting memory.
    """
    import librosa

    excerpt = int(_TEMPO_EXCERPT_SECONDS * sample_rate)
    if waveform.shape[-1] > excerpt:
        offset = (waveform.shape[-1] - excerpt) // 2
        waveform = waveform[..., offset : offset + excerpt]
    envelope = librosa.onset.onset_strength(y=waveform, sr=sample_rate, hop_length=_ONSET_HOP)
    envelope = envelope - envelope.mean()
    fps = sample_rate / _ONSET_HOP
    shortest, longest = int(60.0 / max_bpm * fps), int(math.ceil(60.0 / min_bpm * fps))
    ac = librosa.autocorrelate(envelope, max_size=_REFINE_BEATS * longest + _REFINE_HALF_WINDOW + 2)
    if ac.shape[0] <= longest:
        raise ValueError("Audio is too short to estimate a tempo; pass an explicit BPM")

    lags = np.arange(shortest, longest + 1)
    prior = np.exp(-0.5 * (np.log2(60.0 * fps / lags / _TEMPO_PRIOR_BPM) / _TEMPO_PRIOR_OCTAVES) ** 2)
    coarse = shortest + int(np.argmax(ac[shortest : longest + 1] * prior))
    guess = _REFINE_BEATS * coarse
    lo, hi = guess - _REFINE_HALF_WINDOW, guess + _REFINE_HALF_WINDOW + 1
    if hi >= ac.shape[0]:
        return _parabolic_peak(ac, coarse) / fps
    fine = lo + int(np.argmax(ac[lo:hi]))
    return _parabolic_peak(ac, fine) / fps / _REFINE_BEATS


def measure_vocal_activity(
    vocals: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = DEFAULT_VOICE_THRESHOLD_DB,
    step_seconds: float = ENVELOPE_STEP_SECONDS,
) -> VocalActivity:
    """10 ms RMS envelope of the vocal stem; "active" means within ``threshold_db`` of its 95th percentile."""
    import librosa

    step = max(1, int(round(step_seconds * sample_rate)))
    rms = librosa.feature.rms(y=vocals, frame_length=2 * step, hop_length=step)[0]
    envelope_db = 20.0 * np.log10(rms + 1e-9)
    return VocalActivity(
        envelope_db=envelope_db,
        reference_db=float(np.percentile(envelope_db, 95)),
        threshold_db=threshold_db,
        step_seconds=step / sample_rate,
    )


def find_vocal_onsets(
    activity: VocalActivity,
    *,
    min_breath_seconds: float = DEFAULT_MIN_BREATH_SECONDS,
) -> tuple[VocalOnset, ...]:
    """Points where the voice comes back after at least ``min_breath_seconds`` of quiet."""
    active = activity.active
    if active.shape[0] < 2:
        return ()
    last_active = np.maximum.accumulate(np.where(active, np.arange(active.shape[0]), -1))
    rising = np.flatnonzero(active[1:] & ~active[:-1]) + 1
    onsets = []
    for index in rising:
        previous = int(last_active[index - 1])
        breath_frames = index - previous if previous >= 0 else index
        breath = breath_frames * activity.step_seconds
        if breath >= min_breath_seconds - activity.step_seconds / 2.0:
            onsets.append(VocalOnset(time_seconds=index * activity.step_seconds, breath_seconds=breath))
    return tuple(onsets)


def find_loop_candidates(
    activity: VocalActivity,
    onsets: Sequence[VocalOnset],
    *,
    bar_seconds: float,
    duration_seconds: float,
    bar_counts: Sequence[int] = DEFAULT_BAR_COUNTS,
    pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS,
) -> list[LoopCandidate]:
    """One candidate per (phrase onset, bar count), best score first.

    The loop starts a hair before the phrase (inside the breath) and lasts a whole number of bars, so the
    groove never stumbles at the seam. It scores well when it is mostly voice, when the voice is silent at
    the loop end (no word cut in half), and when the phrase starts after a clear breath.
    """
    candidates = []
    for onset in onsets:
        start = onset.time_seconds - min(pre_roll_seconds, onset.breath_seconds / 2.0)
        if start < 0.0:
            continue
        for bars in bar_counts:
            length = bars * bar_seconds
            end = start + length
            if end + _TAIL_MARGIN_SECONDS > duration_seconds:
                continue
            voice = activity.voiced_fraction(start, end)
            end_voice = activity.level_db(end - _END_WINDOW_SECONDS[0], end + _END_WINDOW_SECONDS[1])
            score = (
                _VOICE_WEIGHT * voice
                + _BREATH_WEIGHT * min(onset.breath_seconds, _BREATH_CAP_SECONDS)
                - _END_VOICE_WEIGHT * max(0.0, end_voice - _END_VOICE_FLOOR_DB)
            )
            candidates.append(
                LoopCandidate(
                    start_seconds=round(start, 4),
                    bars=bars,
                    length_seconds=round(length, 6),
                    voice_fraction=round(voice, 4),
                    end_voice_db=round(end_voice, 2),
                    breath_seconds=round(onset.breath_seconds, 3),
                    score=round(score, 4),
                    longest_gap_seconds=round(activity.longest_gap_seconds(start, end), 3),
                )
            )
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.start_seconds))
    return candidates


def shortlist_candidates(
    candidates: Sequence[LoopCandidate],
    *,
    top: int,
    min_spacing_seconds: float,
    min_voice_fraction: float = 0.0,
    max_gap_seconds: float | None = None,
    allow_overlap: bool = True,
) -> list[LoopCandidate]:
    """Best-first, skipping candidates that start within ``min_spacing_seconds`` of one already kept.

    ``min_voice_fraction`` and ``max_gap_seconds`` keep only loops that are voiced all the way through;
    ``allow_overlap=False`` keeps only loops that share no audio with a better one.
    """

    def clashes(candidate: LoopCandidate, other: LoopCandidate) -> bool:
        if abs(candidate.start_seconds - other.start_seconds) < min_spacing_seconds:
            return True
        if allow_overlap:
            return False
        return (
            candidate.start_seconds < other.start_seconds + other.length_seconds
            and other.start_seconds < candidate.start_seconds + candidate.length_seconds
        )

    kept: list[LoopCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.start_seconds)):
        if len(kept) >= top:
            break
        if candidate.voice_fraction < min_voice_fraction:
            continue
        if max_gap_seconds is not None and candidate.longest_gap_seconds > max_gap_seconds:
            continue
        if not any(clashes(candidate, other) for other in kept):
            kept.append(candidate)
    return kept


def analyze_loops(
    vocals: np.ndarray,
    instrumental: np.ndarray,
    sample_rate: int,
    *,
    bpm: float | None = None,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    bar_counts: Sequence[int] = DEFAULT_BAR_COUNTS,
    threshold_db: float = DEFAULT_VOICE_THRESHOLD_DB,
    min_breath_seconds: float = DEFAULT_MIN_BREATH_SECONDS,
    pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS,
) -> LoopAnalysis:
    """Mono stems in, every scored loop candidate out."""
    if bpm is not None and bpm <= 0.0:
        raise ValueError(f"BPM must be > 0, got {bpm!r}")
    beat_seconds = 60.0 / bpm if bpm else estimate_beat_seconds(instrumental, sample_rate)
    bar_seconds = beats_per_bar * beat_seconds
    activity = measure_vocal_activity(vocals, sample_rate, threshold_db=threshold_db)
    onsets = find_vocal_onsets(activity, min_breath_seconds=min_breath_seconds)
    duration = min(vocals.shape[-1], instrumental.shape[-1]) / sample_rate
    candidates = find_loop_candidates(
        activity,
        onsets,
        bar_seconds=bar_seconds,
        duration_seconds=duration,
        bar_counts=bar_counts,
        pre_roll_seconds=pre_roll_seconds,
    )
    return LoopAnalysis(
        tempo_bpm=60.0 / beat_seconds,
        bar_seconds=bar_seconds,
        beats_per_bar=beats_per_bar,
        bpm_source="given" if bpm else "detected",
        duration_seconds=duration,
        activity=activity,
        onsets=onsets,
        candidates=tuple(candidates),
    )


def repeats_for(target_seconds: float, length_seconds: float) -> int:
    return max(1, int(round(target_seconds / length_seconds)))


def candidate_label(start_seconds: float, bars: int) -> str:
    # "0m42s57" rather than "0m42.57s" so tools that split on the first dot keep the whole stem.
    hundredths = int(round(start_seconds * 100))
    minutes, rest = divmod(hundredths, 6000)
    return f"{minutes}m{rest // 100:02d}s{rest % 100:02d}_{bars}bar"


def render_loop(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    start_seconds: float,
    length_seconds: float,
    repeats: int,
    fade_ms: float = DEFAULT_FADE_MS,
    fade_edges: bool = True,
) -> np.ndarray:
    """Repeat ``waveform[start : start + length]`` with a linear crossfade at every seam.

    The crossfade borrows ``fade_ms`` of audio from just after the loop end, so each repeat still starts
    exactly one loop length after the previous one and the groove stays on the grid. It is equal-gain, not
    equal-power: both sides of a bar-aligned seam carry the same drum hits, and an equal-power fade of
    correlated audio overshoots by up to 3 dB, which clips loud masters. The time axis is the last one.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    total = waveform.shape[-1]
    begin = int(round(start_seconds * sample_rate))
    span = int(round(length_seconds * sample_rate))
    if span <= 0:
        raise ValueError(f"Loop length must be > 0, got {length_seconds!r}s")
    if begin < 0 or begin + span > total:
        raise ValueError(
            f"Loop {start_seconds:.3f}s + {length_seconds:.3f}s does not fit in {total / sample_rate:.3f}s of audio"
        )
    fade = min(int(round(fade_ms / 1000.0 * sample_rate)), span // 2, total - begin - span)
    body = waveform[..., begin : begin + span + fade].astype(np.float32)
    fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    fade_out = fade_in[::-1]

    out = np.zeros(waveform.shape[:-1] + (span * repeats + fade,), dtype=np.float32)
    for index in range(repeats):
        chunk = body.copy()
        if index or fade_edges:
            chunk[..., :fade] *= fade_in
        chunk[..., span:] *= fade_out
        out[..., index * span : index * span + span + fade] += chunk
    out = out[..., : span * repeats]
    if fade_edges and fade:
        out[..., -fade:] *= fade_out
    return out


def render_stuck_voice(
    vocals: np.ndarray,
    instrumental: np.ndarray,
    sample_rate: int,
    *,
    start_seconds: float,
    length_seconds: float,
    fade_ms: float = DEFAULT_FADE_MS,
) -> np.ndarray:
    """The song plays as-is until the loop start; from there the voice repeats while the band carries on."""
    total = min(vocals.shape[-1], instrumental.shape[-1])
    begin = int(round(start_seconds * sample_rate))
    span = int(round(length_seconds * sample_rate))
    repeats = max(1, math.ceil((total - begin) / span))
    looped = render_loop(
        vocals[..., :total],
        sample_rate,
        start_seconds=start_seconds,
        length_seconds=length_seconds,
        repeats=repeats,
        fade_ms=fade_ms,
        fade_edges=False,
    )
    voice = np.concatenate([vocals[..., :begin].astype(np.float32), looped], axis=-1)[..., :total]
    out = instrumental[..., :total].astype(np.float32) + voice
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    return out / peak * 0.99 if peak > 0.99 else out
