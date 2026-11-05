"""The stream's own scoring path: exactly the readings the instrument uses.

The round loop used to call the full arrhythmic scorer (29 fields) and the full
arc scorer for every round *and every best-of-N candidate*. On a 10 s round that
was ~850 ms — more than the generation it was scoring, since fastgen runs the
model at ~8.7x realtime. Almost all of it went on fields nothing was reading:
Phase 1 found the 38-field payload has an effective dimensionality of 5.6, and
the TUI now draws six numbers.

Measured on a real 10 s round (RTX 3060 box, 2026-08-06):

===================================================  ========  ======
component                                            was       now
===================================================  ========  ======
timbral coherence via ``compute_similarity_waveform``  255 ms    10 ms
``chroma_drift`` (two extra chroma-CQTs)                97 ms       —
arc_tv4 (four chroma-CQTs)                             190 ms   190 ms
hpss @ 8 kHz -> ``percussive_energy_ratio``            186 ms   186 ms
the other 26 arrhythmic fields                          94 ms       —
onset analysis -> ``onset_density``                     16 ms    16 ms
silence -> ``silence_fraction``                          7 ms     7 ms
===================================================  ========  ======

853 ms -> 410 ms per round, and the same again per candidate. The single
largest item was not a metric at all but waste: ``timbral_coherence`` only ever
used ``mfcc_cosine``, while ``compute_similarity_waveform`` also built the
spectral and rhythm vectors and threw them away — and the rhythm vector runs
the whole arrhythmic scorer on each half. That fix landed in
``macchina_ricordi.scoring.arc`` and costs nothing; the rest is the fields going away.

**What this gives up.** Stream sidecars from now on carry nine numbers where
they used to carry thirty-eight, so a future stream-statistics pass cannot ask
the 2026-08 questions of 2026-08-06-and-later sessions. That is the deliberate
trade: those questions have been asked and answered, and the answer was that
most of the fields restate each other. Offline scoring is untouched — ``score
file``, ``score dir`` and the sweeps still compute everything, and that is
where a corpus for a *new* question should come from.

``percussive_energy_ratio`` is now 45% of what remains, all of it in the HPSS
that isolates the percussive component. ``onset_strength_mean`` reads the same
force axis for free (rho = +0.55, near-identical between-model variance share)
if that ever matters more than the exact field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .arc import ARC_SR, _to_float, arc_tv4, timbral_coherence
from .features import analyze_onsets
from .features import duration_seconds as _duration_seconds
from .reward import EPS, _silence_stats, auxiliary_features, primary_n_fft

# `score_arrhythmic_percussive` and `compute_arc_coherence` both default to
# loading at 16 kHz mono with soxr_qq — the same load, done twice. Here it is
# done once and shared, which is also what keeps every field below numerically
# identical to the scorer it came from.
LIVE_SR = ARC_SR  # 16 kHz
LIVE_HOP_LENGTH = 512


@dataclass(frozen=True)
class LiveRoundComponents:
    """The six readings the TUI draws, plus what candidate selection needs.

    ``arc_tv4`` rides along because ``arc_coherence_score`` is computed from it
    and logging it is free. ``silence_fraction`` is both a panel row and the
    first gate in ``_select_best_candidate``; ``arc_coherence_score`` is the
    ranking base.
    """

    arc_coherence_score: float = 0.0
    arc_tv4: float = 0.0
    timbral_coherence: float = 0.0
    onset_density: float = 0.0
    percussive_energy_ratio: float = 0.0
    silence_fraction: float = 0.0
    max_silence_gap_seconds: float = 0.0
    duration_seconds: float = 0.0
    arc_tail_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


def score_live_round(
    audio_path: str | Path,
    *,
    prompt_duration_seconds: float | str | None = None,
    res_type: str = "soxr_qq",
) -> LiveRoundComponents:
    """Score one stream round. Never raises on degenerate audio.

    The rate/force/silence half is computed on the same geometry as
    ``score_arrhythmic_percussive`` (via ``auxiliary_features``), and the arc
    half on the same 16 kHz tail as ``compute_arc_coherence``, so every field
    here is numerically the same measurement it was before — there are simply
    fewer of them. ``prompt_duration_seconds`` trims the re-prompt echo off the
    arc half only, exactly as before; the seed round passes ``None``.
    """
    waveform, sr = librosa.load(audio_path, sr=LIVE_SR, mono=True, res_type=res_type)
    sr = int(sr)
    duration = _duration_seconds(waveform, sr)
    if duration <= 0.0:
        return LiveRoundComponents()

    onsets = analyze_onsets(
        waveform,
        sr=sr,
        hop_length=LIVE_HOP_LENGTH,
        n_fft=primary_n_fft(waveform, hop_length=LIVE_HOP_LENGTH),
    )

    aux = auxiliary_features(waveform, sr=sr, hop_length=LIVE_HOP_LENGTH)
    _harmonic, percussive = librosa.effects.hpss(aux.waveform)
    del _harmonic
    total_energy = float(np.sum(np.square(aux.waveform)))
    percussive_energy_ratio = float(np.sum(np.square(percussive))) / max(total_energy, EPS)

    silence_fraction, max_silence_gap_seconds = _silence_stats(
        aux.waveform,
        sr=aux.sr,
        frame_length=aux.n_fft,
        hop_length=aux.hop_length,
    )

    arc = _arc_half(waveform, sr=sr, prompt_duration_seconds=prompt_duration_seconds)

    return LiveRoundComponents(
        arc_coherence_score=arc[0],
        arc_tv4=arc[1],
        timbral_coherence=arc[2],
        onset_density=float(onsets.density),
        percussive_energy_ratio=float(percussive_energy_ratio),
        silence_fraction=float(silence_fraction),
        max_silence_gap_seconds=float(max_silence_gap_seconds),
        duration_seconds=float(duration),
        arc_tail_seconds=arc[3],
    )


def _arc_half(
    waveform: np.ndarray,
    *,
    sr: int,
    prompt_duration_seconds: float | str | None,
) -> tuple[float, float, float, float]:
    """``(arc_coherence_score, arc_tv4, timbral_coherence, tail_seconds)``.

    Takes the already-loaded 16 kHz waveform. The prompt echo is trimmed here
    and only here: the rate/force/silence half deliberately measures the whole
    round, exactly as ``score_arrhythmic_percussive`` always did.
    """
    echo = _to_float(prompt_duration_seconds)
    if echo and echo > 0:
        start = int(echo * sr)
        if start < len(waveform):
            waveform = waveform[start:]

    tail_seconds = float(len(waveform) / sr) if sr else 0.0
    if waveform.size < sr // 4:
        return 0.0, 0.0, 0.0, tail_seconds

    mid = len(waveform) // 2
    coherence = timbral_coherence(waveform[:mid], waveform[mid:], sr=sr)
    tv4 = arc_tv4(waveform, sr=sr)
    return 10.0 * tv4 * max(0.0, coherence), tv4, coherence, tail_seconds
