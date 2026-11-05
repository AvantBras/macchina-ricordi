from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SPEED_MODES = ("varispeed", "stretch")
DEFAULT_SPEED_MODE = "varispeed"
OUTPUT_FORMATS = ("flac", "wav")
DEFAULT_OUTPUT_FORMAT = "flac"


@dataclass(frozen=True)
class SpeedVariant:
    source_path: str
    output_path: str
    ratio: float
    mode: str
    sample_rate: int
    channels: int
    duration_seconds: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def parse_speed_ratios(value: str | Sequence[float] | None) -> tuple[float, ...]:
    """Parse ``"0.8,1,1.25"`` (or a float sequence) into unique positive ratios, order kept."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw_parts: list[object] = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_parts = list(value)

    ratios: list[float] = []
    seen: set[float] = set()
    for part in raw_parts:
        try:
            ratio = float(part)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid speed ratio: {part!r}") from exc
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(f"Speed ratios must be finite and > 0, got {part!r}")
        key = round(ratio, 6)
        if key in seen:
            continue
        seen.add(key)
        ratios.append(ratio)
    return tuple(ratios)


def speed_variant_stem(stem: str, ratio: float) -> str:
    # "p" instead of "." so tools that split on the first dot keep the whole stem.
    return f"{stem}__speed{ratio:g}".replace(".", "p")


def change_speed(
    waveform: np.ndarray,
    sample_rate: int,
    ratio: float,
    *,
    mode: str = DEFAULT_SPEED_MODE,
    res_type: str = "soxr_hq",
) -> np.ndarray:
    """Play ``waveform`` ``ratio`` times faster; the time axis is the last one.

    ``varispeed`` is tape/turntable style: pitch moves with speed
    (12*log2(ratio) semitones) and no phase-vocoder artefacts are introduced.
    ``stretch`` keeps pitch but smears transients, which a model will learn too.
    """
    if mode not in SPEED_MODES:
        raise ValueError(f"Unknown speed mode {mode!r}; expected one of {SPEED_MODES}")
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"Speed ratio must be finite and > 0, got {ratio!r}")
    if ratio == 1.0:
        return waveform

    import librosa

    if mode == "varispeed":
        return librosa.resample(waveform, orig_sr=sample_rate * ratio, target_sr=sample_rate, res_type=res_type)
    return librosa.effects.time_stretch(waveform, rate=ratio)


def write_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    import soundfile

    frames_first = waveform.T if waveform.ndim == 2 else waveform
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), np.clip(frames_first, -1.0, 1.0), sample_rate, subtype="PCM_16")


def export_speed_variants(
    audio_path: str | Path,
    output_dir: str | Path,
    ratios: Sequence[float],
    *,
    mode: str = DEFAULT_SPEED_MODE,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    output_sr: int | None = None,
    mono: bool = False,
    skip_identity: bool = False,
    res_type: str = "soxr_hq",
) -> list[SpeedVariant]:
    """Write one ``<stem>__speed<ratio>.<format>`` file per ratio into ``output_dir``.

    Ratio 1 is written as a re-encoded copy so the output folder is a complete
    set, unless ``skip_identity`` is set (e.g. when the original sits next to it).
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown output format {output_format!r}; expected one of {OUTPUT_FORMATS}")
    parsed = parse_speed_ratios(ratios)
    if skip_identity:
        parsed = tuple(ratio for ratio in parsed if ratio != 1.0)
    if not parsed:
        return []

    import librosa

    source_path = Path(audio_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    waveform, sample_rate = librosa.load(source_path, sr=output_sr, mono=mono, res_type=res_type)
    sample_rate = int(sample_rate)

    variants: list[SpeedVariant] = []
    for ratio in parsed:
        changed = change_speed(waveform, sample_rate, ratio, mode=mode, res_type=res_type)
        output_path = destination / f"{speed_variant_stem(source_path.stem, ratio)}.{output_format}"
        write_audio(output_path, changed, sample_rate)
        variants.append(
            SpeedVariant(
                source_path=str(source_path),
                output_path=str(output_path),
                ratio=ratio,
                mode=mode,
                sample_rate=sample_rate,
                channels=1 if changed.ndim == 1 else int(changed.shape[0]),
                duration_seconds=round(float(changed.shape[-1]) / sample_rate, 3),
            )
        )
    return variants
