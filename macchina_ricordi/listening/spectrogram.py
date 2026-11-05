from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import librosa
import numpy as np

from .schema import ListeningDossier


def default_spectrogram_path(audio_path: str | Path) -> Path:
    path = Path(audio_path).expanduser()
    return path.with_name(f"{path.stem}_spectrogram.png")


def render_mel_spectrogram_png(
    waveform: np.ndarray,
    *,
    sr: int,
    output_path: str | Path,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    dpi: int = 100,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from librosa import display as librosa_display

    if waveform.ndim != 1:
        raise ValueError("waveform must be a 1D mono array")
    if sr <= 0:
        raise ValueError("sr must be positive")

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    effective_fmax = float(fmax) if fmax is not None else float(sr) / 2.0
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=effective_fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    figure, axes = plt.subplots(figsize=figsize, dpi=dpi)
    try:
        image = librosa_display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            fmin=fmin,
            fmax=effective_fmax,
            ax=axes,
        )
        axes.set_title("Mel spectrogram")
        figure.colorbar(image, ax=axes, format="%+2.0f dB")
        figure.tight_layout()
        figure.savefig(out, dpi=dpi)
    finally:
        plt.close(figure)
    return out


def attach_spectrogram_path(dossier: ListeningDossier, *, path: str | Path) -> ListeningDossier:
    new_sketches = replace(dossier.sketches, mel_spectrogram_path=str(path))
    return replace(dossier, sketches=new_sketches)


def render_mel_spectrogram_png_from_file(
    audio_path: str | Path,
    *,
    output_path: str | Path,
    sr: int = 32_000,
    res_type: str = "soxr_hq",
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    dpi: int = 100,
) -> Path:
    waveform, loaded_sr = librosa.load(audio_path, sr=sr, mono=True, res_type=res_type)
    return render_mel_spectrogram_png(
        waveform,
        sr=int(loaded_sr),
        output_path=output_path,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        figsize=figsize,
        dpi=dpi,
    )
