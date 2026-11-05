from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile

from macchina_ricordi.augmentation import export_speed_variants, parse_speed_ratios

SR = 8_000


def _sine(freq: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("value", ["0", "-1", "fast", "nan", "inf"])
def test_parse_speed_ratios_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        parse_speed_ratios(value)


def test_export_speed_variants_writes_one_file_per_ratio(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    soundfile.write(source, np.stack([_sine(200.0), _sine(300.0)]).T, SR)

    variants = export_speed_variants(source, tmp_path / "out", [0.5, 1, 2])

    assert [Path(v.output_path).name for v in variants] == [
        "clip__speed0p5.flac",
        "clip__speed1.flac",
        "clip__speed2.flac",
    ]
    assert [v.duration_seconds for v in variants] == [2.0, 1.0, 0.5]
    for variant in variants:
        info = soundfile.info(variant.output_path)
        assert (info.samplerate, info.channels) == (SR, 2)

    skipped = export_speed_variants(source, tmp_path / "skip", [1, 2], skip_identity=True)
    assert [v.ratio for v in skipped] == [2.0]
