"""WAV output with the standard library, so recording costs no dependency.

``soundfile`` is a fine library and 3 MB of native code the player does not need
for the one thing it would be used for: writing mono PCM.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, *, bits: int = 24) -> Path:
    if bits not in (16, 24):
        raise ValueError(f"bits must be 16 or 24, got {bits}")
    x = np.clip(np.asarray(audio, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if bits == 16:
        data = np.round(x * 32767).astype("<i2").tobytes()
    else:
        # 24-bit has no numpy dtype: write as int32 and keep the low three bytes.
        wide = np.round(x * 8_388_607).astype("<i4")
        data = wide.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(bits // 8)
        out.setframerate(int(sample_rate))
        out.writeframes(data)
    return path
