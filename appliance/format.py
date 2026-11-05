"""The ``.nrsrc`` source file: a model lifted out of its experiment folder.

``appliance.export`` writes one of these with the AudioCraft interpreter, and
``appliance.pack`` turns it into the ``.nrbox`` that actually ships. It is a torch
pickle — the language model's tensors, the decoder as a frozen TorchScript graph,
and the spec — so reading it needs torch, which is exactly why it is not the file
anyone plays.
"""

from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path

import torch

from .spec import ModelSpec

SOURCE_FORMAT = "macchina_ricordi.box/1"
SOURCE_SUFFIX = ".nrsrc"


def write_source(
    path: Path,
    *,
    spec: ModelSpec,
    lm_state: dict[str, torch.Tensor],
    codec_module: torch.jit.ScriptModule,
    half: bool = True,
) -> Path:
    dtype = torch.float16 if half else torch.float32
    buffer = io.BytesIO()
    torch.jit.save(codec_module, buffer)
    payload = {
        "format": SOURCE_FORMAT,
        "spec": asdict(spec),
        "lm": {k: v.detach().to(dtype).contiguous() for k, v in lm_state.items()},
        "lm_dtype": str(dtype).removeprefix("torch."),
        "codec": buffer.getvalue(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Protocol 5 matters: at torch's default of 2, `bytes` has no native opcode
    # and is pickled through a latin-1 round trip that inflates the traced codec
    # by half again -- 118 MB of graph became 177 MB on disk.
    torch.save(payload, path, pickle_protocol=5)
    return path


def read_source(path: Path) -> tuple[ModelSpec, dict[str, torch.Tensor], torch.jit.ScriptModule]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != SOURCE_FORMAT:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"{path} is not a {SOURCE_FORMAT} source (found {found!r})")
    codec = torch.jit.load(io.BytesIO(payload["codec"]), map_location="cpu")
    return ModelSpec.from_dict(payload["spec"]), payload["lm"], codec
