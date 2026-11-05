"""The ``.nrbox`` file: a zip a player can open without torch.

Three or four members, stored uncompressed because float weights barely compress and a
stored member loads without inflating:

- ``spec.json``     — the architecture and the probed tuning, plain JSON
- ``lm.onnx``       — the language model, one step at a time, KV cache as tensors
- ``decoder.onnx``  — the audio decoder, codes in and samples out
- ``encoder.onnx``  — optional: samples in, codes out. Only the prompt-steering
  controls need it (tilt, tape speed, bringing back a pinned moment), and they
  are the only reason a box would carry another 61 MB.

Of a 194 MB box, 118 MB is the decoder and 61 MB the encoder -- Meta's EnCodec,
the same bytes in every box -- and 15 MB is the model itself. The language model
and encoder are stored at half precision (see ``appliance.halve``).

The earlier torch pickle format still exists as the *source* a box is packed
from (``.nrsrc``, see ``appliance.format``); it is not what gets handed to people.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .spec import ModelSpec

BOX_FORMAT = "macchina_ricordi.box/2"
BOX_SUFFIX = ".nrbox"


@dataclass(frozen=True)
class Box:
    spec: ModelSpec
    lm: bytes
    decoder: bytes
    encoder: bytes | None = None


def write_box(
    path: Path, *, spec: ModelSpec, lm_onnx: bytes, decoder_onnx: bytes, encoder_onnx: bytes | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("spec.json", json.dumps({"format": BOX_FORMAT, "spec": spec.to_dict()}, indent=2))
        z.writestr("lm.onnx", lm_onnx)
        z.writestr("decoder.onnx", decoder_onnx)
        if encoder_onnx is not None:
            z.writestr("encoder.onnx", encoder_onnx)
    # A half-written box that looks complete would be found by the panel's scan.
    tmp.replace(path)
    return path


def _read_header(z: zipfile.ZipFile, path: Path) -> ModelSpec:
    try:
        header = json.loads(z.read("spec.json"))
    except KeyError as exc:
        raise ValueError(f"{path} has no spec.json; not a {BOX_FORMAT} file") from exc
    if header.get("format") != BOX_FORMAT:
        raise ValueError(f"{path} is not a {BOX_FORMAT} file (found {header.get('format')!r})")
    return ModelSpec.from_dict(header["spec"])


def read_spec(path: Path) -> ModelSpec:
    """Just the description — cheap enough to call on every box in a folder."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a {BOX_FORMAT} file")
    with zipfile.ZipFile(path) as z:
        return _read_header(z, path)


def read_box(path: Path) -> Box:
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a {BOX_FORMAT} file")
    with zipfile.ZipFile(path) as z:
        spec = _read_header(z, path)
        encoder = z.read("encoder.onnx") if "encoder.onnx" in z.namelist() else None
        return Box(spec=spec, lm=z.read("lm.onnx"), decoder=z.read("decoder.onnx"), encoder=encoder)


def replace_spec(path: Path, spec: ModelSpec) -> Path:
    box = read_box(path)
    return write_box(path, spec=spec, lm_onnx=box.lm, decoder_onnx=box.decoder, encoder_onnx=box.encoder)
