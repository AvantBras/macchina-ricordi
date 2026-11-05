"""Weights stored at half precision, computed at full -- where nobody can hear it.

A halved graph keeps each float weight as float16 behind a Cast back to float32.
ONNX Runtime folds those casts once, when a session opens, so what runs is a
float32 graph as before, at the same speed.

- The **language model** loses nothing. ``appliance.export`` already rounds its
  weights to float16, so the float32 graph was carrying 15 MB of zeros; halved,
  its logits are identical to the last bit.
- The **encoder** only turns the steered past back into codes for the model to
  read. Halved, it picks the same code 99.96% of the time on disco's own music.
- The **decoder** is left alone, because it is what everyone hears. Halved, its
  output on five minutes of the three models' music differed by about 70 dB
  under the music, but in one second of midwife only 24 dB under (-43 dBFS), and
  nobody has listened to whether that matters.

    PYTHONPATH=. .venv-appliance/bin/python -m appliance.halve boxes/*.nrbox

Needs onnx and onnxruntime. Each box is rewritten in place, and only after the
halved graphs have been run beside the originals and agree with them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .box import Box, read_box, write_box
from .spec import ModelSpec

HALF_SUFFIX = "__half"
# Biases and norms are a rounding error in size and not worth any risk.
MIN_ELEMENTS = 1024


def _big_floats(graph):
    from onnx import TensorProto

    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims, dtype=np.int64) >= MIN_ELEMENTS:
            yield "initializer", init
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute and node.attribute[0].name == "value":
            t = node.attribute[0].t
            if t.data_type == TensorProto.FLOAT and np.prod(t.dims, dtype=np.int64) >= MIN_ELEMENTS:
                yield "constant", node


def full_precision_bytes(data: bytes) -> int:
    """Bytes of float32 weights still stored at full precision; 0 once halved."""
    import onnx

    graph = onnx.load_from_string(data).graph
    total = 0
    for kind, item in _big_floats(graph):
        dims = item.dims if kind == "initializer" else item.attribute[0].t.dims
        total += 4 * int(np.prod(dims, dtype=np.int64))
    return total


def halve_graph(data: bytes) -> bytes:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load_from_string(data)
    graph = model.graph
    # By name: protobuf hands out fresh wrapper objects, so identity is no key.
    halved = {item.name if kind == "initializer" else item.output[0] for kind, item in _big_floats(graph)}
    casts = []
    for init in graph.initializer:
        if init.name in halved:
            name = init.name
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), name + HALF_SUFFIX))
            casts.append(helper.make_node("Cast", [name + HALF_SUFFIX], [name], to=TensorProto.FLOAT))
    nodes = []
    for node in graph.node:
        nodes.append(node)
        if node.op_type == "Constant" and node.output[0] in halved:
            out = node.output[0]
            value = node.attribute[0].t
            value.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(value).astype(np.float16)))
            node.output[0] = out + HALF_SUFFIX
            nodes.append(helper.make_node("Cast", [out + HALF_SUFFIX], [out], to=TensorProto.FLOAT))
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def halve(box: Box) -> Box:
    """The language model and the encoder halved; the decoder as it was."""
    return Box(
        spec=box.spec,
        lm=halve_graph(box.lm),
        decoder=box.decoder,
        encoder=None if box.encoder is None else halve_graph(box.encoder),
    )


def is_halved(box: Box) -> bool:
    return not any(full_precision_bytes(g) for g in (box.lm, box.encoder) if g is not None)


def music_like(spec: ModelSpec, frames: int, seed: int) -> np.ndarray:
    """A few drifting partials and some noise, ``[1, samples]``: codes that exercise the quantiser."""
    hop = int(round(spec.sample_rate / spec.frame_rate))
    rng = np.random.default_rng(seed)
    t = np.arange(hop * frames) / spec.sample_rate
    tone = sum(0.12 * np.sin(2 * np.pi * f * t * (1 + 0.01 * np.sin(t))) for f in rng.uniform(60, 3000, 4))
    return (tone + 0.03 * rng.standard_normal(t.size)).astype(np.float32)[None]


def check_halved(full: Box, half: Box, *, steps: int = 200, frames: int = 250) -> dict:
    """Run both beside each other; refuse if the halved graphs choose differently."""
    import onnxruntime as ort

    def session(data):
        return ort.InferenceSession(data, providers=["CPUExecutionProvider"])

    spec = full.spec
    rng = np.random.default_rng(0)
    empty = np.zeros((spec.num_layers, 1, spec.num_heads, 0, spec.head_dim), dtype=np.float32)
    feeds = {
        "tokens": rng.integers(0, spec.card, (1, spec.n_q, steps)),
        "offset": np.array(0, np.int64),
        "past_k": empty,
        "past_v": empty,
    }
    a = session(full.lm).run(None, feeds)[0]
    b = session(half.lm).run(None, feeds)[0]
    report = {
        "lm_max_abs_diff": float(np.abs(a - b).max()),
        "lm_token_agreement": float((a.argmax(-1) == b.argmax(-1)).mean()),
    }

    if full.encoder is not None:
        audio = music_like(spec, frames, seed=1)
        codes = session(full.encoder).run(None, {"audio": audio})[0]
        other = session(half.encoder).run(None, {"audio": audio})[0]
        report["encoder_code_agreement"] = float((codes == other).mean())

    if report["lm_token_agreement"] < 0.999 or report.get("encoder_code_agreement", 1.0) < 0.99:
        raise RuntimeError(f"halved graphs drift from the full-precision ones: {report}")
    return report


def describe_check(report: dict) -> str:
    lm = (
        "language model identical"
        if report["lm_max_abs_diff"] == 0
        else f"language model within {report['lm_max_abs_diff']:.1e}, same token "
        f"{report['lm_token_agreement'] * 100:.2f}%"
    )
    parts = [lm]
    if "encoder_code_agreement" in report:
        parts.append(f"encoder same code {report['encoder_code_agreement'] * 100:.2f}%")
    return ", ".join(parts)


def halve_box(path: Path, out: Path | None = None) -> tuple[Path, dict]:
    path = Path(path)
    box = read_box(path)
    small = halve(box)
    report = check_halved(box, small)
    written = write_box(
        out or path, spec=small.spec, lm_onnx=small.lm, decoder_onnx=small.decoder, encoder_onnx=small.encoder
    )
    return written, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boxes", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.boxes:
        if is_halved(read_box(path)):
            print(f"{path.name}: already halved")
            continue
        before = path.stat().st_size
        written, report = halve_box(path)
        print(f"{path.name}: {before / 1e6:.1f} MB -> {written.stat().st_size / 1e6:.1f} MB; {describe_check(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
