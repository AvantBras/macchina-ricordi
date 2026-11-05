"""A box with the real architecture and random language-model weights.

It lets a build be tested on a machine that has never seen a real model — a CI
runner, a borrowed Mac — without publishing anyone's trained weights. The decoder
is Meta's pretrained EnCodec, the same graph every real box carries; the language
model has the nano shape (192-wide, 10 layers) with random weights, so it costs
what a real one costs and plays noise.

    PYTHONPATH=. python packaging/appliance/make_test_box.py --out test.nrbox

Needs torch, onnx, onnxruntime and (for the real decoder) transformers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from appliance.box import Box, write_box
from appliance.codec_hf import HFDecoder, HFEncoder
from appliance.halve import check_halved, halve
from appliance.lm import NanoLM
from appliance.pack import export_decoder, export_encoder, export_lm, verify, verify_encoder
from appliance.spec import ModelSpec


class StubEncoder(nn.Module):
    """Each frame's loudness as a code, the same in every codebook. Shape, not sound."""

    def __init__(self, hop: int) -> None:
        super().__init__()
        self.hop = hop

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        level = audio.reshape(audio.shape[0], -1, self.hop).abs().mean(dim=-1)
        code = torch.clamp((level * 20000.0).long(), 0, 2047)
        return torch.stack([code] * 4, dim=1)


class StubDecoder(nn.Module):
    """Holds each frame's mean code for one frame's worth of samples. No sound, all shape."""

    def __init__(self, hop: int) -> None:
        super().__init__()
        self.hop = hop

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        level = 0.05 + 0.2 * codes.float().mean(dim=1) / 2048.0
        return level.unsqueeze(-1).expand(-1, -1, self.hop).flatten(1)


def make(out: Path, *, decoder: str, dim: int, layers: int, heads: int, seed: int, with_encoder: bool = True) -> Path:
    spec = ModelSpec(
        dim=dim,
        num_heads=heads,
        num_layers=layers,
        n_q=4,
        card=2048,
        hidden_scale=4.0,
        layer_scale=1e-4,
        delays=[0, 1, 2, 3],
        frame_rate=50.0,
        sample_rate=32000,
        channels=1,
        segment_duration=20.0,
        name=f"test-random-{dim}x{layers}",
        origin="packaging/appliance/make_test_box.py",
        note="random language-model weights: for testing builds, not for listening",
        tuning={"temperature": 1.0},
    )
    torch.manual_seed(seed)
    lm = NanoLM(spec).eval()
    # Real sources are stored in float16 (appliance.export); so are these weights,
    # which is what lets halving the graph lose nothing.
    with torch.no_grad():
        for weight in lm.parameters():
            weight.copy_(weight.half().float())
    codec = (HFDecoder() if decoder == "encodec" else StubDecoder(640)).eval()
    lm_onnx = export_lm(lm, spec)
    decoder_onnx = export_decoder(codec, spec)
    report = verify(lm, codec, spec, lm_onnx, decoder_onnx, prime=64, steps=8, decode_frames=77)
    encoder_onnx = None
    if with_encoder:
        coder = (HFEncoder() if decoder == "encodec" else StubEncoder(640)).eval()
        encoder_onnx = export_encoder(coder, spec)
        report |= verify_encoder(coder, spec, encoder_onnx)
    full = Box(spec=spec, lm=lm_onnx, decoder=decoder_onnx, encoder=encoder_onnx)
    small = halve(full)
    report["halved"] = check_halved(full, small)
    if report["halved"]["lm_max_abs_diff"] != 0:
        raise RuntimeError(f"halving changed the language model: {report}")
    lm_onnx, encoder_onnx = small.lm, small.encoder
    print(f"verified: {report}")
    path = write_box(out, spec=spec, lm_onnx=lm_onnx, decoder_onnx=decoder_onnx, encoder_onnx=encoder_onnx)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decoder", choices=("encodec", "stub"), default="encodec")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-encoder", action="store_true")
    args = ap.parse_args()
    make(
        args.out,
        decoder=args.decoder,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        seed=args.seed,
        with_encoder=not args.no_encoder,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
