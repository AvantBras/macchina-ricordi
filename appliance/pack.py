"""Pack a ``.nrsrc`` source into a ``.nrbox`` that plays without torch.

    PYTHONPATH=. .venv-appliance/bin/python -m appliance.pack boxes/src/*.nrsrc

Needs torch, onnx and onnxruntime; the box it writes needs only onnxruntime and
numpy. Nothing is written until the exported graphs have been run against the
torch reference and agree with it: a box that plays something other than the
model is worse than no box, and a conversion step is exactly where that would go
unnoticed. Then the box is probed for the silent attractor using the runtime it
will actually ship on.
"""

from __future__ import annotations

import argparse
import io
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .box import BOX_SUFFIX, Box, replace_spec, write_box
from .format import read_source
from .halve import check_halved, describe_check, halve, music_like
from .lm import NanoLM
from .spec import ModelSpec

OPSET = 17


class StepGraph(nn.Module):
    """One exportable step: tokens, offset and past KV in; logits and present KV out.

    ``NanoLM`` writes its cache in place, which an ONNX graph cannot express. Here
    the cache is ordinary tensors concatenated per step, and the causal mask is
    built from the cache length, so one graph serves both the many-frame prime and
    the single steps after it.
    """

    def __init__(self, lm: NanoLM) -> None:
        super().__init__()
        self.lm = lm
        self.spec = lm.spec

    def forward(self, tokens, offset, past_k, past_v):
        spec, lm = self.spec, self.lm
        B, K, T = tokens.shape[0], tokens.shape[1], tokens.shape[2]
        P = past_k.shape[3]
        H, D = spec.num_heads, spec.dim
        hd = D // H
        x = lm.emb[0](tokens[:, 0])
        for k in range(1, K):
            x = x + lm.emb[k](tokens[:, k])
        positions = (torch.arange(T, dtype=torch.float32) + offset.to(torch.float32)).view(1, -1, 1)
        half = D // 2
        adim = torch.arange(half, dtype=torch.float32).view(1, 1, -1)
        phase = positions / (spec.max_period ** (adim / (half - 1)))
        x = x + torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        query_pos = torch.arange(T).view(-1, 1) + P
        key_pos = torch.arange(P + T).view(1, -1)
        mask = torch.where(key_pos <= query_pos, 0.0, float("-inf")).to(x.dtype)
        keys, values = [], []
        for i, layer in enumerate(lm.layers):
            q, k, v = F.linear(layer.norm1(x), layer.in_proj_weight).chunk(3, dim=-1)
            q = q.view(B, T, H, hd).transpose(1, 2)
            k = torch.cat([past_k[i], k.view(B, T, H, hd).transpose(1, 2)], dim=2)
            v = torch.cat([past_v[i], v.view(B, T, H, hd).transpose(1, 2)], dim=2)
            keys.append(k)
            values.append(v)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn = layer.out_proj(attn.transpose(1, 2).reshape(B, T, D))
            x = x + (attn * layer.layer_scale_1 if layer.has_scale else attn)
            ff = layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))
            x = x + (ff * layer.layer_scale_2 if layer.has_scale else ff)
        x = lm.out_norm(x)
        logits = torch.stack([linear(x) for linear in lm.linears], dim=1)
        return logits, torch.stack(keys), torch.stack(values)


def export_lm(lm: NanoLM, spec: ModelSpec) -> bytes:
    graph = StepGraph(lm.float().eval()).eval()
    tokens = torch.randint(0, spec.card, (1, spec.n_q, 2))
    past = torch.zeros(spec.num_layers, 1, spec.num_heads, 3, spec.head_dim)
    buffer = io.BytesIO()
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            graph,
            (tokens, torch.tensor(3), past, past),
            buffer,
            dynamo=False,
            opset_version=OPSET,
            input_names=["tokens", "offset", "past_k", "past_v"],
            output_names=["logits", "present_k", "present_v"],
            dynamic_axes={
                "tokens": {0: "batch", 2: "steps"},
                "past_k": {1: "batch", 3: "past"},
                "past_v": {1: "batch", 3: "past"},
                "logits": {0: "batch", 2: "steps"},
                "present_k": {1: "batch", 3: "total"},
                "present_v": {1: "batch", 3: "total"},
            },
        )
    return buffer.getvalue()


def export_decoder(codec: torch.jit.ScriptModule | nn.Module, spec: ModelSpec) -> bytes:
    example = torch.randint(0, spec.card, (1, spec.n_q, 37))
    buffer = io.BytesIO()
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            codec,
            (example,),
            buffer,
            dynamo=False,
            opset_version=OPSET,
            input_names=["codes"],
            output_names=["audio"],
            dynamic_axes={"codes": {2: "frames"}, "audio": {1: "samples"}},
        )
    return buffer.getvalue()


def export_encoder(encoder: nn.Module, spec: ModelSpec) -> bytes:
    hop = int(round(spec.sample_rate / spec.frame_rate))
    example = torch.randn(1, hop * 37) * 0.1
    buffer = io.BytesIO()
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            encoder.eval(),
            (example,),
            buffer,
            dynamo=False,
            opset_version=OPSET,
            input_names=["audio"],
            output_names=["codes"],
            dynamic_axes={"audio": {1: "samples"}, "codes": {2: "frames"}},
        )
    return buffer.getvalue()


def verify_encoder(encoder: nn.Module, spec: ModelSpec, encoder_onnx: bytes, *, lengths=(61, 173)) -> dict:
    """The exported encoder must pick the same codes as the torch one, at lengths it was not traced at."""
    import onnxruntime as ort

    session = ort.InferenceSession(encoder_onnx, providers=["CPUExecutionProvider"])
    agreement = []
    for i, frames in enumerate(lengths):
        audio = music_like(spec, frames, seed=i)
        with torch.no_grad():
            reference = encoder.eval()(torch.from_numpy(audio)).numpy()
        exported = session.run(None, {"audio": audio})[0]
        if exported.shape != reference.shape:
            raise RuntimeError(f"exported encoder gives shape {exported.shape}, torch gives {reference.shape}")
        agreement.append(float((exported == reference).mean()))
    report = {"encoder_code_agreement": min(agreement)}
    if report["encoder_code_agreement"] < 0.999:
        raise RuntimeError(f"exported encoder does not reproduce the torch encoder: {report}")
    return report


def check_codec_pairing(source_codec, reference_decoder: nn.Module, spec: ModelSpec) -> float:
    """An encoder is only meaningful beside the decoder it was trained with.

    Boxes gain their encoder from the pretrained codec rather than from the
    experiment, so the source's own decoder is compared with that codec's first:
    same codes in, the same audio out, or no encoder is added.
    """
    codes = torch.randint(0, spec.card, (1, spec.n_q, 64), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours = source_codec(codes)
        theirs = reference_decoder(codes)
    difference = float((ours - theirs).abs().max()) if ours.shape == theirs.shape else float("inf")
    if difference > 1e-3:
        raise RuntimeError(
            f"this model's decoder differs from {getattr(reference_decoder, 'name', 'the pretrained codec')} "
            f"by {difference:.3g}; an encoder from it would not speak the same codes"
        )
    return difference


def verify(
    lm: NanoLM,
    codec,
    spec: ModelSpec,
    lm_onnx: bytes,
    decoder_onnx: bytes,
    *,
    prime: int = 300,
    steps: int = 24,
    decode_frames: int = 123,
) -> dict:
    """Run the exported graphs beside the torch reference; refuse on disagreement.

    The prime and the decode length are deliberately different from the example
    shapes used at export, so a dimension that was baked in as a constant fails
    here instead of in somebody's living room.
    """
    import onnxruntime as ort

    lm = lm.float().eval()
    generator = torch.Generator().manual_seed(0)
    total = prime + steps
    seq = torch.randint(0, spec.card, (1, spec.n_q, total), generator=generator)
    with torch.no_grad():
        reference = lm(seq, 0, lm.fresh_cache(1, total, torch.float32)).numpy()

    session = ort.InferenceSession(lm_onnx, providers=["CPUExecutionProvider"])
    empty = np.zeros((spec.num_layers, 1, spec.num_heads, 0, spec.head_dim), dtype=np.float32)
    feeds = {"tokens": seq[:, :, :prime].numpy(), "offset": np.array(0, np.int64), "past_k": empty, "past_v": empty}
    logits, past_k, past_v = session.run(None, feeds)
    pieces = [logits]
    for t in range(prime, total):
        logits, past_k, past_v = session.run(
            None,
            {
                "tokens": seq[:, :, t : t + 1].numpy(),
                "offset": np.array(t, np.int64),
                "past_k": past_k,
                "past_v": past_v,
            },
        )
        pieces.append(logits)
    exported = np.concatenate(pieces, axis=2)

    codes = torch.randint(0, spec.card, (1, spec.n_q, decode_frames), generator=generator)
    with torch.no_grad():
        audio_reference = codec(codes).numpy()
    decoder = ort.InferenceSession(decoder_onnx, providers=["CPUExecutionProvider"])
    audio = decoder.run(None, {"codes": codes.numpy()})[0]

    same_shape = audio.shape == audio_reference.shape
    report = {
        "lm_max_abs_diff": float(np.abs(exported - reference).max()),
        "lm_token_agreement": float((exported.argmax(-1) == reference.argmax(-1)).mean()),
        "decoder_max_abs_diff": float(np.abs(audio - audio_reference).max()) if same_shape else float("inf"),
        "decoder_shape_ok": same_shape,
    }
    if report["lm_token_agreement"] < 0.999 or not same_shape or report["decoder_max_abs_diff"] > 1e-3:
        raise RuntimeError(f"exported graphs do not reproduce the source model: {report}")
    return report


def pack(
    source: Path,
    out: Path,
    *,
    probe: bool = True,
    keep_tuning: bool = False,
    encoder: bool = True,
    half: bool = True,
    takes: int = 12,
    seconds: float = 45.0,
    threads: int | None = None,
) -> Path:
    spec, weights, codec = read_source(source)
    lm = NanoLM(spec)
    lm.load_audiocraft_state({k: v.float() for k, v in weights.items()})
    lm.eval()

    lm_onnx = export_lm(lm, spec)
    decoder_onnx = export_decoder(codec, spec)
    report = verify(lm, codec, spec, lm_onnx, decoder_onnx)
    print(
        f"{spec.name}: verified — logits within {report['lm_max_abs_diff']:.1e}, "
        f"same token {report['lm_token_agreement'] * 100:.2f}%, "
        f"decoder within {report['decoder_max_abs_diff']:.1e}"
    )

    encoder_onnx = None
    if encoder:
        from .codec_hf import HFDecoder, HFEncoder

        pairing = check_codec_pairing(codec, HFDecoder(), spec)
        hf_encoder = HFEncoder()
        encoder_onnx = export_encoder(hf_encoder, spec)
        agreement = verify_encoder(hf_encoder, spec, encoder_onnx)["encoder_code_agreement"]
        print(
            f"{spec.name}: encoder added — decoder matches the pretrained codec within {pairing:.1e}, "
            f"exported encoder picks the same code {agreement * 100:.2f}% of the time"
        )

    if half:
        full = Box(spec=spec, lm=lm_onnx, decoder=decoder_onnx, encoder=encoder_onnx)
        small = halve(full)
        print(f"{spec.name}: halved — {describe_check(check_halved(full, small))}")
        lm_onnx, encoder_onnx = small.lm, small.encoder

    path = write_box(out, spec=spec, lm_onnx=lm_onnx, decoder_onnx=decoder_onnx, encoder_onnx=encoder_onnx)
    if probe and not (keep_tuning and spec.tuning.get("temperature")):
        from .probe import probe_tuning

        tuning = probe_tuning(path, takes=takes, seconds=seconds, threads=threads)
        spec = replace(spec, tuning=tuning)
        replace_spec(path, spec)
    temperature = spec.tuning.get("temperature") if spec.tuning else None
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, temperature {temperature})")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path, help=".nrsrc files written by appliance.export")
    ap.add_argument("--out", type=Path, default=None, help="output file (only with a single source)")
    ap.add_argument("--out-dir", type=Path, default=Path("boxes"))
    ap.add_argument("--no-probe", action="store_true", help="skip measuring the silent attractor")
    ap.add_argument("--keep-tuning", action="store_true", help="reuse a temperature already in the source")
    ap.add_argument(
        "--no-encoder",
        action="store_true",
        help="leave out the encoder: 61 MB smaller, but no tilt, tape speed or bringing back pins",
    )
    ap.add_argument(
        "--full-precision",
        action="store_true",
        help="store the language model and encoder in float32 too (see appliance.halve)",
    )
    ap.add_argument("--takes", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.out and len(args.sources) > 1:
        ap.error("--out takes a single source; use --out-dir for several")
    for source in args.sources:
        if args.out:
            out = args.out
        else:
            name = read_source(source)[0].name
            out = args.out_dir / f"{name}{BOX_SUFFIX}"
        pack(
            source,
            out,
            probe=not args.no_probe,
            keep_tuning=args.keep_tuning,
            encoder=not args.no_encoder,
            half=not args.full_precision,
            takes=args.takes,
            seconds=args.seconds,
            threads=args.threads,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
