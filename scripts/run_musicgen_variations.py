"""Variations: re-perform a piece by freezing coarse RVQ books and resampling the rest.

Encode the input WAV with the model's own EnCodec, pre-seed the generation
grid with the frozen book(s) (book 0 = the structural skeleton), and run the
standard delay-pattern loop — pre-filled tokens survive sampling and steer it.
All n variations render in one batch (batching is near-free; the AR loop is
launch-bound). Also writes the codec round-trip of the reference for a fair
listening comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from run_musicgen_generate import (
    _load_model,
    _read_prompt_audio,
    _to_numpy_audio_batch,
    _write_wav,
)

from macchina_ricordi.instrument.variations import UNKNOWN_TOKEN, seed_frozen_codes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render frozen-structure variations of a WAV")
    parser.add_argument("--input", required=True, help="Reference audio (any format librosa reads)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="facebook/musicgen-small")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--xp-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--n", type=int, default=4, help="Variations to render (one batch)")
    parser.add_argument(
        "--freeze-books",
        type=int,
        default=1,
        help="How many coarse RVQ books to freeze (default: 1 = book 0, the structure)",
    )
    parser.add_argument(
        "--keep-intro-seconds",
        type=float,
        default=0.0,
        help="Freeze *all* books for the first N seconds so variations open identically",
    )
    parser.add_argument("--seconds", type=float, default=None, help="Cap on reference length")
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def _generate_frozen(
    lm,
    gen_codes: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """The vendored LMModel.generate loop with a pre-seeded gen_codes grid.

    Unconditional single-pass path only (cfg_conditions={}), which is the
    micro-model regime; the ``torch.where`` write-guard is what lets the
    frozen tokens survive and condition the resampled books.
    """
    B, K, max_gen_len = gen_codes.shape
    pattern = lm.pattern_provider.get_pattern(max_gen_len)
    gen_sequence, _, mask = pattern.build_pattern_sequence(gen_codes, lm.special_token_id)
    start_offset_sequence = pattern.get_first_step_with_timesteps(0)
    assert start_offset_sequence is not None

    with lm.streaming():
        unconditional_state = lm.get_streaming_state()
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]
        for offset in range(start_offset_sequence, gen_sequence_len):
            curr_sequence = gen_sequence[..., prev_offset:offset]
            next_token = lm._sample_next_token(
                curr_sequence,
                {},
                unconditional_state,
                True,
                temperature,
                top_k,
                top_p,
                cfg_coef=None,
                two_step_cfg=False,
            )
            valid_mask = mask[..., offset : offset + 1].expand(B, -1, -1)
            next_token[~valid_mask] = lm.special_token_id
            gen_sequence[..., offset : offset + 1] = torch.where(
                gen_sequence[..., offset : offset + 1] == UNKNOWN_TOKEN,
                next_token,
                gen_sequence[..., offset : offset + 1],
            )
            prev_offset = offset
    unconditional_state.clear()

    out_codes, _, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=UNKNOWN_TOKEN)
    out_codes = out_codes[..., :max_gen_len]
    assert (out_codes != UNKNOWN_TOKEN).all()
    assert (out_codes >= 0).all() and (out_codes <= lm.card).all()
    return out_codes


def main() -> int:
    args = _parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)

    model, source = _load_model(
        SimpleNamespace(
            model_id=args.model_id,
            profile=args.profile,
            xp_path=args.xp_path,
            checkpoint_path=args.checkpoint_path,
        ),
        device,
    )
    model.lm._nr_min_p = float(args.min_p)
    frame_rate = float(model.compression_model.frame_rate)
    cap_seconds = float(model.max_duration)
    if args.seconds is not None:
        cap_seconds = min(cap_seconds, float(args.seconds))

    wav, sample_rate = _read_prompt_audio(
        SimpleNamespace(
            prompt_audio=args.input,
            prompt_start_seconds=0.0,
            prompt_duration_seconds=cap_seconds,
            duration_seconds=cap_seconds + 1.0,  # only used for a sanity check
        )
    )
    from audiocraft.data.audio_utils import convert_audio

    wav = convert_audio(wav, sample_rate, int(model.sample_rate), 1)
    wav = wav[None].to(device)

    ref_codes, scale = model.compression_model.encode(wav)
    assert scale is None
    total_len = min(ref_codes.shape[-1], int(cap_seconds * frame_rate))
    ref_codes = ref_codes[..., :total_len]

    n = max(1, int(args.n))
    gen_codes = seed_frozen_codes(
        ref_codes.expand(n, -1, -1),
        freeze_books=int(args.freeze_books),
        total_len=total_len,
        keep_intro_steps=int(args.keep_intro_seconds * frame_rate),
    )

    t0 = time.monotonic()
    with model.autocast:
        out_codes = _generate_frozen(
            model.lm,
            gen_codes,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
        )
    elapsed = time.monotonic() - t0

    frozen = int(args.freeze_books)
    if frozen > 0:
        preserved = bool((out_codes[:, :frozen, :] == ref_codes[:, :frozen, :]).all())
        print(f"frozen_books_preserved: {preserved}")
        assert preserved, "frozen books were altered by the loop"

    reference_audio = _to_numpy_audio_batch(model.compression_model.decode(ref_codes[:1]))
    _write_wav(output_dir / "reference_roundtrip.wav", reference_audio[0], int(model.sample_rate))
    variation_audio = _to_numpy_audio_batch(model.compression_model.decode(out_codes))
    files = []
    for index in range(variation_audio.shape[0]):
        path = output_dir / f"variation_{index:02d}.wav"
        _write_wav(path, variation_audio[index], int(model.sample_rate))
        files.append(path.name)

    manifest = {
        "schema": "macchina_ricordi.variations.v0",
        "input": str(Path(args.input).expanduser().resolve()),
        "source": source,
        "sample_rate": int(model.sample_rate),
        "frame_rate": frame_rate,
        "seconds": total_len / frame_rate,
        "n": n,
        "freeze_books": frozen,
        "keep_intro_seconds": args.keep_intro_seconds,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 2),
        "reference_roundtrip": "reference_roundtrip.wav",
        "files": files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"variations_dir: {output_dir}")
    print(f"elapsed: {elapsed:.1f}s for {n} x {total_len / frame_rate:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
