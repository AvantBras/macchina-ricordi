"""Two-model duet renderer: each voice continues a blend of itself + its partner.

Runs inside the AudioCraft venv with both models resident (fp16 + fastgen arm
automatically via ``_load_model``). Every round, each voice's continuation
prompt is ``(1 - listen) * own_tail + listen * partner_tail`` — both voices
read the *previous* round's state, so there is no ordering bias. Output is a
session directory with two mono stems, a panned stereo master, per-round
metadata, and a manifest.

Intended pairing: the 50 Hz deep (arc voice) with the 25 Hz LM (percussive
voice), but any two unconditional micro XPs work — cross-codec is free since
each model encodes the shared audio with its own EnCodec.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from run_musicgen_generate import _load_model, _to_numpy_audio_batch, _write_wav

from macchina_ricordi.instrument.duet import (
    crossfade_append,
    mix_tails,
    pick_least_silent,
    stereo_master,
)

CROSSFADE_SECONDS = 0.06


@dataclass
class Voice:
    name: str
    model: object
    source: dict
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    listen: float
    stem: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a two-model duet session")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--round-duration", type=float, default=9.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    for voice in ("a", "b"):
        parser.add_argument(f"--model-id-{voice}", default="facebook/musicgen-small")
        parser.add_argument(f"--profile-{voice}", default=None)
        parser.add_argument(f"--xp-path-{voice}", default=None)
        parser.add_argument(f"--checkpoint-path-{voice}", default=None)
        parser.add_argument(f"--temperature-{voice}", type=float, default=1.2)
        parser.add_argument(f"--top-k-{voice}", type=int, default=250)
        parser.add_argument(f"--top-p-{voice}", type=float, default=0.0)
        parser.add_argument(f"--min-p-{voice}", type=float, default=0.0)
        parser.add_argument(f"--listen-{voice}", type=float, default=0.35)
        parser.add_argument(f"--seed-audio-{voice}", default=None)
    parser.add_argument("--pan-a", type=float, default=-0.35)
    parser.add_argument("--pan-b", type=float, default=0.35)
    parser.add_argument("--gain-a", type=float, default=1.0)
    parser.add_argument("--gain-b", type=float, default=0.8)
    return parser.parse_args()


def _load_voice(args: argparse.Namespace, name: str, device: str) -> Voice:
    ns = SimpleNamespace(
        model_id=getattr(args, f"model_id_{name}"),
        profile=getattr(args, f"profile_{name}"),
        xp_path=getattr(args, f"xp_path_{name}"),
        checkpoint_path=getattr(args, f"checkpoint_path_{name}"),
    )
    model, source = _load_model(ns, device)
    return Voice(
        name=name,
        model=model,
        source=source,
        temperature=float(getattr(args, f"temperature_{name}")),
        top_k=int(getattr(args, f"top_k_{name}")),
        top_p=float(getattr(args, f"top_p_{name}")),
        min_p=float(getattr(args, f"min_p_{name}")),
        listen=float(getattr(args, f"listen_{name}")),
        stem=np.zeros(0, dtype=np.float32),
    )


def _configure(voice: Voice, *, duration: float) -> None:
    model = voice.model
    if duration > float(model.max_duration):
        raise ValueError(
            f"--round-duration {duration:g}s exceeds voice {voice.name}'s max duration ({model.max_duration:g}s)"
        )
    model.set_generation_params(
        use_sampling=True,
        top_k=voice.top_k,
        top_p=voice.top_p,
        temperature=voice.temperature,
        duration=duration,
        cfg_coef=3.0,
        extend_stride=min(8.0, float(model.max_duration) - 1.0),
    )
    model.lm._nr_min_p = voice.min_p


def _seed_round(voice: Voice, args: argparse.Namespace, *, sample_rate: int, n_samples: int) -> np.ndarray:
    """Round 0 contribution: a seed excerpt if given, else unconditional."""
    seed_audio = getattr(args, f"seed_audio_{voice.name}")
    target = int(args.round_duration * sample_rate)
    if seed_audio:
        import librosa

        excerpt, _ = librosa.load(seed_audio, sr=sample_rate, mono=True, duration=args.round_duration)
        if excerpt.size < target:
            excerpt = np.pad(excerpt, (0, target - excerpt.size))
        return excerpt.astype(np.float32)
    audio = _to_numpy_audio_batch(voice.model.generate_unconditional(n_samples))
    batch = audio[:, 0, :]
    winner, fractions = pick_least_silent(batch)
    print(f"round 0 voice {voice.name}: silence={[f'{f:.2f}' for f in fractions]} pick={winner}")
    return batch[winner]


@torch.no_grad()
def main() -> int:
    args = _parse_args()
    if args.tau >= args.round_duration:
        raise ValueError("--tau must be shorter than --round-duration")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    voices = [_load_voice(args, "a", device), _load_voice(args, "b", device)]
    rates = {int(v.model.sample_rate) for v in voices}
    if len(rates) != 1:
        raise ValueError(f"Voices must share a sample rate, got {sorted(rates)}")
    sample_rate = rates.pop()
    for voice in voices:
        _configure(voice, duration=args.round_duration)

    n_samples = max(1, min(8, int(args.candidates)))
    tau_samples = int(args.tau * sample_rate)
    crossfade_samples = int(CROSSFADE_SECONDS * sample_rate)
    rounds_log = output_dir / "rounds.jsonl"
    started = time.time()

    for round_idx in range(int(args.rounds)):
        # Snapshot both tails first: each voice listens to the *previous* state.
        tails = {v.name: v.stem[-tau_samples:] if v.stem.size else None for v in voices}
        for index, voice in enumerate(voices):
            if args.seed is not None:
                torch.manual_seed(int(args.seed) + 31 * round_idx + index)
                if device.startswith("cuda"):
                    torch.cuda.manual_seed_all(int(args.seed) + 31 * round_idx + index)

            t0 = time.monotonic()
            if round_idx == 0:
                contribution = _seed_round(voice, args, sample_rate=sample_rate, n_samples=n_samples)
                row = {"round": 0, "voice": voice.name, "mode": "seed"}
            else:
                partner = voices[1 - index]
                prompt = mix_tails(tails[voice.name], tails[partner.name], listen=voice.listen)
                prompt_batch = torch.from_numpy(prompt).to(device).reshape(1, 1, -1).repeat(n_samples, 1, 1)
                audio = _to_numpy_audio_batch(
                    voice.model.generate_continuation(prompt_batch, sample_rate, descriptions=[None] * n_samples)
                )
                batch = audio[:, 0, prompt.size :]
                winner, fractions = pick_least_silent(batch)
                contribution = batch[winner]
                row = {
                    "round": round_idx,
                    "voice": voice.name,
                    "mode": "continuation",
                    "listen": voice.listen,
                    "silence_fractions": [round(f, 3) for f in fractions],
                    "picked": winner,
                }
            voice.stem = crossfade_append(voice.stem, contribution, crossfade_samples=crossfade_samples)
            row["elapsed_seconds"] = round(time.monotonic() - t0, 2)
            with rounds_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"round {round_idx} voice {voice.name}: gen={row['elapsed_seconds']:.1f}s "
                f"stem={voice.stem.size / sample_rate:.1f}s"
            )

    stem_paths = {}
    for voice in voices:
        stem_path = output_dir / f"voice_{voice.name}.wav"
        _write_wav(stem_path, voice.stem[None, :], sample_rate)
        stem_paths[voice.name] = stem_path
    master = stereo_master(
        voices[0].stem,
        voices[1].stem,
        pan_a=args.pan_a,
        pan_b=args.pan_b,
        gain_a=args.gain_a,
        gain_b=args.gain_b,
    )
    mix_path = output_dir / "mix.wav"
    _write_wav(mix_path, master, sample_rate)

    manifest = {
        "schema": "macchina_ricordi.duet_session.v0",
        "output_dir": str(output_dir),
        "sample_rate": sample_rate,
        "rounds": int(args.rounds),
        "round_duration_seconds": args.round_duration,
        "tau_seconds": args.tau,
        "candidates": n_samples,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - started, 1),
        "mix": mix_path.name,
        "voices": {
            v.name: {
                "stem": stem_paths[v.name].name,
                "source": v.source,
                "temperature": v.temperature,
                "top_k": v.top_k,
                "top_p": v.top_p,
                "min_p": v.min_p,
                "listen": v.listen,
                "pan": getattr(args, f"pan_{v.name}"),
                "gain": getattr(args, f"gain_{v.name}"),
            }
            for v in voices
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"duet_dir: {output_dir}")
    print(f"mix: {mix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
