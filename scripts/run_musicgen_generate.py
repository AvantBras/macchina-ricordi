from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
from audiocraft.data.audio import audio_read
from audiocraft.models import MusicGen

from macchina_ricordi.instrument.provenance import resolve_generation_profile, resolve_generation_xp_path
from macchina_ricordi.instrument.saving import _write_wav

AUDIOCRAFT_CONSTRUCTOR_EXTEND_STRIDE = 18.0
DEFAULT_EXTEND_STRIDE_SECONDS = 8.0


def _to_numpy_audio_batch(audio_values: object) -> np.ndarray:
    array = audio_values
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()

    np_audio = np.asarray(array)
    if np_audio.ndim == 2:
        np_audio = np_audio[:, None, :]
    if np_audio.ndim != 3:
        raise ValueError(f"Unsupported audio tensor rank: ndim={np_audio.ndim}. Expected [B, T] or [B, C, T].")
    return np_audio.astype(np.float32, copy=False)


def _validate_written_wav(path: Path, *, channels: int, frames: int, sample_rate: int) -> None:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != int(channels):
            raise ValueError(f"WAV channel count mismatch for {path}.")
        if wav_file.getframerate() != int(sample_rate):
            raise ValueError(f"WAV sample rate mismatch for {path}.")
        if wav_file.getnframes() != int(frames):
            raise ValueError(f"WAV frame count mismatch for {path}.")
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"WAV sample width mismatch for {path}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MusicGen audio inside the AudioCraft runtime")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("unconditional", "continuation"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--xp-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--prompt-start-seconds", type=float, default=0.0)
    parser.add_argument("--prompt-duration-seconds", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--cfg-coef", type=float, default=3.0)
    parser.add_argument("--extend-stride", type=float, default=DEFAULT_EXTEND_STRIDE_SECONDS)
    parser.add_argument("--use-sampling", dest="use_sampling", action="store_true", default=True)
    parser.add_argument("--no-use-sampling", dest="use_sampling", action="store_false")
    return parser.parse_args()


def _nested_get_float(root: object, dotted_path: str) -> float | None:
    value = root
    for part in dotted_path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_max_duration(solver: object) -> float:
    for root in (getattr(solver, "cfg", None), getattr(getattr(solver, "model", None), "cfg", None)):
        duration = _nested_get_float(root, "dataset.segment_duration") if root is not None else None
        if duration is not None and duration > 0:
            return duration
    raise ValueError("Could not infer max_duration from the Dora XP config dataset.segment_duration.")


def _detect_bundle(args: argparse.Namespace) -> Path | None:
    """A resolved bundle dir from --xp-path or --checkpoint-path, else None.

    Checked before any Dora resolution: a bundle is self-contained and has no
    .hydra/config to feed ``get_solver_from_sig``, so it must be routed to the
    bundle runtime loader instead of the Dora/checkpoint path.
    """
    from macchina_ricordi.models.bundle_runtime import bundle_dir_or_none

    for attribute in ("xp_path", "checkpoint_path"):
        value = getattr(args, attribute, None)
        resolved = bundle_dir_or_none(value)
        if resolved is not None:
            return resolved
    return None


def _build_musicgen_from_solver(name: str, solver: object, max_duration: float) -> MusicGen:
    constructor_max_duration = max(max_duration, AUDIOCRAFT_CONSTRUCTOR_EXTEND_STRIDE + 1.0)
    model = MusicGen(name, solver.compression_model, solver.model, max_duration=constructor_max_duration)
    model.max_duration = max_duration
    return model


def _optimize_for_inference(model: MusicGen) -> None:
    """Inference-only LM speedups; no effect on output quality.

    Validated on the RTX 3060 with the reshape-deep mega model (2026-06-08):
    xRT 2.43 -> 3.56 for a 9 s round. Two independent effects:

    - profiles trained with ``transformer_lm.checkpointing=torch`` run the
      gradient-checkpoint wrapper on every layer even under ``no_grad``
      (~25% of step time);
    - fp16 weights skip the per-step autocast casts (logits were already
      computed in fp16 under autocast, so sampling numerics are unchanged;
      verified rms/silence/arc parity over seeds).

    Set ``MACCHINA_RICORDI_NO_INFER_OPT=1`` to disable both.
    """
    import os

    # min-p sampler patch (macchina_ricordi.musicgen.sampling): a sampler feature, not
    # a speed path, so it arms regardless of MACCHINA_RICORDI_NO_INFER_OPT. Inert
    # until a positive min_p is requested (delegates to the stock sampler).
    try:
        from macchina_ricordi.musicgen.sampling import enable_min_p

        enable_min_p(model)
    except Exception as exc:
        print(f"minp_disabled: {type(exc).__name__}: {exc}")

    if os.environ.get("MACCHINA_RICORDI_NO_INFER_OPT") == "1":
        return
    transformer = getattr(model.lm, "transformer", None)
    if transformer is not None and getattr(transformer, "checkpointing", "none") != "none":
        transformer.checkpointing = "none"
    try:
        on_cuda = next(model.lm.parameters()).device.type == "cuda"
    except (AttributeError, StopIteration):
        return
    if on_cuda:
        model.lm.half()
        if os.environ.get("MACCHINA_RICORDI_NO_FASTGEN") != "1":
            # Static-KV-cache + CUDA-graph step (macchina_ricordi.musicgen.fastgen):
            # ~3x over the eager loop with exact greedy parity. Arms itself
            # only for the unconditional micro architecture; anything else
            # keeps the original AudioCraft forward.
            try:
                from macchina_ricordi.musicgen.fastgen import enable_fast_generation

                armed = enable_fast_generation(model)
            except Exception as exc:  # never break generation for a speed path
                print(f"fastgen_disabled: {type(exc).__name__}: {exc}")
            else:
                if armed:
                    print("fastgen: static-KV CUDA-graph step armed")


def _set_seed(seed: object, device: str) -> None:
    if seed is None:
        return
    seed_value = int(seed)
    torch.manual_seed(seed_value)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed_value)


def _load_model(args: argparse.Namespace, device: str) -> tuple[MusicGen, dict[str, object]]:
    # A bundle is self-contained and precedes Dora/checkpoint resolution; it
    # has no .hydra config to feed get_solver_from_sig.
    bundle_path = _detect_bundle(args)
    if bundle_path is not None:
        from macchina_ricordi.models.bundle_runtime import load_bundle_model

        model, source = load_bundle_model(bundle_path, device=device)
        _optimize_for_inference(model)
        return model, source

    if args.checkpoint_path:
        from audiocraft import train
        from audiocraft.utils import checkpoint as checkpoint_utils

        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        xp_path = resolve_generation_xp_path(xp_path=args.xp_path, checkpoint_path=checkpoint_path)
        if xp_path is None:
            raise ValueError(
                "Could not infer a Dora XP config for --checkpoint-path. "
                "Pass --xp-path or keep a merged_from.json/perturbed_from.json sidecar next to the checkpoint."
            )
        profile = resolve_generation_profile(
            profile_name=args.profile,
            xp_path=xp_path,
            checkpoint_path=checkpoint_path,
        )
        solver = train.get_solver_from_sig(
            xp_path.name,
            override_cfg={
                "device": device,
                "autocast": device.startswith("cuda"),
                "execute_inplace": True,
                "logging": {"log_wandb": False, "log_tensorboard": False},
                "dataset": {"num_workers": 0},
            },
            restore=False,
            load_best=False,
            disable_fsdp=True,
        )
        state = checkpoint_utils.load_checkpoint(checkpoint_path)
        stateful_sources = getattr(getattr(solver, "stateful", None), "sources", {})
        if isinstance(stateful_sources, dict) and "scaler" in state and "scaler" not in stateful_sources:
            state = dict(state)
            state.pop("scaler", None)
        solver.load_state_dict(state)
        solver.model.eval()
        max_duration = _infer_max_duration(solver)
        model = _build_musicgen_from_solver(args.model_id, solver, max_duration)
        _optimize_for_inference(model)
        source_kind = "xp" if checkpoint_path.parent == xp_path else "checkpoint"
        return model, {
            "kind": source_kind,
            "model_id": args.model_id,
            "profile": profile,
            "xp_path": str(xp_path),
            "checkpoint_path": str(checkpoint_path),
            "max_duration": max_duration,
        }

    model = MusicGen.get_pretrained(args.model_id, device=device)
    _optimize_for_inference(model)
    profile = resolve_generation_profile(profile_name=args.profile)
    return model, {"kind": "model_id", "model_id": args.model_id, "profile": profile}


def _read_prompt_audio(args: argparse.Namespace) -> tuple[torch.Tensor, int]:
    if args.prompt_audio is None:
        raise ValueError("--prompt-audio is required for continuation mode.")
    if args.prompt_duration_seconds is not None and args.prompt_duration_seconds >= args.duration_seconds:
        raise ValueError("--prompt-duration-seconds must be shorter than --duration-seconds.")
    wav, sample_rate = audio_read(
        Path(args.prompt_audio).expanduser().resolve(),
        seek_time=args.prompt_start_seconds,
        duration=args.prompt_duration_seconds or -1,
        pad=False,
    )
    return wav, int(sample_rate)


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)

    model, source = _load_model(args, device)
    if args.extend_stride >= model.max_duration:
        raise ValueError(
            f"--extend-stride must be shorter than this model's max duration "
            f"({model.max_duration:g}s); got {args.extend_stride:g}s."
        )
    model.set_generation_params(
        use_sampling=args.use_sampling,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        duration=args.duration_seconds,
        cfg_coef=args.cfg_coef,
        extend_stride=args.extend_stride,
    )
    # Live threshold read per step by the patched sampler; <=0 keeps stock path.
    try:
        model.lm._nr_min_p = float(args.min_p)
    except AttributeError:  # non-writable stand-in models in tests
        pass

    num_samples = max(1, args.num_samples)
    warnings: list[str] = []
    if args.mode == "unconditional":
        audio_values = model.generate_unconditional(num_samples)
    else:
        prompt_audio, prompt_sample_rate = _read_prompt_audio(args)
        if prompt_audio.dim() == 2:
            prompt_audio = prompt_audio[None]
        prompt_audio = prompt_audio.repeat(num_samples, 1, 1)
        audio_values = model.generate_continuation(prompt_audio, prompt_sample_rate)

    batch_audio = _to_numpy_audio_batch(audio_values)

    audio_paths: list[Path] = []
    for index, sample_ch_t in enumerate(batch_audio):
        audio_path = output_dir / f"sample_{index:03d}.wav"
        _write_wav(audio_path, sample_ch_t, int(model.sample_rate))
        audio_paths.append(audio_path)

    metadata = {
        "backend": "audiocraft",
        "mode": args.mode,
        "model_id": source.get("model_id", args.model_id),
        "device": device,
        "sample_rate": int(model.sample_rate),
        "batch_size": int(batch_audio.shape[0]),
        "source": source,
        "files": [str(path) for path in audio_paths],
        "warnings": warnings,
    }
    (output_dir / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
