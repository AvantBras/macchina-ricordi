from __future__ import annotations

import contextlib
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from run_musicgen_generate import (
    _load_model,
    _read_prompt_audio,
    _set_seed,
    _to_numpy_audio_batch,
    _write_wav,
)


class WorkerState:
    def __init__(self) -> None:
        self.model = None
        self.source: dict[str, object] = {}
        self.source_key: tuple[object, ...] | None = None
        self.device: str | None = None

    def load_for(self, payload: dict[str, object]) -> None:
        source_key = (
            payload.get("model_id"),
            payload.get("device"),
            payload.get("profile"),
            payload.get("xp_path"),
            payload.get("checkpoint_path"),
        )
        if self.model is not None and self.source_key == source_key:
            return

        device = str(payload.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        args = SimpleNamespace(
            model_id=str(payload.get("model_id") or "facebook/musicgen-small"),
            profile=payload.get("profile"),
            xp_path=payload.get("xp_path"),
            checkpoint_path=payload.get("checkpoint_path"),
        )
        self.model, self.source = _load_model(args, device)
        self.source_key = source_key
        self.device = device


def _generate(payload: dict[str, object], state: WorkerState) -> dict[str, object]:
    state.load_for(payload)
    model = state.model
    if model is None or state.device is None:
        raise RuntimeError("Worker model was not initialized.")

    extend_stride = float(payload["extend_stride"])
    if extend_stride >= model.max_duration:
        raise ValueError(
            f"extend_stride must be shorter than this model's max duration "
            f"({model.max_duration:g}s); got {extend_stride:g}s."
        )

    _set_seed(payload.get("seed"), state.device)
    model.set_generation_params(
        use_sampling=bool(payload["use_sampling"]),
        top_k=int(payload["top_k"]),
        top_p=float(payload["top_p"]),
        temperature=float(payload["temperature"]),
        duration=float(payload["duration_seconds"]),
        cfg_coef=float(payload["cfg_coef"]),
        extend_stride=extend_stride,
    )
    # Live min-p threshold read per step by the patched sampler; <=0 = stock.
    try:
        model.lm._nr_min_p = float(payload.get("min_p", 0.0) or 0.0)
    except AttributeError:  # non-writable stand-in models in tests
        pass

    mode = str(payload["mode"])
    num_samples = max(1, int(payload.get("num_samples", 1)))
    warnings: list[str] = []
    if mode == "unconditional":
        audio_values = model.generate_unconditional(num_samples)
    elif mode == "continuation":
        prompt_args = SimpleNamespace(
            prompt_audio=payload.get("prompt_audio_path"),
            prompt_start_seconds=float(payload.get("prompt_start_seconds", 0.0)),
            prompt_duration_seconds=payload.get("prompt_duration_seconds"),
            duration_seconds=float(payload["duration_seconds"]),
        )
        prompt_audio, prompt_sample_rate = _read_prompt_audio(prompt_args)
        if prompt_audio.dim() == 2:
            prompt_audio = prompt_audio[None]
        if num_samples > 1:
            prompt_audio = prompt_audio.repeat(num_samples, 1, 1)
        audio_values = model.generate_continuation(prompt_audio, prompt_sample_rate)
    else:
        raise ValueError(f"Unsupported stream generation mode: {mode}")

    batch_audio = _to_numpy_audio_batch(audio_values)
    output_path = Path(str(payload["output_path"])).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_paths: list[Path] = [output_path]
    for index in range(1, min(num_samples, batch_audio.shape[0])):
        candidate_paths.append(output_path.parent / f"{output_path.stem}_cand{index}{output_path.suffix}")
    for path, audio in zip(candidate_paths, batch_audio, strict=False):
        _write_wav(path, audio, int(model.sample_rate))
    return {
        "output_path": str(output_path),
        "candidate_paths": [str(path) for path in candidate_paths],
        "sample_rate": int(model.sample_rate),
        "backend": "audiocraft",
        "mode": mode,
        "model_id": str(payload.get("model_id") or "facebook/musicgen-small"),
        "device": state.device,
        "source": state.source,
        "warnings": warnings,
    }


def _write_response(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    state = WorkerState()
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("Worker request must be a JSON object.")
            request_id = str(payload.get("request_id", ""))
            command = payload.get("command")
            if command == "shutdown":
                _write_response({"request_id": request_id, "ok": True, "shutdown": True})
                return 0
            if command != "generate":
                raise ValueError(f"Unsupported worker command: {command!r}")
            with contextlib.redirect_stdout(sys.stderr):
                result = _generate(payload, state)
            response = {"request_id": request_id, "ok": True}
            response.update(result)
        except Exception as exc:  # noqa: BLE001
            request_id = ""
            try:
                maybe_payload = json.loads(line)
                if isinstance(maybe_payload, dict):
                    request_id = str(maybe_payload.get("request_id", ""))
            except Exception:  # noqa: BLE001, S110
                pass
            traceback.print_exc(file=sys.stderr)
            response = {"request_id": request_id, "ok": False, "error": str(exc)}
        _write_response(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
