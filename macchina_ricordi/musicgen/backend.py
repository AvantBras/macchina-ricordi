from __future__ import annotations

import json
import re
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..instrument.provenance import resolve_generation_profile
from .config import musicgen_script_env
from .runtime import load_musicgen_project_config, require_musicgen_python

VALID_GENERATION_MODES = {"unconditional", "continuation"}
CHECKPOINT_EPOCH_PATTERN = re.compile(r"^checkpoint_(\d+)\.th$")
DEFAULT_EXTEND_STRIDE_SECONDS = 8.0


@dataclass(frozen=True)
class GenerationRequest:
    prompts: Sequence[str]
    batch_size: int = 1
    mode: str = "unconditional"
    duration_seconds: float = 10.0
    max_new_tokens: int = 256
    device: str | None = None
    seed: int | None = None
    prompt_audio_path: str | None = None
    prompt_start_seconds: float = 0.0
    prompt_duration_seconds: float | None = None
    use_sampling: bool = True
    top_k: int = 250
    top_p: float = 0.0
    min_p: float = 0.0
    temperature: float = 1.0
    cfg_coef: float = 3.0
    extend_stride: float = DEFAULT_EXTEND_STRIDE_SECONDS


@dataclass
class GenerationResult:
    audio_values: object
    sample_rate: int
    backend: str
    model_id: str
    device: str
    warnings: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


def build_audiocraft_backend(
    *,
    model_id: str | None = None,
    device: str | None = None,
    profile_name: str | None = None,
    xp_path: str | Path | None = None,
    checkpoint: str | None = None,
    checkpoint_path: str | Path | None = None,
):
    resolved_checkpoint = resolve_generation_checkpoint(
        xp_path=xp_path,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    resolved_profile = resolve_generation_profile(
        profile_name=profile_name,
        xp_path=xp_path,
        checkpoint_path=resolved_checkpoint,
    )
    return AudioCraftMusicGenBackend(
        model_id=model_id,
        device=device,
        profile_name=resolved_profile,
        xp_path=xp_path,
        checkpoint=checkpoint,
        checkpoint_path=resolved_checkpoint,
    )


def resolve_generation_checkpoint(
    *,
    xp_path: str | Path | None = None,
    checkpoint: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path | None:
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Generation checkpoint does not exist: {resolved}")
        return resolved

    if xp_path is None:
        return None

    resolved_xp = Path(xp_path).expanduser().resolve()

    # A model bundle is self-contained: its weights live in a safetensors file,
    # not checkpoint_*.th. Treat the bundle dir itself as the model source and
    # pass it through rather than globbing for a Dora checkpoint it will never
    # contain.
    from macchina_ricordi.models.bundle_runtime import is_bundle_dir

    if is_bundle_dir(resolved_xp):
        return resolved_xp

    if not resolved_xp.exists() or not resolved_xp.is_dir():
        raise FileNotFoundError(f"Generation XP folder does not exist: {resolved_xp}")

    requested = (checkpoint or "latest").strip().lower()
    if requested == "final":
        candidate = resolved_xp / "checkpoint.th"
    elif requested == "latest":
        epoch_checkpoints = sorted(
            (
                (int(match.group(1)), path)
                for path in resolved_xp.glob("checkpoint_*.th")
                if (match := CHECKPOINT_EPOCH_PATTERN.match(path.name)) is not None
            ),
            key=lambda item: item[0],
        )
        candidate = epoch_checkpoints[-1][1] if epoch_checkpoints else resolved_xp / "checkpoint.th"
    elif requested.isdigit():
        candidate = resolved_xp / f"checkpoint_{requested}.th"
    else:
        raise ValueError("Unsupported checkpoint selector. Expected latest, final, or an epoch number.")

    if not candidate.exists():
        raise FileNotFoundError(f"Generation checkpoint does not exist: {candidate}")
    return candidate.resolve()


def _read_wav_file(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_bytes = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width {sample_width} for {path}. Expected 16-bit PCM.")

    audio = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32, copy=False)
    audio = audio.reshape(-1, channels).T
    audio /= 32767.0
    return audio, int(sample_rate)


def _load_helper_result(output_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    metadata_path = output_dir / "result.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_rate = int(payload["sample_rate"])
    file_paths = [Path(path) for path in payload["files"]]
    if not file_paths:
        raise ValueError("Generation helper did not report any output audio files.")

    batch: list[np.ndarray] = []
    frame_count: int | None = None
    channel_count: int | None = None
    for file_path in file_paths:
        audio, file_sample_rate = _read_wav_file(file_path)
        if file_sample_rate != sample_rate:
            raise ValueError(f"Inconsistent sample rate in {file_path}: {file_sample_rate} != {sample_rate}")
        if frame_count is None:
            frame_count = int(audio.shape[1])
            channel_count = int(audio.shape[0])
        elif audio.shape != (channel_count, frame_count):
            raise ValueError("Generation helper returned WAV files with inconsistent shapes.")
        batch.append(audio)

    return np.stack(batch, axis=0), payload


class AudioCraftMusicGenBackend:
    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
        profile_name: str | None = None,
        xp_path: str | Path | None = None,
        checkpoint: str | None = None,
        checkpoint_path: str | Path | None = None,
    ):
        self.model_id = model_id or "facebook/musicgen-small"
        self.requested_device = device
        self.profile_name = profile_name
        self.xp_path = Path(xp_path).expanduser().resolve() if xp_path is not None else None
        self.checkpoint = checkpoint
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else None

    def _helper_script(self) -> Path:
        config = load_musicgen_project_config()
        return config.repo_root / "scripts" / "run_musicgen_generate.py"

    def _build_command(self, request: GenerationRequest, output_dir: Path) -> list[str]:
        mode = request.mode.strip().lower()
        if mode not in VALID_GENERATION_MODES:
            raise ValueError(
                f"Unsupported generation mode '{request.mode}'. Expected one of: {sorted(VALID_GENERATION_MODES)}"
            )
        if mode == "continuation":
            if request.prompt_audio_path is None:
                raise ValueError("continuation generation requires prompt_audio_path.")
            prompt_duration = request.prompt_duration_seconds
            if prompt_duration is not None and prompt_duration >= request.duration_seconds:
                raise ValueError("prompt_duration_seconds must be shorter than duration_seconds.")

        musicgen_python = Path(require_musicgen_python()).expanduser()
        helper_script = self._helper_script()
        if not helper_script.exists():
            raise FileNotFoundError(f"AudioCraft generation helper script not found: {helper_script}")

        resolved_checkpoint = resolve_generation_checkpoint(
            xp_path=self.xp_path,
            checkpoint=self.checkpoint,
            checkpoint_path=self.checkpoint_path,
        )

        command = [
            str(musicgen_python),
            str(helper_script),
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
            "--model-id",
            self.model_id,
            "--num-samples",
            str(max(1, request.batch_size)),
            "--duration-seconds",
            str(request.duration_seconds),
            "--temperature",
            str(request.temperature),
            "--top-k",
            str(request.top_k),
            "--top-p",
            str(request.top_p),
            "--min-p",
            str(request.min_p),
            "--cfg-coef",
            str(request.cfg_coef),
            "--extend-stride",
            str(request.extend_stride),
        ]
        command.append("--use-sampling" if request.use_sampling else "--no-use-sampling")
        if self.requested_device:
            command.extend(["--device", self.requested_device])
        if request.seed is not None:
            command.extend(["--seed", str(request.seed)])
        if self.profile_name:
            command.extend(["--profile", self.profile_name])
        if self.xp_path is not None:
            command.extend(["--xp-path", str(self.xp_path)])
        if resolved_checkpoint is not None:
            command.extend(["--checkpoint-path", str(resolved_checkpoint)])
        if request.prompt_audio_path is not None:
            command.extend(["--prompt-audio", str(Path(request.prompt_audio_path).expanduser().resolve())])
            command.extend(["--prompt-start-seconds", str(request.prompt_start_seconds)])
            if request.prompt_duration_seconds is not None:
                command.extend(["--prompt-duration-seconds", str(request.prompt_duration_seconds)])
        return command

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with tempfile.TemporaryDirectory(prefix="macchina_ricordi_audiocraft_") as temp_dir:
            output_dir = Path(temp_dir)
            command = self._build_command(request, output_dir)
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=musicgen_script_env(),
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() if exc.stderr else ""
                message = f"AudioCraft MusicGen helper failed with exit code {exc.returncode}."
                if stderr:
                    message = f"{message} {stderr}"
                raise RuntimeError(message) from exc

            audio_values, payload = _load_helper_result(output_dir)

        metadata: dict[str, object] = {
            "mode": str(payload["mode"]),
            "source": payload.get("source", {}),
        }
        if "requested_mode" in payload:
            metadata["requested_mode"] = str(payload["requested_mode"])

        return GenerationResult(
            audio_values=audio_values,
            sample_rate=int(payload["sample_rate"]),
            backend=str(payload.get("backend", "audiocraft")),
            model_id=str(payload["model_id"]),
            device=str(payload["device"]),
            warnings=[str(warning) for warning in payload.get("warnings", [])],
            metadata=metadata,
        )
