from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol

from ...musicgen.backend import GenerationRequest
from ...musicgen.config import musicgen_script_env
from ...musicgen.runtime import load_musicgen_project_config, require_musicgen_python


@dataclass(frozen=True)
class StreamModelSource:
    model_id: str
    device: str | None = None
    profile_name: str | None = None
    xp_path: str | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class StreamGenerationResult:
    output_path: Path
    sample_rate: int
    backend: str
    model_id: str
    device: str
    warnings: list[str]
    metadata: dict[str, object]
    # All WAVs written for this request (best-of-N candidates); always
    # contains at least ``output_path``.
    candidate_paths: list[Path] = field(default_factory=list)


class StreamGenerator(Protocol):
    def generate(
        self,
        request: GenerationRequest,
        *,
        output_path: Path,
        source: StreamModelSource,
    ) -> StreamGenerationResult: ...

    def close(self) -> None: ...


class PersistentStreamGenerator:
    """Keeps one worker subprocess alive so the model loads once per session.

    ``stderr_path`` redirects the worker's diagnostics to a file instead of the
    parent's terminal. The worker deliberately routes generation-time stdout
    into stderr (see ``scripts/run_musicgen_stream_worker.py``), which is fine
    when the parent is scrolling text and fatal when it is drawing a TUI. A
    file rather than a pipe: a pipe would need a drain thread that never blocks
    and copes with tqdm's bare carriage returns, which is risk on the
    performance path for no gain.
    """

    def __init__(self, *, stderr_path: Path | None = None) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._request_index = 0
        self._stderr_path = Path(stderr_path) if stderr_path is not None else None
        self._stderr_handle: IO[str] | None = None

    def generate(
        self,
        request: GenerationRequest,
        *,
        output_path: Path,
        source: StreamModelSource,
    ) -> StreamGenerationResult:
        process = self._ensure_process()
        request_id = self._next_request_id()
        payload = _request_payload(
            request_id=request_id,
            request=request,
            output_path=output_path,
            source=source,
        )
        response = self._round_trip(process, payload, request_id=request_id)
        candidate_paths = [Path(str(p)) for p in response.get("candidate_paths") or []]
        if not candidate_paths:
            candidate_paths = [Path(str(response["output_path"]))]
        return StreamGenerationResult(
            output_path=Path(str(response["output_path"])),
            candidate_paths=candidate_paths,
            sample_rate=int(response["sample_rate"]),
            backend=str(response.get("backend", "audiocraft")),
            model_id=str(response["model_id"]),
            device=str(response["device"]),
            warnings=[str(warning) for warning in response.get("warnings", [])],
            metadata={
                "mode": str(response["mode"]),
                "source": response.get("source", {}),
            },
        )

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        try:
            if process.poll() is None and process.stdin is not None:
                shutdown_id = self._next_request_id()
                process.stdin.write(json.dumps({"request_id": shutdown_id, "command": "shutdown"}) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            for handle in (process.stdin, process.stdout, self._stderr_handle):
                try:
                    if handle is not None:
                        handle.close()
                except OSError:
                    pass
            self._stderr_handle = None

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        config = load_musicgen_project_config()
        worker_script = config.repo_root / "scripts" / "run_musicgen_stream_worker.py"
        if not worker_script.exists():
            raise FileNotFoundError(f"MusicGen stream worker script not found: {worker_script}")

        musicgen_python = Path(require_musicgen_python()).expanduser()
        env = musicgen_script_env()

        self._process = subprocess.Popen(
            [str(musicgen_python), str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._open_stderr(),
            text=True,
            bufsize=1,
            env=env,
        )
        return self._process

    def _open_stderr(self) -> IO[str] | None:
        """The child's stderr target: a session log when asked, else inherited."""
        if self._stderr_path is None:
            return None
        if self._stderr_handle is None or self._stderr_handle.closed:
            self._stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_handle = self._stderr_path.open("a", encoding="utf-8", buffering=1)
        return self._stderr_handle

    def _round_trip(
        self,
        process: subprocess.Popen[str],
        payload: dict[str, object],
        *,
        request_id: str,
    ) -> dict[str, object]:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("MusicGen stream worker pipes are unavailable.")
        try:
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("MusicGen stream worker exited before it accepted the request.") from exc

        line = process.stdout.readline()
        if line == "":
            exit_code = process.poll()
            detail = f" exit_code={exit_code}" if exit_code is not None else ""
            raise RuntimeError(f"MusicGen stream worker exited before responding.{detail}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MusicGen stream worker returned malformed JSON: {line.strip()}") from exc
        if not isinstance(response, dict):
            raise RuntimeError("MusicGen stream worker returned a non-object response.")
        if response.get("request_id") != request_id:
            raise RuntimeError(
                f"MusicGen stream worker response id mismatch: {response.get('request_id')!r} != {request_id!r}"
            )
        if not response.get("ok", False):
            error = str(response.get("error", "unknown worker error"))
            raise RuntimeError(f"MusicGen stream worker failed: {error}")
        return response

    def _next_request_id(self) -> str:
        self._request_index += 1
        return f"stream-{self._request_index:06d}"


def _request_payload(
    *,
    request_id: str,
    request: GenerationRequest,
    output_path: Path,
    source: StreamModelSource,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "command": "generate",
        "output_path": str(output_path.expanduser().resolve()),
        "num_samples": max(1, int(request.batch_size)),
        "mode": request.mode,
        "duration_seconds": request.duration_seconds,
        "seed": request.seed,
        "prompt_audio_path": request.prompt_audio_path,
        "prompt_start_seconds": request.prompt_start_seconds,
        "prompt_duration_seconds": request.prompt_duration_seconds,
        "use_sampling": request.use_sampling,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "min_p": request.min_p,
        "temperature": request.temperature,
        "cfg_coef": request.cfg_coef,
        "extend_stride": request.extend_stride,
        "model_id": source.model_id,
        "device": source.device,
        "profile": source.profile_name,
        "xp_path": source.xp_path,
        "checkpoint_path": source.checkpoint_path,
    }
