from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import librosa

from .schema import CaptionerBackendInfo, CaptionerWitness, ListeningDossier

DEFAULT_CAPTIONER_BACKEND = "qwen2-audio"
DEFAULT_QWEN2_AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
# Music-specialized Audio Flamingo 3 variant; NVIDIA OneWay Noncommercial license.
DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID = "nvidia/music-flamingo-hf"
CAPTIONER_BACKEND_MODEL_IDS = {
    "qwen2-audio": DEFAULT_QWEN2_AUDIO_MODEL_ID,
    "audio-flamingo-3": DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID,
}
SUPPORTED_CAPTIONER_BACKENDS = tuple(CAPTIONER_BACKEND_MODEL_IDS)
DEFAULT_CAPTIONER_MODEL_ID = ""
DEFAULT_CAPTIONER_DEVICE = "auto"
DEFAULT_CAPTIONER_DTYPE = "auto"
DEFAULT_CAPTIONER_QUANTIZATION = "auto"
DEFAULT_CAPTIONER_MAX_NEW_TOKENS = 220
DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS = 120.0
DEFAULT_CAPTIONER_PROMPT_PROFILE = "experimental_texture_v1"
DEFAULT_CAPTIONER_TRUST_REMOTE_CODE = True
CAPTIONER_INSTALL_HINT = 'Install captioner witnesses with: pip install -e ".[captioner-witness]"'

EXPERIMENTAL_TEXTURE_PROMPT = """Describe this audio for an experimental music archive.
Do not assume there is melody, harmony, rhythm, or conventional instrumentation.
Focus on texture, density, transients, space, repetition, change over time, and
uncertainty. Use timestamps when possible. Mark guesses as guesses."""

_CLAIM_PATTERNS: dict[str, re.Pattern[str]] = {
    "steady_beat": re.compile(
        r"\b((steady|stable|regular|consistent)\s+(beat|rhythm|pulse)|dance\s+groove)\b",
        re.IGNORECASE,
    ),
    "voice": re.compile(r"\b(voice|voices|vocal|vocals|sung|singing|speech|spoken|speaker)\b", re.IGNORECASE),
    "melody": re.compile(r"\b(melody|melodic|tune)\b", re.IGNORECASE),
    "harmony": re.compile(r"\b(harmony|harmonic)\b", re.IGNORECASE),
    "chords": re.compile(r"\b(chord|chords)\b", re.IGNORECASE),
    "piano": re.compile(r"\bpiano\b", re.IGNORECASE),
    "guitar": re.compile(r"\bguitar\b", re.IGNORECASE),
    "drums": re.compile(r"\b(drum|drums|snare|kick drum|hi-hat|hihat)\b", re.IGNORECASE),
    "pop": re.compile(r"\b(pop song|pop music)\b", re.IGNORECASE),
    "dance": re.compile(r"\b(dance|danceable|groove)\b", re.IGNORECASE),
}

_CONVENTIONAL_CLAIM_TO_SEMANTIC = {
    "piano": "piano_chords",
    "guitar": "acoustic_guitar",
    "pop": "pop_song",
    "dance": "dance_groove",
}


class CaptionerBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested captioner backend cannot be used."""


class CaptionerBackend(Protocol):
    name: str
    model_id: str
    device: str
    dtype: str
    quantization: str
    trust_remote_code: bool

    def generate_caption(
        self,
        audio_path: str | Path,
        *,
        prompt: str,
        max_new_tokens: int,
        max_audio_seconds: float,
    ) -> str:
        """Return a free-form captioner response for one audio file."""


def default_captioner_model_id(backend: str) -> str:
    try:
        return CAPTIONER_BACKEND_MODEL_IDS[backend]
    except KeyError as exc:
        raise CaptionerBackendUnavailable(
            f"Unsupported captioner backend: {backend}. Supported: {', '.join(SUPPORTED_CAPTIONER_BACKENDS)}"
        ) from exc


class Qwen2AudioCaptionerBackend:
    name = "qwen2-audio"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_QWEN2_AUDIO_MODEL_ID,
        device: str = DEFAULT_CAPTIONER_DEVICE,
        dtype: str = DEFAULT_CAPTIONER_DTYPE,
        quantization: str = DEFAULT_CAPTIONER_QUANTIZATION,
        trust_remote_code: bool = DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except Exception as exc:  # pragma: no cover - exercised through public factory
            raise CaptionerBackendUnavailable(
                f"Qwen2-Audio dependencies are not available. {CAPTIONER_INSTALL_HINT}"
            ) from exc

        self.model_id = model_id
        self.device = _resolve_captioner_device(device)
        self.dtype, torch_dtype = _resolve_captioner_dtype(dtype, device=self.device)
        self.quantization = _resolve_captioner_quantization(quantization, device=self.device)
        self.trust_remote_code = trust_remote_code

        try:
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
            if torch_dtype is not None:
                kwargs["dtype"] = torch_dtype
            if self.quantization == "bnb-4bit":
                kwargs["quantization_config"] = _bnb_4bit_config(torch_dtype=torch_dtype, device=self.device)
                kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                kwargs["device_map"] = "auto"
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
            if self.device == "cpu":
                self._model = self._model.to("cpu")
        except Exception as exc:  # pragma: no cover - depends on external model runtime
            raise CaptionerBackendUnavailable(
                f"Qwen2-Audio could not initialize: {exc}. {CAPTIONER_INSTALL_HINT}"
            ) from exc

        self._torch = torch

    def generate_caption(
        self,
        audio_path: str | Path,
        *,
        prompt: str,
        max_new_tokens: int,
        max_audio_seconds: float,
    ) -> str:
        path = Path(audio_path).expanduser().resolve()
        try:
            sampling_rate = int(self._processor.feature_extractor.sampling_rate)
            audio, _ = librosa.load(path, sr=sampling_rate, mono=True, duration=max_audio_seconds)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": str(path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            if hasattr(self._processor, "apply_chat_template"):
                text = self._processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                text = f"<|audio_bos|><|AUDIO|><|audio_eos|>{prompt}"
            inputs = self._processor(
                text=text,
                audio=[audio],
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(_model_device(self._model))
            with self._torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            input_ids = inputs["input_ids"]
            generated = generated[:, input_ids.size(1) :]
            return self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except Exception as exc:
            raise CaptionerBackendUnavailable(f"Qwen2-Audio captioning failed for {path}: {exc}") from exc


class AudioFlamingo3CaptionerBackend:
    name = "audio-flamingo-3"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID,
        device: str = DEFAULT_CAPTIONER_DEVICE,
        dtype: str = DEFAULT_CAPTIONER_DTYPE,
        quantization: str = DEFAULT_CAPTIONER_QUANTIZATION,
        trust_remote_code: bool = DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    ) -> None:
        try:
            import torch
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        except Exception as exc:  # pragma: no cover - exercised through public factory
            raise CaptionerBackendUnavailable(
                f"Audio Flamingo 3 requires transformers>=5.0.0. {CAPTIONER_INSTALL_HINT}"
            ) from exc

        self.model_id = model_id
        self.device = _resolve_captioner_device(device)
        self.dtype, torch_dtype = _resolve_captioner_dtype(dtype, device=self.device)
        self.quantization = _resolve_captioner_quantization(quantization, device=self.device)
        self.trust_remote_code = trust_remote_code
        self._torch_dtype = torch_dtype

        try:
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
            if torch_dtype is not None:
                kwargs["dtype"] = torch_dtype
            if self.quantization == "bnb-4bit":
                kwargs["quantization_config"] = _bnb_4bit_config(torch_dtype=torch_dtype, device=self.device)
                kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                kwargs["device_map"] = "auto"
            self._model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
            if self.device == "cpu":
                self._model = self._model.to("cpu")
            audio_tower = getattr(getattr(self._model, "model", None), "audio_tower", None)
            if torch_dtype is not None and audio_tower is not None:
                # transformers keeps embed_positions in fp32, which promotes hidden
                # states to fp32 and crashes the half-precision layer norms.
                audio_tower.to(torch_dtype)
        except Exception as exc:  # pragma: no cover - depends on external model runtime
            raise CaptionerBackendUnavailable(
                f"Audio Flamingo 3 could not initialize: {exc}. {CAPTIONER_INSTALL_HINT}"
            ) from exc

        self._torch = torch

    def generate_caption(
        self,
        audio_path: str | Path,
        *,
        prompt: str,
        max_new_tokens: int,
        max_audio_seconds: float,
    ) -> str:
        path = Path(audio_path).expanduser().resolve()
        try:
            sampling_rate = int(self._processor.feature_extractor.sampling_rate)
            audio, _ = librosa.load(path, sr=sampling_rate, mono=True, duration=max_audio_seconds)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": str(path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            if hasattr(self._processor, "apply_chat_template"):
                text = self._processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                text = f"<sound>{prompt}"
            inputs = self._processor(
                text=text,
                audio=[audio],
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            inputs = inputs.to(_model_device(self._model))
            if self._torch_dtype is not None:
                # AF3's audio tower does not auto-cast mel features to the model dtype.
                inputs = inputs.to(self._torch_dtype)
            with self._torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            input_ids = inputs["input_ids"]
            generated = generated[:, input_ids.size(1) :]
            return self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except Exception as exc:
            raise CaptionerBackendUnavailable(f"Audio Flamingo 3 captioning failed for {path}: {exc}") from exc


def create_captioner_backend(
    *,
    backend: str = DEFAULT_CAPTIONER_BACKEND,
    model_id: str = DEFAULT_CAPTIONER_MODEL_ID,
    device: str = DEFAULT_CAPTIONER_DEVICE,
    dtype: str = DEFAULT_CAPTIONER_DTYPE,
    quantization: str = DEFAULT_CAPTIONER_QUANTIZATION,
    trust_remote_code: bool = DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
) -> CaptionerBackend:
    resolved_model_id = model_id or default_captioner_model_id(backend)
    backend_classes: dict[str, type] = {
        "qwen2-audio": Qwen2AudioCaptionerBackend,
        "audio-flamingo-3": AudioFlamingo3CaptionerBackend,
    }
    backend_cls = backend_classes.get(backend)
    if backend_cls is None:
        raise CaptionerBackendUnavailable(
            f"Unsupported captioner backend: {backend}. Supported: {', '.join(SUPPORTED_CAPTIONER_BACKENDS)}"
        )
    return backend_cls(
        model_id=resolved_model_id,
        device=device,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
    )


def run_captioner_witness(
    audio_path: str | Path,
    dossier: ListeningDossier | Mapping[str, Any],
    *,
    backend: str = DEFAULT_CAPTIONER_BACKEND,
    model_id: str = DEFAULT_CAPTIONER_MODEL_ID,
    device: str = DEFAULT_CAPTIONER_DEVICE,
    dtype: str = DEFAULT_CAPTIONER_DTYPE,
    quantization: str = DEFAULT_CAPTIONER_QUANTIZATION,
    max_new_tokens: int = DEFAULT_CAPTIONER_MAX_NEW_TOKENS,
    max_audio_seconds: float = DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS,
    prompt_profile: str = DEFAULT_CAPTIONER_PROMPT_PROFILE,
    trust_remote_code: bool = DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    captioner_backend: CaptionerBackend | None = None,
) -> CaptionerWitness:
    if prompt_profile != DEFAULT_CAPTIONER_PROMPT_PROFILE:
        raise ValueError(f"Unsupported captioner prompt profile: {prompt_profile}")
    resolved_backend = captioner_backend or create_captioner_backend(
        backend=backend,
        model_id=model_id,
        device=device,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
    )
    raw_response = resolved_backend.generate_caption(
        audio_path,
        prompt=EXPERIMENTAL_TEXTURE_PROMPT,
        max_new_tokens=max_new_tokens,
        max_audio_seconds=max_audio_seconds,
    )
    payload = _dossier_payload(dossier)
    claims = _extract_claims(raw_response)
    warnings = _cross_examine_claims(claims, payload)
    return CaptionerWitness(
        backend=CaptionerBackendInfo(
            name=getattr(resolved_backend, "name", backend),
            model_id=getattr(resolved_backend, "model_id", model_id),
            device=getattr(resolved_backend, "device", device),
            dtype=getattr(resolved_backend, "dtype", dtype),
            quantization=getattr(resolved_backend, "quantization", "none"),
            trust_remote_code=getattr(resolved_backend, "trust_remote_code", trust_remote_code),
        ),
        prompt_profile=prompt_profile,
        prompt=EXPERIMENTAL_TEXTURE_PROMPT,
        summary=_compact_summary(raw_response),
        raw_response=raw_response,
        warnings=warnings,
        agreement=_agreement_from_warnings(claims, payload, warnings),
        claims=claims,
    )


def attach_captioner_witness(
    dossier: ListeningDossier,
    *,
    audio_path: str | Path | None = None,
    backend: str = DEFAULT_CAPTIONER_BACKEND,
    model_id: str = DEFAULT_CAPTIONER_MODEL_ID,
    device: str = DEFAULT_CAPTIONER_DEVICE,
    dtype: str = DEFAULT_CAPTIONER_DTYPE,
    quantization: str = DEFAULT_CAPTIONER_QUANTIZATION,
    max_new_tokens: int = DEFAULT_CAPTIONER_MAX_NEW_TOKENS,
    max_audio_seconds: float = DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS,
    prompt_profile: str = DEFAULT_CAPTIONER_PROMPT_PROFILE,
    trust_remote_code: bool = DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    captioner_backend: CaptionerBackend | None = None,
) -> ListeningDossier:
    captioner_witness = run_captioner_witness(
        audio_path or dossier.audio.path,
        dossier,
        backend=backend,
        model_id=model_id,
        device=device,
        dtype=dtype,
        quantization=quantization,
        max_new_tokens=max_new_tokens,
        max_audio_seconds=max_audio_seconds,
        prompt_profile=prompt_profile,
        trust_remote_code=trust_remote_code,
        captioner_backend=captioner_backend,
    )
    return replace(dossier, captioner_witness=captioner_witness)


def captioner_summary_from_payload(payload: Mapping[str, object]) -> dict[str, object] | None:
    witness = payload.get("captioner_witness")
    if not isinstance(witness, dict):
        return None
    backend = witness.get("backend") if isinstance(witness.get("backend"), dict) else {}
    return {
        "backend": backend.get("name") if isinstance(backend, dict) else None,
        "model_id": backend.get("model_id") if isinstance(backend, dict) else None,
        "quantization": backend.get("quantization") if isinstance(backend, dict) else None,
        "prompt_profile": witness.get("prompt_profile"),
        "summary": witness.get("summary") if isinstance(witness.get("summary"), str) else "",
        "warnings": list(witness.get("warnings", [])) if isinstance(witness.get("warnings"), list) else [],
        "agreement": dict(witness.get("agreement", {})) if isinstance(witness.get("agreement"), dict) else {},
        "claims": dict(witness.get("claims", {})) if isinstance(witness.get("claims"), dict) else {},
    }


def extract_caption_claims(text: str) -> dict[str, bool]:
    return _extract_claims(text)


def cross_examine_caption_claims(
    claims: dict[str, bool],
    dossier: ListeningDossier | Mapping[str, Any],
) -> list[str]:
    return _cross_examine_claims(claims, _dossier_payload(dossier))


def _dossier_payload(dossier: ListeningDossier | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(dossier, Mapping):
        return dossier
    return dossier.to_dict()


def _extract_claims(text: str) -> dict[str, bool]:
    return {key: bool(pattern.search(text)) for key, pattern in _CLAIM_PATTERNS.items()}


def _cross_examine_claims(claims: dict[str, bool], payload: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    pulse, pulse_instability = _pulse_evidence(payload)
    if claims.get("steady_beat") and (pulse in {"drifting", "limping", "unstable"} or pulse_instability >= 0.55):
        warnings.append(
            f"captioner claims steady beat, but deterministic pulse is {pulse or 'unknown'} "
            f"with pulse_instability={pulse_instability:.3f}"
        )

    semantic = payload.get("semantic_probes")
    if claims.get("voice"):
        if not isinstance(semantic, dict):
            warnings.append("captioner claims voice/singing, but no semantic probes are present; treat as unverified")
        elif _semantic_key_is_weak(semantic, "sung_voice") or _semantic_score(semantic, "sung_voice") < 0.45:
            warnings.append("captioner claims voice/singing, but semantic sung_voice evidence is weak")

    if claims.get("melody") or claims.get("harmony") or claims.get("chords"):
        warnings.append(
            "captioner claims melody/harmony/chords, but no melodic detector is available; mark unsupported"
        )

    for claim, semantic_key in _CONVENTIONAL_CLAIM_TO_SEMANTIC.items():
        if not claims.get(claim):
            continue
        if not isinstance(semantic, dict):
            warnings.append(f"captioner claims {claim}, but no semantic probes are present; treat as captioner-only")
        elif _semantic_key_is_weak(semantic, semantic_key) or _semantic_score(semantic, semantic_key) < 0.45:
            warnings.append(f"captioner claims {claim}, but semantic {semantic_key} evidence is weak")
    if claims.get("drums"):
        warnings.append("captioner claims drums, but no drum identity detector is available; treat as captioner-only")
    return warnings


def _agreement_from_warnings(
    claims: dict[str, bool],
    payload: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, str]:
    has_claims = any(claims.values())
    feature_related = [warning for warning in warnings if "deterministic" in warning or "melodic detector" in warning]
    semantic_related = [warning for warning in warnings if "semantic" in warning]
    semantic = payload.get("semantic_probes")
    return {
        "with_features": "low" if feature_related else "medium" if has_claims else "unknown",
        "with_semantic_probes": (
            "unknown"
            if not isinstance(semantic, dict)
            else "low"
            if semantic_related
            else "medium"
            if has_claims
            else "high"
        ),
    }


def _pulse_evidence(payload: Mapping[str, Any]) -> tuple[str, float]:
    whole_clip = payload.get("whole_clip")
    if not isinstance(whole_clip, Mapping):
        return "", 0.0
    description = whole_clip.get("description")
    features = whole_clip.get("features")
    pulse = ""
    if isinstance(description, Mapping) and isinstance(description.get("pulse"), str):
        pulse = description["pulse"]
    pulse_instability = 0.0
    if isinstance(features, Mapping) and isinstance(features.get("pulse_instability"), int | float):
        pulse_instability = float(features["pulse_instability"])
    return pulse, pulse_instability


def _semantic_key_is_weak(semantic: Mapping[str, Any], key: str) -> bool:
    weak = semantic.get("weak")
    if not isinstance(weak, list):
        return False
    return any(isinstance(item, Mapping) and item.get("key") == key for item in weak)


def _semantic_score(semantic: Mapping[str, Any], key: str) -> float:
    scores = semantic.get("scores")
    if not isinstance(scores, list):
        return 0.0
    for item in scores:
        if isinstance(item, Mapping) and item.get("key") == key and isinstance(item.get("score"), int | float):
            return float(item["score"])
    return 0.0


def _compact_summary(raw_response: str) -> str:
    text = " ".join(raw_response.split())
    return text[:500]


def _resolve_captioner_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device not in {"cpu", "cuda"}:
        raise CaptionerBackendUnavailable(f"Unsupported captioner device: {device}")
    return device


def _resolve_captioner_dtype(dtype: str, *, device: str) -> tuple[str, object | None]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised through public factory
        raise CaptionerBackendUnavailable(f"PyTorch is not available. {CAPTIONER_INSTALL_HINT}") from exc
    if dtype == "auto":
        dtype = "float16" if device == "cuda" else "float32"
    if dtype == "float32":
        return dtype, torch.float32
    if dtype == "float16":
        return dtype, torch.float16
    if dtype == "bfloat16":
        return dtype, torch.bfloat16
    raise CaptionerBackendUnavailable(f"Unsupported captioner dtype: {dtype}")


def _resolve_captioner_quantization(quantization: str, *, device: str) -> str:
    if quantization == "auto":
        return "bnb-4bit" if device == "cuda" else "none"
    if quantization in {"none", "bnb-4bit"}:
        return quantization
    raise CaptionerBackendUnavailable(f"Unsupported captioner quantization: {quantization}")


def _bnb_4bit_config(*, torch_dtype: object | None, device: str) -> object:
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - exercised through public factory
        raise CaptionerBackendUnavailable(
            f"Qwen2-Audio 4-bit loading requires transformers, torch, and bitsandbytes. {CAPTIONER_INSTALL_HINT}"
        ) from exc
    if device != "cuda" or not torch.cuda.is_available():
        raise CaptionerBackendUnavailable("Qwen2-Audio 4-bit loading requires a CUDA device")
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise CaptionerBackendUnavailable(
            f"Qwen2-Audio 4-bit loading requires bitsandbytes. {CAPTIONER_INSTALL_HINT}"
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype or torch.float16,
    )


def _model_device(model: object) -> object:
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"
