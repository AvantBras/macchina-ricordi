from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Protocol

import numpy as np

from .schema import ListeningDossier, SemanticBackendInfo, SemanticProbeReport, SemanticProbeScore

VOCABULARY_VERSION = "macchina_ricordi.semantic_probes.v0"
DEFAULT_SEMANTIC_BACKEND = "msclap"
DEFAULT_SEMANTIC_MODEL_VERSION = "2023"
DEFAULT_SEMANTIC_DEVICE = "auto"
SEMANTIC_INSTALL_HINT = 'Install semantic probes with: pip install -e ".[semantic-listening]"'

STRONG_THRESHOLD = 0.60
WEAK_NEGATIVE_THRESHOLD = 0.20
UNCERTAIN_NEGATIVE_THRESHOLD = 0.45
MAX_FINDINGS = 8


class SemanticBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested semantic backend cannot be used."""


class SemanticProbeBackend(Protocol):
    name: str
    version: str
    device: str
    model_path: str | None

    def score_audio_text(self, audio_path: str | Path, prompts: Sequence[str]) -> Sequence[float]:
        """Return one audio-text similarity score per prompt."""


@dataclass(frozen=True)
class SemanticProbeDefinition:
    key: str
    group: str
    label: str
    prompt: str


DEFAULT_SEMANTIC_PROBES: tuple[SemanticProbeDefinition, ...] = (
    SemanticProbeDefinition("metallic", "material", "metallic", "a metallic sound texture"),
    SemanticProbeDefinition("wooden", "material", "wooden", "a wooden sound texture"),
    SemanticProbeDefinition("plastic", "material", "plastic", "a plastic sound texture"),
    SemanticProbeDefinition("glassy", "material", "glassy", "a glassy sound texture"),
    SemanticProbeDefinition("dusty", "material", "dusty", "a dusty sound texture"),
    SemanticProbeDefinition("granular", "material", "granular", "a granular sound texture"),
    SemanticProbeDefinition("brittle", "material", "brittle", "a brittle sound texture"),
    SemanticProbeDefinition("smeared", "material", "smeared", "a smeared sound texture"),
    SemanticProbeDefinition("corroded", "material", "corroded", "a corroded sound texture"),
    SemanticProbeDefinition("scraping", "gesture", "scraping", "a scraping sound gesture"),
    SemanticProbeDefinition("ticking", "gesture", "ticking", "a ticking sound gesture"),
    SemanticProbeDefinition("knocking", "gesture", "knocking", "a knocking sound gesture"),
    SemanticProbeDefinition("collapsing", "gesture", "collapsing", "a collapsing sound gesture"),
    SemanticProbeDefinition("stuttering", "gesture", "stuttering", "a stuttering sound gesture"),
    SemanticProbeDefinition("rolling", "gesture", "rolling", "a rolling sound gesture"),
    SemanticProbeDefinition("bursting", "gesture", "bursting", "a bursting sound gesture"),
    SemanticProbeDefinition("swelling", "gesture", "swelling", "a swelling sound gesture"),
    SemanticProbeDefinition("grinding", "gesture", "grinding", "a grinding sound gesture"),
    SemanticProbeDefinition("sparse", "temporal_form", "sparse", "a sparse temporal form"),
    SemanticProbeDefinition("saturated", "temporal_form", "saturated", "a saturated temporal form"),
    SemanticProbeDefinition("limping", "temporal_form", "limping", "a limping temporal form"),
    SemanticProbeDefinition("arrhythmic", "temporal_form", "arrhythmic", "an arrhythmic temporal form"),
    SemanticProbeDefinition("drifting", "temporal_form", "drifting", "a drifting temporal form"),
    SemanticProbeDefinition("machine_like", "temporal_form", "machine-like", "a machine-like temporal form"),
    SemanticProbeDefinition("unstable", "temporal_form", "unstable", "an unstable temporal form"),
    SemanticProbeDefinition("suspended", "temporal_form", "suspended", "a suspended temporal form"),
    SemanticProbeDefinition("fragmentary", "temporal_form", "fragmentary", "a fragmentary temporal form"),
    SemanticProbeDefinition("metallic_scrape", "composite", "metallic scrape", "a metallic scraping sound"),
    SemanticProbeDefinition("granular_cloud", "composite", "granular cloud", "a cloud of granular noisy sound"),
    SemanticProbeDefinition("unstable_pulse", "composite", "unstable pulse", "an unstable pulsing arrhythmic sound"),
    SemanticProbeDefinition("brittle_ticking", "composite", "brittle ticking", "brittle ticking sounds"),
    SemanticProbeDefinition("smeared_swelling", "composite", "smeared swelling", "a smeared swelling texture"),
    SemanticProbeDefinition("corroded_grinding", "composite", "corroded grinding", "a corroded grinding texture"),
    SemanticProbeDefinition("clear_melody", "negative_control", "clear melody", "clear melody"),
    SemanticProbeDefinition("stable_beat", "negative_control", "stable beat", "stable beat"),
    SemanticProbeDefinition("sung_voice", "negative_control", "sung voice", "sung voice"),
    SemanticProbeDefinition("piano_chords", "negative_control", "piano chords", "piano chords"),
    SemanticProbeDefinition("acoustic_guitar", "negative_control", "acoustic guitar", "acoustic guitar"),
    SemanticProbeDefinition("pop_song", "negative_control", "pop song", "pop song"),
    SemanticProbeDefinition("dance_groove", "negative_control", "dance groove", "dance groove"),
)


class MsClapSemanticProbeBackend:
    name = "msclap"

    def __init__(
        self,
        *,
        model_version: str = DEFAULT_SEMANTIC_MODEL_VERSION,
        device: str = DEFAULT_SEMANTIC_DEVICE,
        model_path: str | Path | None = None,
    ) -> None:
        self.version = model_version
        self.device = _resolve_device(device)
        self.model_path = str(Path(model_path).expanduser().resolve()) if model_path is not None else None
        try:
            from msclap import CLAP
        except Exception as exc:  # pragma: no cover - exercised through public factory
            raise SemanticBackendUnavailable(f"msclap is not available. {SEMANTIC_INSTALL_HINT}") from exc

        kwargs = {"version": model_version, "use_cuda": self.device == "cuda"}
        if self.model_path is not None:
            kwargs["model_fp"] = self.model_path
        try:
            self._model = CLAP(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on external model runtime
            raise SemanticBackendUnavailable(f"msclap could not initialize: {exc}. {SEMANTIC_INSTALL_HINT}") from exc
        _patch_msclap_audio_loader(self._model)
        _patch_msclap_tokenizer_compat(self._model)
        self._text_embedding_cache: dict[tuple[str, ...], object] = {}

    def score_audio_text(self, audio_path: str | Path, prompts: Sequence[str]) -> Sequence[float]:
        path = str(Path(audio_path).expanduser().resolve())
        try:
            prompt_key = tuple(prompts)
            text_embeddings = self._text_embedding_cache.get(prompt_key)
            if text_embeddings is None:
                text_embeddings = self._model.get_text_embeddings(list(prompts))
                self._text_embedding_cache[prompt_key] = text_embeddings
            try:
                audio_embeddings = self._model.get_audio_embeddings([path], resample=True)
            except TypeError:
                audio_embeddings = self._model.get_audio_embeddings([path])
            similarities = self._model.compute_similarity(audio_embeddings, text_embeddings)
        except Exception as exc:
            raise SemanticBackendUnavailable(f"msclap scoring failed for {path}: {exc}") from exc
        return _tensor_to_scores(similarities)


def create_semantic_probe_backend(
    *,
    backend: str = DEFAULT_SEMANTIC_BACKEND,
    model_version: str = DEFAULT_SEMANTIC_MODEL_VERSION,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    model_path: str | Path | None = None,
) -> SemanticProbeBackend:
    if backend != "msclap":
        raise SemanticBackendUnavailable(f"Unsupported semantic backend: {backend}")
    return MsClapSemanticProbeBackend(model_version=model_version, device=device, model_path=model_path)


def run_semantic_probes(
    audio_path: str | Path,
    *,
    backend: str = DEFAULT_SEMANTIC_BACKEND,
    model_version: str = DEFAULT_SEMANTIC_MODEL_VERSION,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    model_path: str | Path | None = None,
    probe_backend: SemanticProbeBackend | None = None,
    definitions: Sequence[SemanticProbeDefinition] = DEFAULT_SEMANTIC_PROBES,
) -> SemanticProbeReport:
    resolved_backend = probe_backend or create_semantic_probe_backend(
        backend=backend,
        model_version=model_version,
        device=device,
        model_path=model_path,
    )
    prompts = [definition.prompt for definition in definitions]
    raw_scores = [float(score) for score in resolved_backend.score_audio_text(audio_path, prompts)]
    if len(raw_scores) != len(definitions):
        raise ValueError(f"semantic backend returned {len(raw_scores)} scores for {len(definitions)} prompts")

    normalized = _normalize_scores(raw_scores)
    scores = [
        SemanticProbeScore(
            key=definition.key,
            group=definition.group,
            label=definition.label,
            prompt=definition.prompt,
            raw_similarity=raw,
            score=score,
        )
        for definition, raw, score in zip(definitions, raw_scores, normalized, strict=True)
    ]
    strong = sorted(
        [score for score in scores if score.group != "negative_control" and score.score >= STRONG_THRESHOLD],
        key=lambda score: (-score.score, score.key),
    )[:MAX_FINDINGS]
    weak = sorted(
        [score for score in scores if score.group == "negative_control" and score.score <= WEAK_NEGATIVE_THRESHOLD],
        key=lambda score: (score.score, score.key),
    )[:MAX_FINDINGS]
    uncertain = [
        replace(score, reason="negative control scored high; verify against deterministic dossier evidence")
        for score in sorted(
            [
                score
                for score in scores
                if score.group == "negative_control" and score.score >= UNCERTAIN_NEGATIVE_THRESHOLD
            ],
            key=lambda score: (-score.score, score.key),
        )[:MAX_FINDINGS]
    ]
    return SemanticProbeReport(
        backend=SemanticBackendInfo(
            name=getattr(resolved_backend, "name", backend),
            version=getattr(resolved_backend, "version", model_version),
            device=getattr(resolved_backend, "device", device),
            model_path=getattr(resolved_backend, "model_path", None),
        ),
        vocabulary_version=VOCABULARY_VERSION,
        scores=scores,
        strong=strong,
        weak=weak,
        uncertain=uncertain,
    )


def attach_semantic_probes(
    dossier: ListeningDossier,
    *,
    audio_path: str | Path | None = None,
    backend: str = DEFAULT_SEMANTIC_BACKEND,
    model_version: str = DEFAULT_SEMANTIC_MODEL_VERSION,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    model_path: str | Path | None = None,
    probe_backend: SemanticProbeBackend | None = None,
) -> ListeningDossier:
    semantic_probes = run_semantic_probes(
        audio_path or dossier.audio.path,
        backend=backend,
        model_version=model_version,
        device=device,
        model_path=model_path,
        probe_backend=probe_backend,
    )
    return replace(dossier, semantic_probes=semantic_probes)


def semantic_summary_from_payload(payload: dict[str, object]) -> dict[str, object] | None:
    semantic = payload.get("semantic_probes")
    if not isinstance(semantic, dict):
        return None
    backend = semantic.get("backend") if isinstance(semantic.get("backend"), dict) else {}
    return {
        "backend": backend.get("name") if isinstance(backend, dict) else None,
        "backend_version": backend.get("version") if isinstance(backend, dict) else None,
        "vocabulary_version": semantic.get("vocabulary_version"),
        "strong": _finding_keys(semantic.get("strong")),
        "weak": _finding_keys(semantic.get("weak")),
        "uncertain": _finding_keys(semantic.get("uncertain")),
    }


def _finding_keys(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    keys: list[str] = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("key"), str):
            keys.append(item["key"])
    return keys


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device not in {"cpu", "cuda"}:
        raise SemanticBackendUnavailable(f"Unsupported semantic device: {device}")
    return device


def _normalize_scores(raw_scores: Sequence[float]) -> list[float]:
    if not raw_scores:
        return []
    values = np.asarray(raw_scores, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        return [0.5 for _ in raw_scores]
    return [float((value - min_value) / (max_value - min_value)) for value in values]


def _tensor_to_scores(value: object) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return [float(array)]
    if array.ndim > 1:
        array = array[0]
    return [float(score) for score in array.reshape(-1).tolist()]


def _patch_msclap_audio_loader(model: object) -> None:
    def read_audio(self: object, audio_path: str | Path, resample: bool = True) -> tuple[object, int]:
        import librosa
        import torch

        target_sr = int(self.args.sampling_rate)
        loaded_sr = target_sr if resample else None
        waveform, sr = librosa.load(audio_path, sr=loaded_sr, mono=True)
        return torch.from_numpy(waveform).reshape(1, -1).float(), int(sr)

    if hasattr(model, "read_audio"):
        model.read_audio = MethodType(read_audio, model)


def _patch_msclap_tokenizer_compat(model: object) -> None:
    # transformers>=5 removed the legacy encode_plus API msclap still calls.
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or hasattr(tokenizer, "encode_plus"):
        return

    def encode_plus(text: str, **kwargs: object) -> object:
        return tokenizer(text=text, **kwargs)

    tokenizer.encode_plus = encode_plus
