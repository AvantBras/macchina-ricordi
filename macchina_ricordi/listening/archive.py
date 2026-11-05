from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np

from ..audio_files import DEFAULT_EXTENSIONS, iter_audio_files, parse_extensions
from ..scoring.similarity import feature_vectors, similarity_from_vectors
from .captioner import (
    DEFAULT_CAPTIONER_BACKEND,
    DEFAULT_CAPTIONER_DEVICE,
    DEFAULT_CAPTIONER_DTYPE,
    DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS,
    DEFAULT_CAPTIONER_MAX_NEW_TOKENS,
    DEFAULT_CAPTIONER_MODEL_ID,
    DEFAULT_CAPTIONER_PROMPT_PROFILE,
    DEFAULT_CAPTIONER_QUANTIZATION,
    DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    CaptionerBackend,
    attach_captioner_witness,
    captioner_summary_from_payload,
    create_captioner_backend,
)
from .features import analyze_waveform
from .render import (
    default_json_report_path,
    default_markdown_report_path,
    render_dossier_json,
    render_dossier_markdown,
)
from .semantic import (
    DEFAULT_SEMANTIC_BACKEND,
    DEFAULT_SEMANTIC_DEVICE,
    DEFAULT_SEMANTIC_MODEL_VERSION,
    SemanticProbeBackend,
    attach_semantic_probes,
    create_semantic_probe_backend,
    semantic_summary_from_payload,
)

ARCHIVE_SCHEMA_VERSION = "macchina_ricordi.listening_archive.v0"
ARCHIVE_ANALYSIS_VERSION = "0.1.0"


@dataclass(frozen=True)
class ArchiveIndexParameters:
    sr: int
    hop_length: int
    res_type: str
    tempo_method: str
    window_seconds: float
    window_hop_seconds: float
    recursive: bool
    extensions: tuple[str, ...]
    neighbors: int
    read_only: bool
    refresh: bool
    semantic_probes: bool = False
    semantic_backend: str = DEFAULT_SEMANTIC_BACKEND
    semantic_model_version: str = DEFAULT_SEMANTIC_MODEL_VERSION
    semantic_device: str = DEFAULT_SEMANTIC_DEVICE
    semantic_model_path: str | None = None
    captioner_witness: bool = False
    captioner_backend: str = DEFAULT_CAPTIONER_BACKEND
    captioner_model_id: str = DEFAULT_CAPTIONER_MODEL_ID
    captioner_device: str = DEFAULT_CAPTIONER_DEVICE
    captioner_dtype: str = DEFAULT_CAPTIONER_DTYPE
    captioner_quantization: str = DEFAULT_CAPTIONER_QUANTIZATION
    captioner_max_new_tokens: int = DEFAULT_CAPTIONER_MAX_NEW_TOKENS
    captioner_max_audio_seconds: float = DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS
    captioner_prompt_profile: str = DEFAULT_CAPTIONER_PROMPT_PROFILE
    captioner_trust_remote_code: bool = DEFAULT_CAPTIONER_TRUST_REMOTE_CODE


@dataclass(frozen=True)
class ArchiveAnalysisInfo:
    version: str
    indexed_at: str
    parameters: ArchiveIndexParameters


@dataclass(frozen=True)
class ArchiveNeighbor:
    path: str
    similarity: float
    mfcc_cosine: float
    spectral_cosine: float
    rhythm_cosine: float


@dataclass(frozen=True)
class ArchiveEntry:
    path: str
    duration_seconds: float | None
    report_path: str | None
    markdown_path: str | None
    summary: str
    labels: list[str]
    reward_score: float | None
    vectors: dict[str, list[float]]
    neighbors: list[ArchiveNeighbor]
    novelty_score: float | None
    semantic: dict[str, Any] | None = None
    captioner: dict[str, Any] | None = None
    musicgen: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class ListeningArchiveIndex:
    schema: str
    analysis: ArchiveAnalysisInfo
    root: str
    entries: list[ArchiveEntry]
    groups: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchivePortrait:
    ref: str
    generated_root: str
    index: ListeningArchiveIndex
    summary: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_archive_index(
    root: str | Path,
    *,
    output_path: str | Path | None = None,
    audio_paths: list[str | Path] | None = None,
    sr: int = 32_000,
    hop_length: int = 512,
    res_type: str = "soxr_hq",
    tempo_method: Literal["autocorr", "tempogram"] = "tempogram",
    window_seconds: float = 2.0,
    window_hop_seconds: float = 1.0,
    recursive: bool = True,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    neighbors: int = 3,
    read_only: bool = False,
    refresh: bool = False,
    repo_root: str | Path | None = None,
    semantic_probes: bool = False,
    semantic_backend: str = DEFAULT_SEMANTIC_BACKEND,
    semantic_model_version: str = DEFAULT_SEMANTIC_MODEL_VERSION,
    semantic_device: str = DEFAULT_SEMANTIC_DEVICE,
    semantic_model_path: str | Path | None = None,
    semantic_probe_backend: SemanticProbeBackend | None = None,
    captioner_witness: bool = False,
    captioner_backend: str = DEFAULT_CAPTIONER_BACKEND,
    captioner_model_id: str = DEFAULT_CAPTIONER_MODEL_ID,
    captioner_device: str = DEFAULT_CAPTIONER_DEVICE,
    captioner_dtype: str = DEFAULT_CAPTIONER_DTYPE,
    captioner_quantization: str = DEFAULT_CAPTIONER_QUANTIZATION,
    captioner_max_new_tokens: int = DEFAULT_CAPTIONER_MAX_NEW_TOKENS,
    captioner_max_audio_seconds: float = DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS,
    captioner_prompt_profile: str = DEFAULT_CAPTIONER_PROMPT_PROFILE,
    captioner_trust_remote_code: bool = DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    captioner_backend_instance: CaptionerBackend | None = None,
) -> ListeningArchiveIndex:
    resolved_root = Path(root).expanduser().resolve()
    resolved_repo = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
    params = ArchiveIndexParameters(
        sr=sr,
        hop_length=hop_length,
        res_type=res_type,
        tempo_method=tempo_method,
        window_seconds=window_seconds,
        window_hop_seconds=window_hop_seconds,
        recursive=recursive,
        extensions=tuple(extensions),
        neighbors=max(0, int(neighbors)),
        read_only=read_only,
        refresh=refresh,
        semantic_probes=semantic_probes,
        semantic_backend=semantic_backend,
        semantic_model_version=semantic_model_version,
        semantic_device=semantic_device,
        semantic_model_path=str(Path(semantic_model_path).expanduser().resolve()) if semantic_model_path else None,
        captioner_witness=captioner_witness,
        captioner_backend=captioner_backend,
        captioner_model_id=captioner_model_id,
        captioner_device=captioner_device,
        captioner_dtype=captioner_dtype,
        captioner_quantization=captioner_quantization,
        captioner_max_new_tokens=captioner_max_new_tokens,
        captioner_max_audio_seconds=captioner_max_audio_seconds,
        captioner_prompt_profile=captioner_prompt_profile,
        captioner_trust_remote_code=captioner_trust_remote_code,
    )
    warnings: list[str] = []
    metadata_by_path = scan_musicgen_audio_metadata(resolved_repo)
    paths = _archive_audio_paths(resolved_root, audio_paths=audio_paths, recursive=recursive, extensions=extensions)
    resolved_probe_backend = semantic_probe_backend
    if params.semantic_probes and not params.read_only and resolved_probe_backend is None:
        resolved_probe_backend = create_semantic_probe_backend(
            backend=params.semantic_backend,
            model_version=params.semantic_model_version,
            device=params.semantic_device,
            model_path=params.semantic_model_path,
        )
    resolved_captioner_backend = captioner_backend_instance
    if params.captioner_witness and not params.read_only and resolved_captioner_backend is None:
        resolved_captioner_backend = create_captioner_backend(
            backend=params.captioner_backend,
            model_id=params.captioner_model_id,
            device=params.captioner_device,
            dtype=params.captioner_dtype,
            quantization=params.captioner_quantization,
            trust_remote_code=params.captioner_trust_remote_code,
        )
    entries: list[ArchiveEntry] = []
    for path in paths:
        entry = _build_entry(
            path,
            params=params,
            metadata=metadata_by_path.get(str(path)),
            warnings=warnings,
            semantic_probe_backend=resolved_probe_backend,
            captioner_backend=resolved_captioner_backend,
        )
        if entry is not None:
            entries.append(entry)
    entries = _attach_neighbors(entries, neighbor_count=params.neighbors)
    index = ListeningArchiveIndex(
        schema=ARCHIVE_SCHEMA_VERSION,
        analysis=ArchiveAnalysisInfo(
            version=ARCHIVE_ANALYSIS_VERSION,
            indexed_at=datetime.now(UTC).isoformat(),
            parameters=params,
        ),
        root=str(resolved_root),
        entries=entries,
        groups=_build_groups(entries),
        warnings=warnings,
    )
    if output_path is not None:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_archive_index_json(index), encoding="utf-8")
    return index


def load_archive_index(path: str | Path) -> ListeningArchiveIndex:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    raw_params = dict(payload["analysis"]["parameters"])
    raw_params["extensions"] = tuple(raw_params.get("extensions", ()))
    params = ArchiveIndexParameters(**raw_params)
    analysis = ArchiveAnalysisInfo(
        version=payload["analysis"]["version"],
        indexed_at=payload["analysis"]["indexed_at"],
        parameters=params,
    )
    entries = [
        ArchiveEntry(
            path=entry["path"],
            duration_seconds=entry.get("duration_seconds"),
            report_path=entry.get("report_path"),
            markdown_path=entry.get("markdown_path"),
            summary=entry.get("summary", ""),
            labels=list(entry.get("labels", [])),
            reward_score=entry.get("reward_score"),
            vectors=dict(entry.get("vectors", {})),
            neighbors=[
                ArchiveNeighbor(
                    path=neighbor["path"],
                    similarity=float(neighbor.get("similarity", 0.0)),
                    mfcc_cosine=float(neighbor.get("mfcc_cosine", 0.0)),
                    spectral_cosine=float(neighbor.get("spectral_cosine", 0.0)),
                    rhythm_cosine=float(neighbor.get("rhythm_cosine", 0.0)),
                )
                for neighbor in entry.get("neighbors", [])
            ],
            novelty_score=entry.get("novelty_score"),
            semantic=entry.get("semantic"),
            captioner=entry.get("captioner"),
            musicgen=entry.get("musicgen"),
            error=entry.get("error"),
        )
        for entry in payload.get("entries", [])
    ]
    return ListeningArchiveIndex(
        schema=payload.get("schema", ARCHIVE_SCHEMA_VERSION),
        analysis=analysis,
        root=payload.get("root", ""),
        entries=entries,
        groups=list(payload.get("groups", [])),
        warnings=list(payload.get("warnings", [])),
    )


def build_similar_report(
    audio_path: str | Path,
    root: str | Path,
    *,
    index: ListeningArchiveIndex | None = None,
    neighbors: int = 5,
    **index_kwargs: Any,
) -> dict[str, Any]:
    resolved_audio = Path(audio_path).expanduser().resolve()
    archive_index = index or build_archive_index(root, neighbors=max(neighbors, 1), **index_kwargs)
    entry = _entry_for_path(archive_index, resolved_audio)
    if entry is None:
        temp_index = build_archive_index(
            root,
            audio_paths=[resolved_audio],
            neighbors=0,
            **{**index_kwargs, "read_only": False},
        )
        if not temp_index.entries:
            return {"target": str(resolved_audio), "neighbors": [], "error": "target audio could not be analyzed"}
        entry = temp_index.entries[0]
        neighbor_rows = [
            _entry_similarity(entry, candidate)
            for candidate in archive_index.entries
            if candidate.error is None and candidate.vectors
        ]
        neighbor_rows.sort(key=lambda row: (-row.similarity, row.path))
        neighbors_list = neighbor_rows[: max(0, neighbors)]
    else:
        neighbors_list = entry.neighbors[: max(0, neighbors)]
    return {
        "target": entry.path,
        "summary": entry.summary,
        "labels": entry.labels,
        "semantic": entry.semantic,
        "captioner": entry.captioner,
        "neighbors": [asdict(neighbor) for neighbor in neighbors_list],
    }


def build_novelty_report(index: ListeningArchiveIndex, *, limit: int = 20) -> dict[str, Any]:
    rows = [
        {
            "path": entry.path,
            "novelty_score": entry.novelty_score,
            "summary": entry.summary,
            "labels": entry.labels,
            "reward_score": entry.reward_score,
            "semantic": entry.semantic,
            "captioner": entry.captioner,
        }
        for entry in index.entries
        if entry.error is None and entry.novelty_score is not None
    ]
    rows.sort(key=lambda row: (-(row["novelty_score"] or 0.0), row["path"]))
    return {"root": index.root, "entries": rows[: max(0, limit)]}


def build_archive_portrait(
    ref: str,
    *,
    generated_root: str | Path = "artifacts/runs",
    repo_root: str | Path | None = None,
    **index_kwargs: Any,
) -> ArchivePortrait:
    resolved_repo = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
    resolved_generated = Path(generated_root).expanduser()
    if not resolved_generated.is_absolute():
        resolved_generated = resolved_repo / resolved_generated
    resolved_generated = resolved_generated.resolve()
    metadata = scan_musicgen_audio_metadata(resolved_repo)
    paths = [Path(path) for path, info in metadata.items() if _metadata_matches_ref(info, ref) and Path(path).exists()]
    warnings: list[str] = []
    if not paths:
        warnings.append(f"No generated audio matched {ref!r}.")
    index = build_archive_index(
        resolved_generated,
        audio_paths=paths,
        repo_root=resolved_repo,
        neighbors=int(index_kwargs.pop("neighbors", 3)),
        **index_kwargs,
    )
    return ArchivePortrait(
        ref=ref,
        generated_root=str(resolved_generated),
        index=index,
        summary=_portrait_summary(index),
        warnings=[*warnings, *index.warnings],
    )


def render_archive_index_json(index: ListeningArchiveIndex) -> str:
    return json.dumps(index.to_dict(), indent=2)


def render_similar_markdown(report: dict[str, Any]) -> str:
    lines = ["# Similar Sounds", "", f"- target: {report.get('target', '')}"]
    if report.get("summary"):
        lines.append(f"- summary: {report['summary']}")
    if report.get("labels"):
        lines.append(f"- labels: {', '.join(report['labels'])}")
    semantic = report.get("semantic")
    if isinstance(semantic, dict) and semantic.get("strong"):
        lines.append(f"- semantic_strong: {', '.join(semantic['strong'])}")
    captioner = report.get("captioner")
    if isinstance(captioner, dict) and captioner.get("summary"):
        lines.append(f"- captioner_summary: {captioner['summary']}")
    lines.extend(["", "## Neighbors", ""])
    neighbors = report.get("neighbors", [])
    if not neighbors:
        lines.append("- none")
    for neighbor in neighbors:
        lines.append(
            f"- {neighbor['similarity']:.3f}: {neighbor['path']} "
            f"(mfcc={neighbor['mfcc_cosine']:.3f}, spectral={neighbor['spectral_cosine']:.3f}, "
            f"rhythm={neighbor['rhythm_cosine']:.3f})"
        )
    return "\n".join(lines) + "\n"


def render_novelty_markdown(report: dict[str, Any]) -> str:
    lines = ["# Archive Novelty", "", f"- root: {report.get('root', '')}", "", "## Most Novel", ""]
    entries = report.get("entries", [])
    if not entries:
        lines.append("- none")
    for entry in entries:
        labels = ", ".join(entry.get("labels", []))
        lines.append(f"- {entry['novelty_score']:.3f}: {entry['path']}")
        if labels:
            lines.append(f"  labels: {labels}")
        if entry.get("summary"):
            lines.append(f"  summary: {entry['summary']}")
        semantic = entry.get("semantic")
        if isinstance(semantic, dict) and semantic.get("strong"):
            lines.append(f"  semantic_strong: {', '.join(semantic['strong'])}")
        captioner = entry.get("captioner")
        if isinstance(captioner, dict) and captioner.get("summary"):
            lines.append(f"  captioner_summary: {captioner['summary']}")
    return "\n".join(lines) + "\n"


def render_portrait_markdown(portrait: ArchivePortrait) -> str:
    summary = portrait.summary
    lines = [
        "# Listening Portrait",
        "",
        f"- ref: {portrait.ref}",
        f"- generated_root: {portrait.generated_root}",
        f"- samples: {summary.get('sample_count', 0)}",
    ]
    if summary.get("score_mean") is not None:
        lines.append(f"- score_mean: {summary['score_mean']:.3f}")
    if summary.get("common_labels"):
        labels = ", ".join(f"{label} ({count})" for label, count in summary["common_labels"])
        lines.append(f"- common_labels: {labels}")
    lines.extend(["", "## Representative Samples", ""])
    for entry in summary.get("representative_samples", []):
        lines.append(f"- {entry['path']}")
    if not summary.get("representative_samples"):
        lines.append("- none")
    lines.extend(["", "## Outliers", ""])
    for entry in summary.get("outliers", []):
        lines.append(f"- {entry['novelty_score']:.3f}: {entry['path']}")
    if not summary.get("outliers"):
        lines.append("- none")
    if summary.get("parameter_ranges"):
        lines.extend(["", "## Parameters", ""])
        for key, value in summary["parameter_ranges"].items():
            lines.append(f"- {key}: {value}")
    if portrait.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in portrait.warnings)
    return "\n".join(lines) + "\n"


def scan_musicgen_audio_metadata(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    from macchina_ricordi.instrument.run_paths import (
        iter_generation_metadata_paths,
        iter_sweep_index_paths,
        resolve_metadata_file_path,
    )

    root = Path(repo_root).expanduser().resolve()
    metadata_by_path: dict[str, dict[str, Any]] = {}
    for metadata_path in iter_generation_metadata_paths(root):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = _generation_source(payload)
        model_info = payload.get("model") if isinstance(payload.get("model"), dict) else {}
        request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        for sample_index, file_value in enumerate(files):
            path = resolve_metadata_file_path(file_value, metadata_path=metadata_path)
            if path is None:
                continue
            metadata_by_path[str(path.resolve())] = {
                "run_metadata_path": str(metadata_path),
                "run_id": payload.get("run_id"),
                "sample_index": sample_index,
                "model_label": model_info.get("label") if isinstance(model_info, dict) else None,
                "model_workspace": model_info.get("workspace") if isinstance(model_info, dict) else None,
                "xp_path": source.get("xp_path"),
                "checkpoint_path": source.get("checkpoint_path"),
                "temperature": request.get("temperature"),
                "top_k": request.get("top_k"),
                "top_p": request.get("top_p"),
                "cfg_coef": request.get("cfg_coef"),
            }
    for csv_path in iter_sweep_index_paths(root):
        _scan_sweep_index(csv_path, metadata_by_path)
    return metadata_by_path


def _archive_audio_paths(
    root: Path,
    *,
    audio_paths: list[str | Path] | None,
    recursive: bool,
    extensions: tuple[str, ...],
) -> list[Path]:
    if audio_paths is not None:
        return sorted({Path(path).expanduser().resolve() for path in audio_paths})
    if root.is_file():
        return [root.resolve()]
    return sorted(path.resolve() for path in iter_audio_files(root, recursive=recursive, extensions=extensions))


def _build_entry(
    audio_path: Path,
    *,
    params: ArchiveIndexParameters,
    metadata: dict[str, Any] | None,
    warnings: list[str],
    semantic_probe_backend: SemanticProbeBackend | None,
    captioner_backend: CaptionerBackend | None,
) -> ArchiveEntry | None:
    report_path = default_json_report_path(audio_path).resolve()
    markdown_path = default_markdown_report_path(audio_path).resolve()
    try:
        dossier_payload, waveform, loaded_sr = _load_or_create_dossier(
            audio_path,
            report_path,
            markdown_path,
            params=params,
            warnings=warnings,
            semantic_probe_backend=semantic_probe_backend,
            captioner_backend=captioner_backend,
        )
        if waveform is None or loaded_sr is None:
            waveform, loaded_sr = librosa.load(audio_path, sr=params.sr, mono=True, res_type=params.res_type)
        vectors = _vectors_for_waveform(waveform, sr=int(loaded_sr))
    except FileNotFoundError as exc:
        warnings.append(f"{audio_path}: {exc}")
        return None
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        message = f"{audio_path}: {error}"
        warnings.append(message)
        return ArchiveEntry(
            path=str(audio_path),
            duration_seconds=None,
            report_path=str(report_path) if report_path.exists() else None,
            markdown_path=str(markdown_path) if markdown_path.exists() else None,
            summary="",
            labels=[],
            reward_score=None,
            vectors={},
            neighbors=[],
            novelty_score=None,
            semantic=None,
            captioner=None,
            musicgen=metadata,
            error=error,
        )
    return ArchiveEntry(
        path=str(audio_path),
        duration_seconds=_duration_from_payload(dossier_payload),
        report_path=str(report_path) if report_path.exists() else None,
        markdown_path=str(markdown_path) if markdown_path.exists() else None,
        summary=_summary_from_payload(dossier_payload),
        labels=_labels_from_payload(dossier_payload),
        reward_score=_score_from_payload(dossier_payload),
        vectors=vectors,
        neighbors=[],
        novelty_score=None,
        semantic=semantic_summary_from_payload(dossier_payload),
        captioner=captioner_summary_from_payload(dossier_payload),
        musicgen=metadata,
    )


def _load_or_create_dossier(
    audio_path: Path,
    report_path: Path,
    markdown_path: Path,
    *,
    params: ArchiveIndexParameters,
    warnings: list[str],
    semantic_probe_backend: SemanticProbeBackend | None,
    captioner_backend: CaptionerBackend | None,
) -> tuple[dict[str, Any], np.ndarray | None, int | None]:
    if report_path.exists() and not params.refresh:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if params.read_only:
                raise ValueError(f"unreadable listening sidecar: {report_path}") from exc
            warnings.append(f"{audio_path}: re-analyzing unreadable sidecar {report_path}")
        else:
            missing_optional: list[str] = []
            if params.semantic_probes and "semantic_probes" not in payload:
                missing_optional.append("semantic probes")
            if params.captioner_witness and "captioner_witness" not in payload:
                missing_optional.append("captioner witness")
            if not missing_optional:
                return payload, None, None
            if params.read_only:
                warnings.append(
                    f"{audio_path}: listening sidecar has no {', '.join(missing_optional)} in read-only mode"
                )
                return payload, None, None
            warnings.append(f"{audio_path}: enriching listening sidecar with {', '.join(missing_optional)}")
    elif params.read_only:
        raise FileNotFoundError(f"missing listening sidecar: {report_path}")

    return _analyze_and_write_dossier(
        audio_path,
        report_path,
        markdown_path,
        params=params,
        warnings=warnings,
        semantic_probe_backend=semantic_probe_backend,
        captioner_backend=captioner_backend,
    )


def _analyze_and_write_dossier(
    audio_path: Path,
    report_path: Path,
    markdown_path: Path,
    *,
    params: ArchiveIndexParameters,
    warnings: list[str],
    semantic_probe_backend: SemanticProbeBackend | None,
    captioner_backend: CaptionerBackend | None,
) -> tuple[dict[str, Any], np.ndarray, int]:
    waveform, loaded_sr = librosa.load(audio_path, sr=params.sr, mono=True, res_type=params.res_type)
    loaded_sr = int(loaded_sr)
    dossier = analyze_waveform(
        waveform,
        path=audio_path,
        sr=loaded_sr,
        hop_length=params.hop_length,
        res_type=params.res_type,
        tempo_method=params.tempo_method,
        window_seconds=params.window_seconds,
        window_hop_seconds=params.window_hop_seconds,
    )
    if params.semantic_probes:
        try:
            dossier = attach_semantic_probes(
                dossier,
                audio_path=audio_path,
                backend=params.semantic_backend,
                model_version=params.semantic_model_version,
                device=params.semantic_device,
                model_path=params.semantic_model_path,
                probe_backend=semantic_probe_backend,
            )
        except Exception as exc:
            warnings.append(f"{audio_path}: semantic probes failed: {exc}")
    if params.captioner_witness:
        try:
            dossier = attach_captioner_witness(
                dossier,
                audio_path=audio_path,
                backend=params.captioner_backend,
                model_id=params.captioner_model_id,
                device=params.captioner_device,
                dtype=params.captioner_dtype,
                quantization=params.captioner_quantization,
                max_new_tokens=params.captioner_max_new_tokens,
                max_audio_seconds=params.captioner_max_audio_seconds,
                prompt_profile=params.captioner_prompt_profile,
                trust_remote_code=params.captioner_trust_remote_code,
                captioner_backend=captioner_backend,
            )
        except Exception as exc:
            warnings.append(f"{audio_path}: captioner witness failed: {exc}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_dossier_json(dossier), encoding="utf-8")
    markdown_path.write_text(render_dossier_markdown(dossier), encoding="utf-8")
    return dossier.to_dict(), waveform, loaded_sr


def _vectors_for_waveform(waveform: np.ndarray, *, sr: int) -> dict[str, list[float]]:
    vectors = feature_vectors(waveform, sr=sr)
    return {key: np.nan_to_num(vector.astype(float)).tolist() for key, vector in vectors.items()}


def _entry_similarity(a: ArchiveEntry, b: ArchiveEntry) -> ArchiveNeighbor:
    components = similarity_from_vectors(a.vectors, b.vectors)
    return ArchiveNeighbor(
        path=b.path,
        similarity=components.combined,
        mfcc_cosine=components.mfcc_cosine,
        spectral_cosine=components.spectral_cosine,
        rhythm_cosine=components.rhythm_cosine,
    )


def _attach_neighbors(entries: list[ArchiveEntry], *, neighbor_count: int) -> list[ArchiveEntry]:
    from dataclasses import replace

    valid = [entry for entry in entries if entry.error is None and entry.vectors]
    if len(valid) <= 1:
        return [
            replace(entry, neighbors=[], novelty_score=1.0 if entry.error is None and entry.vectors else None)
            for entry in entries
        ]

    updated: list[ArchiveEntry] = []
    for entry in entries:
        if entry.error is not None or not entry.vectors:
            updated.append(entry)
            continue
        neighbors = [_entry_similarity(entry, other) for other in valid if other.path != entry.path]
        neighbors.sort(key=lambda neighbor: (-neighbor.similarity, neighbor.path))
        clipped = neighbors[: max(0, neighbor_count)]
        best = neighbors[0].similarity if neighbors else 0.0
        updated.append(replace(entry, neighbors=clipped, novelty_score=float(1.0 - best)))
    return updated


def _build_groups(entries: list[ArchiveEntry]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ArchiveEntry]] = {}
    for entry in entries:
        if not entry.musicgen:
            continue
        key = str(
            entry.musicgen.get("xp_path") or entry.musicgen.get("checkpoint_path") or entry.musicgen.get("run_id") or ""
        )
        if not key:
            continue
        grouped.setdefault(key, []).append(entry)
    groups: list[dict[str, Any]] = []
    for key, group_entries in sorted(grouped.items()):
        scores = [entry.reward_score for entry in group_entries if entry.reward_score is not None]
        groups.append(
            {
                "key": key,
                "count": len(group_entries),
                "score_mean": sum(scores) / len(scores) if scores else None,
                "labels": _label_counts(group_entries),
            }
        )
    return groups


def _portrait_summary(index: ListeningArchiveIndex) -> dict[str, Any]:
    entries = [entry for entry in index.entries if entry.error is None]
    scores = [entry.reward_score for entry in entries if entry.reward_score is not None]
    representative = sorted(
        entries,
        key=lambda entry: (entry.novelty_score if entry.novelty_score is not None else 1.0, entry.path),
    )
    outliers = sorted(
        entries,
        key=lambda entry: (-(entry.novelty_score if entry.novelty_score is not None else -1.0), entry.path),
    )
    return {
        "sample_count": len(entries),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "common_labels": _label_counts(entries)[:10],
        "representative_samples": [_entry_brief(entry) for entry in representative[:5]],
        "outliers": [_entry_brief(entry) for entry in outliers[:5]],
        "parameter_ranges": _parameter_ranges(entries),
    }


def _entry_brief(entry: ArchiveEntry) -> dict[str, Any]:
    return {
        "path": entry.path,
        "summary": entry.summary,
        "labels": entry.labels,
        "reward_score": entry.reward_score,
        "novelty_score": entry.novelty_score,
        "semantic": entry.semantic,
        "captioner": entry.captioner,
    }


def _parameter_ranges(entries: list[ArchiveEntry]) -> dict[str, list[Any]]:
    values: dict[str, set[Any]] = {}
    for entry in entries:
        if not entry.musicgen:
            continue
        for key in ("temperature", "top_k", "top_p", "cfg_coef"):
            value = entry.musicgen.get(key)
            if value not in (None, ""):
                values.setdefault(key, set()).add(value)
    return {key: sorted(items) for key, items in sorted(values.items())}


def _label_counts(entries: list[ArchiveEntry]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for entry in entries:
        for label in entry.labels:
            counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _scan_sweep_index(csv_path: Path, metadata_by_path: dict[str, dict[str, Any]]) -> None:
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return
    for row in rows:
        path_value = row.get("path")
        if not path_value:
            continue
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = csv_path.parent / path
        metadata = metadata_by_path.setdefault(str(path.resolve()), {})
        metadata.update(
            {
                "sweep_index_path": str(csv_path),
                "sweep_id": row.get("sweep_id") or None,
                "combo_id": row.get("combo_id") or None,
                "combo_index": _int_or_none(row.get("combo_index")),
                "sample_index": _int_or_none(row.get("sample_index")),
                "xp_path": row.get("xp_path") or metadata.get("xp_path"),
                "checkpoint_path": row.get("checkpoint_path") or metadata.get("checkpoint_path"),
                "temperature": _float_or_none(row.get("temperature")),
                "top_k": _int_or_none(row.get("top_k")),
                "top_p": _float_or_none(row.get("top_p")),
                "cfg_coef": _float_or_none(row.get("cfg_coef")),
            }
        )


def _generation_source(payload: dict[str, Any]) -> dict[str, Any]:
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    source = generation.get("source") if isinstance(generation, dict) else {}
    return source if isinstance(source, dict) else {}


def _metadata_matches_ref(metadata: dict[str, Any], ref: str) -> bool:
    needle = ref.strip().lower()
    if not needle:
        return False
    for key in ("xp_path", "checkpoint_path", "model_label", "model_workspace", "run_id", "sweep_id"):
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).lower()
        if needle == text or needle in text or needle == Path(str(value)).name.lower():
            return True
    return False


def _entry_for_path(index: ListeningArchiveIndex, path: Path) -> ArchiveEntry | None:
    resolved = str(path.resolve())
    for entry in index.entries:
        if entry.path == resolved:
            return entry
    return None


def _summary_from_payload(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("global_summary"), str):
        return summary["global_summary"]
    return ""


def _duration_from_payload(payload: dict[str, Any]) -> float | None:
    audio = payload.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("duration_seconds"), int | float):
        return float(audio["duration_seconds"])
    return None


def _labels_from_payload(payload: dict[str, Any]) -> list[str]:
    whole_clip = payload.get("whole_clip")
    description = whole_clip.get("description") if isinstance(whole_clip, dict) else None
    if not isinstance(description, dict):
        return []
    labels: list[str] = []
    for key in ("loudness", "brightness", "density", "pulse", "noisiness", "attack", "resonance"):
        value = description.get(key)
        if isinstance(value, str) and value:
            labels.append(value)
    return list(dict.fromkeys(labels))


def _score_from_payload(payload: dict[str, Any]) -> float | None:
    whole_clip = payload.get("whole_clip")
    reward = whole_clip.get("reward") if isinstance(whole_clip, dict) else None
    if isinstance(reward, dict) and isinstance(reward.get("score"), int | float):
        return float(reward["score"])
    return None


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def archive_extensions(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    return parse_extensions(value)
