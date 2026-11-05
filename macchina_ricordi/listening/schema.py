from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "macchina_ricordi.listening_dossier.v0"
ANALYSIS_VERSION = "0.4.0"


@dataclass(frozen=True)
class AnalysisParameters:
    sr: int
    hop_length: int
    res_type: str
    tempo_method: str
    window_seconds: float
    window_hop_seconds: float


@dataclass(frozen=True)
class AnalysisInfo:
    version: str
    analyzed_at: str
    parameters: AnalysisParameters


@dataclass(frozen=True)
class AudioInfo:
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    mono: bool


@dataclass(frozen=True)
class ProvenanceInfo:
    source: str
    audio_path: str


@dataclass(frozen=True)
class WholeClip:
    reward: dict[str, Any]
    description: dict[str, str]
    features: dict[str, float]


@dataclass(frozen=True)
class TimelineWindow:
    start_seconds: float
    end_seconds: float
    label: str
    features: dict[str, float]


@dataclass(frozen=True)
class TimelineRegion:
    start_seconds: float
    end_seconds: float
    label: str
    evidence: dict[str, str]
    window_indices: list[int]


@dataclass(frozen=True)
class Timeline:
    window_seconds: float
    window_hop_seconds: float
    windows: list[TimelineWindow]
    regions: list[TimelineRegion]


@dataclass(frozen=True)
class Sketches:
    spectral: str
    mel_spectrogram_path: str | None = None
    numeric_spectrogram: str | None = None


@dataclass(frozen=True)
class NotableEvent:
    time_seconds: float
    end_seconds: float | None
    kind: str
    strength: float
    detail: str


@dataclass(frozen=True)
class ListeningSummary:
    global_summary: str
    time_contour: str


@dataclass(frozen=True)
class SemanticBackendInfo:
    name: str
    version: str
    device: str
    model_path: str | None = None


@dataclass(frozen=True)
class SemanticProbeScore:
    key: str
    group: str
    label: str
    prompt: str
    raw_similarity: float
    score: float
    reason: str | None = None


@dataclass(frozen=True)
class SemanticProbeReport:
    backend: SemanticBackendInfo
    vocabulary_version: str
    scores: list[SemanticProbeScore]
    strong: list[SemanticProbeScore]
    weak: list[SemanticProbeScore]
    uncertain: list[SemanticProbeScore]


@dataclass(frozen=True)
class CaptionerBackendInfo:
    name: str
    model_id: str
    device: str
    dtype: str
    quantization: str = "none"
    trust_remote_code: bool = True


@dataclass(frozen=True)
class CaptionerWitness:
    backend: CaptionerBackendInfo
    prompt_profile: str
    prompt: str
    summary: str
    raw_response: str
    warnings: list[str]
    agreement: dict[str, str]
    claims: dict[str, bool]


@dataclass(frozen=True)
class ListeningDossier:
    schema: str
    analysis: AnalysisInfo
    audio: AudioInfo
    provenance: ProvenanceInfo
    whole_clip: WholeClip
    timeline: Timeline
    sketches: Sketches
    summary: ListeningSummary
    warnings: list[str]
    notable_events: list[NotableEvent] = field(default_factory=list)
    semantic_probes: SemanticProbeReport | None = None
    captioner_witness: CaptionerWitness | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        captioner_witness = payload.pop("captioner_witness")
        semantic_probes = payload.pop("semantic_probes")
        warnings = payload.pop("warnings")
        notable_events = payload.pop("notable_events")
        payload["notable_events"] = notable_events
        if semantic_probes is not None:
            payload["semantic_probes"] = semantic_probes
        if captioner_witness is not None:
            payload["captioner_witness"] = captioner_witness
        payload["warnings"] = warnings
        return payload
