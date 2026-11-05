from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .captioner import (
    DEFAULT_CAPTIONER_BACKEND,
    DEFAULT_CAPTIONER_DEVICE,
    DEFAULT_CAPTIONER_DTYPE,
    DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS,
    DEFAULT_CAPTIONER_MAX_NEW_TOKENS,
    DEFAULT_CAPTIONER_MODEL_ID,
    DEFAULT_CAPTIONER_QUANTIZATION,
    DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    CaptionerBackend,
    create_captioner_backend,
    cross_examine_caption_claims,
    extract_caption_claims,
)
from .schema import CaptionerBackendInfo, ListeningDossier

INTERROGATION_SCHEMA = "macchina_ricordi.listening.interrogation.v0"


@dataclass(frozen=True)
class InterrogationExchange:
    question: str
    answer: str
    claims: dict[str, bool]
    warnings: list[str]
    asked_at: str


@dataclass(frozen=True)
class InterrogationReport:
    schema: str
    audio: str
    backend: CaptionerBackendInfo
    exchanges: list[InterrogationExchange]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def interrogate_audio(
    audio_path: str | Path,
    questions: Sequence[str],
    *,
    dossier: ListeningDossier | Mapping[str, Any] | None = None,
    backend: str = DEFAULT_CAPTIONER_BACKEND,
    model_id: str = DEFAULT_CAPTIONER_MODEL_ID,
    device: str = DEFAULT_CAPTIONER_DEVICE,
    dtype: str = DEFAULT_CAPTIONER_DTYPE,
    quantization: str = DEFAULT_CAPTIONER_QUANTIZATION,
    max_new_tokens: int = DEFAULT_CAPTIONER_MAX_NEW_TOKENS,
    max_audio_seconds: float = DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS,
    trust_remote_code: bool = DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    captioner_backend: CaptionerBackend | None = None,
) -> InterrogationReport:
    if not questions:
        raise ValueError("interrogate_audio needs at least one question")
    path = Path(audio_path).expanduser().resolve()
    if dossier is None:
        from .features import analyze_audio

        dossier = analyze_audio(path)
    resolved_backend = captioner_backend or create_captioner_backend(
        backend=backend,
        model_id=model_id,
        device=device,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
    )
    exchanges: list[InterrogationExchange] = []
    for question in questions:
        answer = resolved_backend.generate_caption(
            path,
            prompt=question,
            max_new_tokens=max_new_tokens,
            max_audio_seconds=max_audio_seconds,
        )
        claims = extract_caption_claims(answer)
        warnings = cross_examine_caption_claims(claims, dossier)
        exchanges.append(
            InterrogationExchange(
                question=question,
                answer=answer,
                claims=claims,
                warnings=warnings,
                asked_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )
    return InterrogationReport(
        schema=INTERROGATION_SCHEMA,
        audio=str(path),
        backend=CaptionerBackendInfo(
            name=getattr(resolved_backend, "name", backend),
            model_id=getattr(resolved_backend, "model_id", model_id),
            device=getattr(resolved_backend, "device", device),
            dtype=getattr(resolved_backend, "dtype", dtype),
            quantization=getattr(resolved_backend, "quantization", "none"),
            trust_remote_code=getattr(resolved_backend, "trust_remote_code", trust_remote_code),
        ),
        exchanges=exchanges,
    )


def default_interrogation_report_path(audio_path: str | Path) -> Path:
    path = Path(audio_path).expanduser()
    return path.with_name(f"{path.stem}_interrogation_report.json")


def append_interrogation_report(
    report: InterrogationReport,
    *,
    report_path: str | Path | None = None,
) -> Path:
    import json

    path = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else default_interrogation_report_path(report.audio).resolve()
    )
    payload = report.to_dict()
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and isinstance(existing.get("exchanges"), list):
            payload["exchanges"] = existing["exchanges"] + payload["exchanges"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def render_interrogation_markdown(report: InterrogationReport) -> str:
    lines = [
        "# Interrogation",
        "",
        f"- audio: {report.audio}",
        f"- backend: {report.backend.name} ({report.backend.model_id}, {report.backend.quantization})",
        "",
        "Answers are model testimony, not measured DSP evidence; warnings flag",
        "claims the deterministic dossier cannot support.",
        "",
    ]
    for exchange in report.exchanges:
        lines.append(f"## Q: {exchange.question}")
        lines.append("")
        lines.append(exchange.answer if exchange.answer else "(empty answer)")
        lines.append("")
        if exchange.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in exchange.warnings)
            lines.append("")
    return "\n".join(lines)
