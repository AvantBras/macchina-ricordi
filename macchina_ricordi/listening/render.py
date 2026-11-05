from __future__ import annotations

import json
from pathlib import Path

from ..cli.common import format_metric_value
from .schema import ListeningDossier


def default_json_report_path(audio_path: str | Path) -> Path:
    path = Path(audio_path).expanduser()
    return path.with_name(f"{path.stem}_listening_report.json")


def default_markdown_report_path(audio_path: str | Path) -> Path:
    path = Path(audio_path).expanduser()
    return path.with_name(f"{path.stem}_listening.md")


def _format_event_time(event_time: float, event_end: float | None) -> str:
    if event_end is None:
        return f"{event_time:.2f}s"
    return f"{event_time:.2f}s–{event_end:.2f}s"


def _notable_events_lines(dossier: ListeningDossier) -> list[str]:
    events = getattr(dossier, "notable_events", []) or []
    if not events:
        return []
    lines = [
        "",
        "## Notable Events",
        "",
        "| time | kind | strength | detail |",
        "| --- | --- | ---: | --- |",
    ]
    for event in events:
        lines.append(
            f"| {_format_event_time(event.time_seconds, event.end_seconds)} "
            f"| {event.kind} "
            f"| {event.strength:.2f} "
            f"| {event.detail} |"
        )
    return lines


def _spectrogram_image_line(dossier: ListeningDossier) -> list[str]:
    sketch_path = getattr(dossier.sketches, "mel_spectrogram_path", None)
    if not sketch_path:
        return []
    return ["", f"![Mel spectrogram]({sketch_path})"]


def render_dossier_json(dossier: ListeningDossier) -> str:
    return json.dumps(dossier.to_dict(), indent=2, sort_keys=False)


def _evidence_rows(dossier: ListeningDossier) -> list[tuple[str, object]]:
    reward = dossier.whole_clip.reward
    features = dossier.whole_clip.features
    return [
        ("score", reward.get("score", 0.0)),
        ("tempo_instability", reward.get("tempo_instability", 0.0)),
        ("onset_density", features.get("onset_density", 0.0)),
        ("pulse_instability", features.get("pulse_instability", 0.0)),
        ("spectral_centroid_mean", features.get("spectral_centroid_mean", 0.0)),
        ("spectral_flatness_mean", features.get("spectral_flatness_mean", 0.0)),
        ("silence_fraction", features.get("silence_fraction", 0.0)),
        ("max_silence_gap_seconds", features.get("max_silence_gap_seconds", 0.0)),
    ]


def _semantic_lines(dossier: ListeningDossier) -> list[str]:
    semantic = getattr(dossier, "semantic_probes", None)
    if semantic is None:
        return []
    lines = [
        "",
        "## Semantic Probes",
        "",
        "Semantic probes are comparative CLAP hints, not measured DSP evidence.",
        "",
        f"- backend: {semantic.backend.name} {semantic.backend.version} ({semantic.backend.device})",
        f"- vocabulary: {semantic.vocabulary_version}",
        "",
    ]
    if semantic.strong:
        lines.append("Strong:")
        lines.extend(f"- {score.key}: {format_metric_value(score.score)} ({score.label})" for score in semantic.strong)
        lines.append("")
    if semantic.weak:
        lines.append("Weak Negative Controls:")
        lines.extend(f"- {score.key}: {format_metric_value(score.score)}" for score in semantic.weak)
        lines.append("")
    if semantic.uncertain:
        lines.append("Uncertain:")
        for score in semantic.uncertain:
            reason = f" - {score.reason}" if score.reason else ""
            lines.append(f"- {score.key}: {format_metric_value(score.score)}{reason}")
        lines.append("")
    return lines


def _captioner_lines(dossier: ListeningDossier) -> list[str]:
    witness = getattr(dossier, "captioner_witness", None)
    if witness is None:
        return []
    lines = [
        "",
        "## Captioner Witness",
        "",
        "Captioner output is a model witness, not ground truth.",
        "",
        f"- backend: {witness.backend.name}",
        f"- model: {witness.backend.model_id}",
        f"- quantization: {witness.backend.quantization}",
        f"- prompt_profile: {witness.prompt_profile}",
        f"- with_features: {witness.agreement.get('with_features', 'unknown')}",
        f"- with_semantic_probes: {witness.agreement.get('with_semantic_probes', 'unknown')}",
        "",
        witness.summary,
        "",
    ]
    active_claims = [key for key, value in witness.claims.items() if value]
    if active_claims:
        lines.append("Claims:")
        lines.append(f"- {', '.join(active_claims)}")
        lines.append("")
    if witness.warnings:
        lines.append("Cross-Examination:")
        lines.extend(f"- {warning}" for warning in witness.warnings)
        lines.append("")
    return lines


def render_dossier_markdown(dossier: ListeningDossier) -> str:
    reward = dossier.whole_clip.reward
    description = dossier.whole_clip.description
    features = dossier.whole_clip.features
    lines = [
        f"# Listening Dossier: {dossier.audio.path}",
        "",
        "## Source",
        "",
        f"- source: {dossier.provenance.source}",
        f"- audio_path: {dossier.provenance.audio_path}",
        f"- analyzed_at: {dossier.analysis.analyzed_at}",
        f"- analysis_version: {dossier.analysis.version}",
    ]
    lines.extend(_spectrogram_image_line(dossier))
    lines.extend(
        [
            "",
            "## Summary",
            "",
            dossier.summary.global_summary,
            "",
            dossier.summary.time_contour,
            "",
            "## Whole Clip",
            "",
            f"- duration: {dossier.audio.duration_seconds:.2f}s",
            f"- sample_rate: {dossier.audio.sample_rate}",
            f"- score: {format_metric_value(reward.get('score', 0.0))}",
            f"- loudness: {description.get('loudness', '')}",
            f"- brightness: {description.get('brightness', '')}",
            f"- attack: {description.get('attack', '')}",
            f"- noisiness: {description.get('noisiness', '')}",
            f"- density: {description.get('density', '')}",
            f"- pulse: {description.get('pulse', '')}",
            f"- resonance: {description.get('resonance', '')}",
            f"- silence_fraction: {format_metric_value(features.get('silence_fraction', 0.0))}",
            f"- max_silence_gap_seconds: {format_metric_value(features.get('max_silence_gap_seconds', 0.0))}",
        ]
    )
    lines.extend(_semantic_lines(dossier))
    lines.extend(_captioner_lines(dossier))
    lines.extend(["", "## Form Regions", ""])
    for region in dossier.timeline.regions:
        evidence = ", ".join(f"{key}={value}" for key, value in region.evidence.items())
        lines.append(f"- {region.start_seconds:.2f}-{region.end_seconds:.2f}s: {region.label}")
        if evidence:
            lines.append(f"  evidence: {evidence}")

    lines.extend(_notable_events_lines(dossier))

    lines.extend(
        [
            "",
            "## Timeline",
            "",
        ]
    )
    for window in dossier.timeline.windows:
        lines.append(f"- {window.start_seconds:.2f}-{window.end_seconds:.2f}s: {window.label}")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "| metric | value |",
            "| --- | ---: |",
        ]
    )
    for key, value in _evidence_rows(dossier):
        lines.append(f"| {key} | {format_metric_value(value)} |")

    lines.extend(
        [
            "",
            "## Sketch",
            "",
            "```text",
            dossier.sketches.spectral,
            "```",
        ]
    )
    numeric_grid = getattr(dossier.sketches, "numeric_spectrogram", None)
    if numeric_grid:
        lines.extend(["", "```text", numeric_grid, "```"])
    if dossier.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in dossier.warnings)

    return "\n".join(lines) + "\n"
