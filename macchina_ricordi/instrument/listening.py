from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from macchina_ricordi.listening.render import (
    default_json_report_path,
    default_markdown_report_path,
    render_dossier_json,
    render_dossier_markdown,
)


@dataclass(frozen=True)
class MusicGenListeningResult:
    audio_path: Path
    report_path: Path
    markdown_path: Path
    summary: str
    labels: str
    score: float | None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def write_listening_sidecars(
    audio_path: str | Path,
    *,
    analyzer: Callable[[str | Path], Any] | None = None,
) -> MusicGenListeningResult:
    resolved_audio = Path(audio_path).expanduser().resolve()
    report_path = default_json_report_path(resolved_audio).resolve()
    markdown_path = default_markdown_report_path(resolved_audio).resolve()
    if analyzer is None:
        from macchina_ricordi.listening.features import analyze_audio

        analyzer = analyze_audio

    try:
        dossier = analyzer(resolved_audio)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_dossier_json(dossier), encoding="utf-8")
        markdown_path.write_text(render_dossier_markdown(dossier), encoding="utf-8")
        return MusicGenListeningResult(
            audio_path=resolved_audio,
            report_path=report_path,
            markdown_path=markdown_path,
            summary=_summary_from_dossier(dossier),
            labels=_labels_from_dossier(dossier),
            score=_score_from_dossier(dossier),
        )
    except Exception as exc:
        return MusicGenListeningResult(
            audio_path=resolved_audio,
            report_path=report_path,
            markdown_path=markdown_path,
            summary="",
            labels="",
            score=None,
            error=str(exc),
        )


def write_listening_sidecars_for_paths(
    audio_paths: Sequence[str | Path],
    *,
    analyzer: Callable[[str | Path], Any] | None = None,
) -> list[MusicGenListeningResult]:
    return [write_listening_sidecars(path, analyzer=analyzer) for path in audio_paths]


def _summary_from_dossier(dossier: Any) -> str:
    summary = getattr(dossier, "summary", None)
    global_summary = getattr(summary, "global_summary", "")
    return str(global_summary) if global_summary else ""


def _score_from_dossier(dossier: Any) -> float | None:
    whole_clip = getattr(dossier, "whole_clip", None)
    reward = getattr(whole_clip, "reward", None)
    if isinstance(reward, dict):
        score = reward.get("score")
        if isinstance(score, int | float):
            return float(score)
    return None


def _labels_from_dossier(dossier: Any) -> str:
    whole_clip = getattr(dossier, "whole_clip", None)
    description = getattr(whole_clip, "description", None)
    labels: list[str] = []
    if isinstance(description, dict):
        for key in ("loudness", "brightness", "density", "pulse", "noisiness", "attack", "resonance"):
            value = description.get(key)
            if value:
                labels.append(str(value))
    return ", ".join(dict.fromkeys(labels))
