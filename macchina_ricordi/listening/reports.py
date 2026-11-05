"""Read a take's listening sidecars: the one reader for what ``listen file --sidecars`` left next to the audio.

The session reader (a take's ``listening_score``) and the catalog (a run's listening summary) both read through
here, so they agree on where the score is: ``whole_clip.reward.score``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .render import default_json_report_path, default_markdown_report_path

DESCRIPTION_KEYS = ("loudness", "brightness", "density", "pulse", "noisiness", "attack", "resonance")


@dataclass(frozen=True)
class ListeningReport:
    report_path: Path
    markdown_path: Path | None
    summary: str | None
    score: float | None
    labels: tuple[str, ...]
    error: str | None = None
    """Set when the report exists but cannot be read."""


def read_listening_report(audio: str | Path | None) -> ListeningReport | None:
    """The listening report beside ``audio``; ``None`` when there is none."""
    if audio is None:
        return None
    report_path = default_json_report_path(audio)
    if not report_path.exists():
        return None
    markdown = default_markdown_report_path(audio)
    markdown_path = markdown if markdown.exists() else None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ListeningReport(report_path, markdown_path, None, None, (), error=str(exc))
    if not isinstance(payload, dict):
        return ListeningReport(report_path, markdown_path, None, None, (), error="not a JSON object")
    return ListeningReport(
        report_path=report_path,
        markdown_path=markdown_path,
        summary=_summary(payload),
        score=_score(payload),
        labels=_labels(payload),
    )


def _summary(payload: dict[str, Any]) -> str | None:
    summary = payload.get("summary")
    text = summary.get("global_summary") if isinstance(summary, dict) else None
    return text.strip() if isinstance(text, str) and text.strip() else None


def _score(payload: dict[str, Any]) -> float | None:
    whole_clip = payload.get("whole_clip")
    reward = whole_clip.get("reward") if isinstance(whole_clip, dict) else None
    score = reward.get("score") if isinstance(reward, dict) else None
    return float(score) if isinstance(score, int | float) and not isinstance(score, bool) else None


def _labels(payload: dict[str, Any]) -> tuple[str, ...]:
    whole_clip = payload.get("whole_clip")
    description = whole_clip.get("description") if isinstance(whole_clip, dict) else None
    if not isinstance(description, dict):
        return ()
    return tuple(
        value.strip() for key in DESCRIPTION_KEYS if isinstance(value := description.get(key), str) and value.strip()
    )
