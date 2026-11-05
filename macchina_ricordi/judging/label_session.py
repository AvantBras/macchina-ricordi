"""Line-based single-clip alarm labeling session."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .ab import find_player, play_clip
from .label import (
    LabelRecord,
    LabelSelection,
    analyze_detection,
    append_label,
    get_alarm,
    label_context,
    write_label_sidecar,
)
from .preference import new_session_id
from .questions import SINGLE, get_question

_ANSWER_ALIASES = {
    "y": "yes",
    "yes": "yes",
    "n": "no",
    "no": "no",
    "u": "unsure",
    "unsure": "unsure",
    "r": "replay",
    "replay": "replay",
    "s": "skip",
    "skip": "skip",
    "q": "quit",
    "quit": "quit",
}


def parse_label_answer(raw: str) -> str | None:
    return _ANSWER_ALIASES.get(raw.strip().lower())


def _presented_row(selection: LabelSelection, outcome: str) -> dict[str, Any]:
    candidate = selection.candidate
    alarm = get_alarm(selection.alarm_id)
    return {
        "path": candidate.path,
        "alarm_id": selection.alarm_id,
        "question": selection.question,
        "predictor": alarm.predictor,
        "predictor_value": candidate.audit_values[alarm.predictor],
        "stratum_index": selection.stratum_index,
        "stratum_label": selection.stratum_label,
        "stratum_population": selection.stratum_population,
        "stratum_offered": selection.stratum_offered,
        "inclusion_probability": selection.inclusion_probability,
        "sampling_frame_id": selection.sampling_frame_id,
        "outcome": outcome,
    }


def run_label_session(
    selections: Sequence[LabelSelection],
    *,
    store_path: str | Path,
    listener: str = "me",
    seconds: float | None = None,
    session_id: str = "",
    input_fn: Any = input,
    print_fn: Any = print,
    play_fn: Any = None,
    sidecar: Mapping[str, Any] | None = None,
) -> list[LabelRecord]:
    if not selections:
        return []
    alarm = get_alarm(selections[0].alarm_id)
    question = get_question(alarm.question)
    if question.type != SINGLE:
        raise ValueError(f"question {question.id!r} is not a single-clip question")
    if any(selection.alarm_id != alarm.id for selection in selections):
        raise ValueError("one labeling session must contain exactly one alarm")

    player = find_player()
    if player is None and play_fn is None:
        print_fn("No CLI audio player found (tried ffplay, play, paplay, aplay).")
        return []
    run_id = session_id or new_session_id()
    written: list[LabelRecord] = []
    outcomes = {index: "unreached" for index in range(len(selections))}
    presented: list[dict[str, Any]] = []

    def play(selection: LabelSelection) -> None:
        if play_fn is not None:
            play_fn(selection.candidate.playable_path)
        else:
            play_clip(selection.candidate.playable_path, seconds=seconds, player=player)

    def finish(message: str) -> list[LabelRecord]:
        presented.extend(
            _presented_row(selection, outcomes[index])
            for index, selection in enumerate(selections)
            if outcomes[index] == "unreached"
        )
        sidecar_path = write_label_sidecar(
            {
                "alarm_id": alarm.id,
                "question": question.id,
                "prompt": question.prompt,
                "listener": listener,
                "seconds_played": seconds,
                "player": player[0] if player else None,
                "labels": len(written),
                "clips": presented,
                **dict(sidecar or {}),
            },
            session_id=run_id,
            store_path=store_path,
        )
        print_fn(message)
        print_fn(f"Session sidecar → {sidecar_path}")
        if written:
            report = analyze_detection(written, alarm_id=alarm.id, min_labels=8, session_id=run_id)
            frame = report.frames[0] if report.frames else None
            if frame is not None and frame.recommended_threshold is not None:
                print_fn(f"Session max-F1 threshold: {frame.recommended_threshold:.6g}")
        return written

    def collect() -> list[LabelRecord]:
        prompt = f"  {question.prompt} [yes/no/unsure, r=replay, s=skip, q=quit]: "
        for index, selection in enumerate(selections):
            print_fn(f"\n[{index + 1}/{len(selections)}] listening")
            play(selection)
            while True:
                answer = parse_label_answer(input_fn(prompt))
                while answer is None:
                    answer = parse_label_answer(input_fn("  please answer yes, no, unsure, r, s or q: "))
                if answer == "replay":
                    play(selection)
                    continue
                if answer == "quit":
                    return finish(f"Stopped after {len(written)} labels.")
                if answer == "skip":
                    outcomes[index] = "skipped"
                    presented.append(_presented_row(selection, "skipped"))
                    break
                record = LabelRecord(
                    path=selection.candidate.path,
                    alarm_id=alarm.id,
                    question=question.id,
                    answer=answer,
                    listener=listener,
                    session_id=run_id,
                    seconds_played=seconds,
                    context=label_context(selection),
                )
                append_label(record, path=store_path)
                written.append(record)
                outcomes[index] = answer
                presented.append(_presented_row(selection, answer))
                break
        return finish(f"\nRecorded {len(written)} labels → {store_path}")

    try:
        return collect()
    except KeyboardInterrupt:
        finish(f"Interrupted after {len(written)} labels.")
        raise
