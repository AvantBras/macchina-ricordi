from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from macchina_ricordi.judging.label import (
    LABEL_SCHEMA,
    LabelCandidate,
    LabelRecord,
    alarm_ids,
    analyze_detection,
    append_label,
    get_alarm,
    label_context,
    labeled_identities,
    load_label_candidates,
    load_labels,
    load_stream_label_candidates,
    remove_last_label,
    select_label_candidates,
)


def _candidate(path: str, alarm_id: str, value: float) -> LabelCandidate:
    alarm = get_alarm(alarm_id)
    values = {alarm.predictor: value}
    if alarm_id == "temperature_floor":
        values["temperature_floor_eligible"] = 1.0
    return LabelCandidate(path, "stream", audit_values=values)


def _stream(tmp_path: Path, rows: list[dict]) -> Path:
    session = tmp_path / "stream"
    (session / "rounds").mkdir(parents=True)
    (session / "manifest.json").write_text(json.dumps({"run_id": "stream-id"}))
    for row in rows:
        (session / row["round_wav"]).write_bytes(b"RIFF")
    (session / "performance.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return session


def test_alarm_registry_has_all_five_and_exact_boundaries() -> None:
    assert alarm_ids() == (
        "timbral_coherence",
        "silence_gate",
        "temperature_floor",
        "horizon",
        "top_k_cliff",
    )
    assert get_alarm("timbral_coherence").stratum_index(0.95) == 1
    assert get_alarm("timbral_coherence").predicts_yes(0.985, 0.985) is False
    assert get_alarm("silence_gate").predicts_yes(0.60, 0.60) is True
    assert get_alarm("temperature_floor").predicts_yes(1.4, 1.4) is False
    assert get_alarm("horizon").predicts_yes(0.9, 0.9) is True
    assert get_alarm("top_k_cliff").stratum_index(1.0) == 0
    assert get_alarm("top_k_cliff").predicts_yes(1.0, 1.0) is True


def test_stream_candidate_derives_controls_horizon_topk_and_reroll_positions(tmp_path: Path) -> None:
    rows = [
        {
            "round_idx": 7,
            "round_wav": "rounds/a.wav",
            "duration_seconds": 13.5,
            "prompt_duration_seconds": 3.0,
            "model_xp": ".dora/xps/8aa68bbe",
            "active_control": {"temperature": 1.2, "top_k": 25},
            "score": {"timbral_coherence": 0.94, "arc_silence_fraction": 0.4},
        },
        {
            "round_idx": 7,
            "round_wav": "rounds/reroll.wav",
            "duration_seconds": 15.0,
            "prompt_duration_seconds": 4.0,
            "model_xp": ".dora/xps/8aa68bbe",
            "active_control": {"temperature": 1.5, "top_k": 50},
            "score": {},
        },
    ]
    session = _stream(tmp_path, rows)
    before = {path.relative_to(session): path.read_bytes() for path in session.rglob("*") if path.is_file()}
    candidates = load_stream_label_candidates([session])
    after = {path.relative_to(session): path.read_bytes() for path in session.rglob("*") if path.is_file()}

    assert [candidate.source_position for candidate in candidates] == [0, 1]
    assert candidates[0].source_session_id == "stream-id"
    assert candidates[0].prompt_duration_seconds == 3.0
    assert candidates[0].audit_values["horizon_ratio"] == 0.9
    assert candidates[0].audit_values["top_k_cliff_ratio"] == 1.0
    assert candidates[0].audit_values["temperature_floor_eligible"] == 1.0
    assert candidates[0].logged_scores["timbral_coherence"] == 0.94
    assert candidates[0].active_control == {"temperature": 1.2, "top_k": 25}
    assert "timbral_coherence" not in candidates[1].audit_values
    assert after == before


def test_temperature_exemption_and_sweep_predictor_eligibility(tmp_path: Path) -> None:
    rows = [
        {
            "round_idx": 0,
            "round_wav": "rounds/exempt.wav",
            "duration_seconds": 15.0,
            "model_xp": ".dora/xps/4ae63477",
            "active_control": {"temperature": 1.2, "top_k": 25},
            "score": {"timbral_coherence": 0.9},
        }
    ]
    stream = _stream(tmp_path, rows)
    stream_candidate = load_stream_label_candidates([stream])[0]
    assert stream_candidate.audit_values["temperature_floor_eligible"] == 0.0
    with pytest.raises(ValueError, match="no eligible clips"):
        select_label_candidates([stream_candidate], alarm_id="temperature_floor", count=3)

    sweep = tmp_path / "sweep"
    sweep.mkdir()
    paths = []
    for index in range(3):
        path = sweep / f"clip_{index}.wav"
        path.write_bytes(b"RIFF")
        paths.append(path)
    with (sweep / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "sweep_id", "timbral_coherence"))
        writer.writeheader()
        for index, value in enumerate((0.9, 0.97, 0.99)):
            writer.writerow({"path": paths[index].name, "sweep_id": "s", "timbral_coherence": value})
    sweep_candidates = load_label_candidates([sweep])
    assert len(select_label_candidates(sweep_candidates, alarm_id="timbral_coherence", count=3)) == 3
    with pytest.raises(ValueError, match="no eligible clips"):
        select_label_candidates(sweep_candidates, alarm_id="horizon", count=3)


def test_sampling_is_balanced_interleaved_seeded_and_weighted() -> None:
    candidates = []
    for index, value in enumerate([0.90] * 12 + [0.97] * 15 + [0.99] * 30):
        candidates.append(_candidate(f"clip_{index}.wav", "timbral_coherence", value))

    first = select_label_candidates(candidates, alarm_id="timbral_coherence", count=30, seed=4)
    second = select_label_candidates(candidates, alarm_id="timbral_coherence", count=30, seed=4)

    assert [selection.candidate.path for selection in first] == [selection.candidate.path for selection in second]
    assert [selection.stratum_index for selection in first[:6]] == [0, 1, 2, 0, 1, 2]
    assert [sum(selection.stratum_index == index for selection in first) for index in range(3)] == [10, 10, 10]
    assert [first[index].stratum_population for index in range(3)] == [12, 15, 30]
    assert [first[index].inclusion_probability for index in range(3)] == pytest.approx([10 / 12, 10 / 15, 10 / 30])
    assert len({selection.sampling_frame_id for selection in first}) == 1


def test_sampling_redistributes_short_stratum_and_excludes_alarm_identity() -> None:
    candidates = [
        *[_candidate(f"low_{i}", "timbral_coherence", 0.9) for i in range(2)],
        *[_candidate(f"mid_{i}", "timbral_coherence", 0.97) for i in range(10)],
        *[_candidate(f"high_{i}", "timbral_coherence", 0.99) for i in range(10)],
    ]
    excluded = {("mid_0", "broke", "timbral_coherence")}
    selected = select_label_candidates(
        candidates,
        alarm_id="timbral_coherence",
        count=12,
        seed=0,
        exclude=excluded,
    )
    assert sum(selection.stratum_index == 0 for selection in selected) == 2
    assert all(selection.candidate.path != "mid_0" for selection in selected)
    with pytest.raises(ValueError, match="no eligible clips"):
        select_label_candidates(candidates[2:], alarm_id="timbral_coherence", count=3)


def test_label_store_roundtrip_identity_malformed_tolerance_and_undo(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    selection = select_label_candidates(
        [
            _candidate("low", "timbral_coherence", 0.9),
            _candidate("mid", "timbral_coherence", 0.97),
            _candidate("high", "timbral_coherence", 0.99),
        ],
        alarm_id="timbral_coherence",
        count=3,
    )[0]
    record = LabelRecord(
        path=selection.candidate.path,
        alarm_id=selection.alarm_id,
        question=selection.question,
        answer="yes",
        session_id="session",
        notes="a lurch",
        context=label_context(selection),
    )
    append_label(record, path=store)
    with store.open("a") as handle:
        handle.write("not json\n")
    assert load_labels(store)[0].notes == "a lurch"
    assert labeled_identities(load_labels(store)) == {(record.path, "broke", "timbral_coherence")}
    with pytest.raises(ValueError, match="last label store line"):
        remove_last_label(record, path=store)
    store.write_text(store.read_text().splitlines()[0] + "\n")
    remove_last_label(record, path=store)
    assert store.read_text() == ""


def _record(
    value: float,
    answer: str,
    probability: float,
    *,
    session: str = "s1",
    frame: str = "f1",
) -> LabelRecord:
    return LabelRecord(
        path=f"{session}_{frame}_{value}_{answer}.wav",
        alarm_id="temperature_floor",
        question="gapped",
        answer=answer,
        session_id=session,
        context={
            "audit_values": {"temperature": value},
            "inclusion_probability": probability,
            "sampling_frame_id": frame,
        },
    )


def test_detection_reports_raw_weighted_reference_unsure_and_recommendation() -> None:
    records = [
        _record(0.5, "yes", 0.1),
        _record(1.2, "no", 0.5),
        _record(2.0, "yes", 1.0),
        _record(2.1, "no", 1.0),
        _record(1.3, "unsure", 0.25),
    ]
    report = analyze_detection(records, alarm_id="temperature_floor", min_labels=2)
    frame = report.frames[0]
    reference = next(point for point in frame.points if point.threshold == 1.4)

    assert (reference.raw_tp, reference.raw_fp, reference.raw_tn, reference.raw_fn) == (1, 1, 1, 1)
    assert reference.weighted_tp == 10.0 and reference.weighted_fp == 2.0
    assert reference.weighted_precision == pytest.approx(10 / 12)
    assert reference.weighted_recall == pytest.approx(10 / 11)
    assert frame.unsure == 1 and frame.unsure_weight == 4.0
    assert frame.unsure_weight_share == pytest.approx(4 / 18)
    assert frame.sufficient is True and frame.recommended_threshold is not None
    assert report.raw_aggregate is not None and report.raw_aggregate.recommended_threshold is None


def test_detection_suppresses_one_class_and_keeps_frames_separate() -> None:
    records = [
        _record(0.8, "yes", 0.5, session="s1", frame="f1"),
        _record(1.8, "yes", 0.5, session="s1", frame="f1"),
        _record(0.8, "yes", 0.5, session="s2", frame="f2"),
        _record(1.8, "no", 0.5, session="s2", frame="f2"),
    ]
    report = analyze_detection(records, alarm_id="temperature_floor", min_labels=2)
    assert len(report.frames) == 2
    assert report.frames[0].recommended_threshold is None
    assert report.frames[1].recommended_threshold is not None
    assert report.raw_aggregate is not None and report.raw_aggregate.sufficient is False


def test_label_schema_constant_is_v1() -> None:
    assert LABEL_SCHEMA == "macchina_ricordi.judging.label.v1"


def test_detection_rejects_invalid_minimum() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        analyze_detection([], alarm_id="silence_gate", min_labels=0)
