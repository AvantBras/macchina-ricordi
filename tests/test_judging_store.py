from __future__ import annotations

from pathlib import Path

import pytest

from macchina_ricordi.judging.label import LabelRecord, alarm_ids, append_label, get_alarm
from macchina_ricordi.judging.preference import PreferenceRecord, append_preference
from macchina_ricordi.judging.store import (
    ImpressionRecord,
    WitnessRecord,
    append_impression,
    append_witness,
    load_impressions,
    load_verdicts,
    verdict_counts,
)


def _stores(tmp_path: Path) -> dict[str, Path]:
    return {
        "preferences_path": tmp_path / "preferences.jsonl",
        "labels_path": tmp_path / "labels.jsonl",
        "witness_path": tmp_path / "witness.jsonl",
        "impressions_path": tmp_path / "impressions.jsonl",
    }


def test_impressions_need_a_rating_or_a_note(tmp_path: Path) -> None:
    for bad in ({"rating": 0}, {"rating": 6}, {"rating": True}, {"note": "  "}, {}):
        with pytest.raises(ValueError):
            ImpressionRecord(model="aaaa1111", **bad)
    store = tmp_path / "impressions.jsonl"
    append_impression(ImpressionRecord(model="aaaa1111", rating=3), path=store)
    append_impression(ImpressionRecord(model="bbbb2222", note="glassy", session="s2"), path=store)
    assert [r.rating for r in load_impressions(store)] == [3, None]
    assert [r.note for r in load_impressions(store, model="bbbb2222")] == ["glassy"]
    verdicts = load_verdicts(**_stores(tmp_path), kinds=("impression",))
    assert [(v.question, v.answer, v.session_id) for v in verdicts] == [("aaaa1111", "3", ""), ("bbbb2222", "", "s2")]


def test_load_verdicts_reads_every_kind_oldest_first(tmp_path: Path) -> None:
    stores = _stores(tmp_path)
    append_witness(
        WitnessRecord(path="takes/3.wav", backend="audio-flamingo-3", score=0.6, created_at="2026-10-06T10:00:00"),
        path=stores["witness_path"],
    )
    append_preference(
        PreferenceRecord(
            pair_id="p1", path_a="takes/1.wav", path_b="takes/2.wav", winner="a", created_at="2026-10-06T09:00:00"
        ),
        path=stores["preferences_path"],
    )
    alarm = get_alarm(alarm_ids()[0])
    append_label(
        LabelRecord(
            path="takes/1.wav",
            alarm_id=alarm.id,
            question=alarm.question,
            answer="yes",
            created_at="2026-10-06T09:30:00",
        ),
        path=stores["labels_path"],
    )

    verdicts = load_verdicts(**stores)

    assert [v.kind for v in verdicts] == ["ab", "label", "witness"]
    assert verdicts[0].paths == ("takes/1.wav", "takes/2.wav") and verdicts[0].answer == "a"
    assert verdicts[2].answer == "0.600" and verdicts[2].question == "witness"
    assert verdict_counts(verdicts)["ab"] == {"better": 1}
    assert [v.kind for v in load_verdicts(**stores, kinds=("witness",))] == ["witness"]
