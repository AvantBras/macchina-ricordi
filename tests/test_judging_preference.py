from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.judging.ab import (
    AbCandidate,
    load_stream_candidates,
    parse_verdict,
    run_ab_session,
    select_pairs,
)
from macchina_ricordi.judging.preference import (
    PreferenceRecord,
    append_preference,
    load_preferences,
)


def _record(path_a: str, path_b: str, winner: str, index: int = 0) -> PreferenceRecord:
    return PreferenceRecord(pair_id=f"pair_{index:03d}", path_a=path_a, path_b=path_b, winner=winner)


# ── store ─────────────────────────────────────────────────────────────


def test_append_and_load_roundtrip(tmp_path):
    store = tmp_path / "prefs.jsonl"
    append_preference(_record("a.wav", "b.wav", "a"), path=store)
    append_preference(_record("c.wav", "d.wav", "tie", index=1), path=store)
    loaded = load_preferences(store)
    assert [record.winner for record in loaded] == ["a", "tie"]
    assert loaded[0].created_at, "append should stamp a timestamp"


def test_load_skips_corrupt_lines(tmp_path):
    store = tmp_path / "prefs.jsonl"
    append_preference(_record("a.wav", "b.wav", "a"), path=store)
    with store.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
        handle.write(json.dumps({"winner": "nonsense"}) + "\n")
        handle.write("\n")
    assert len(load_preferences(store)) == 1


def test_append_rejects_invalid_verdict(tmp_path):
    with pytest.raises(ValueError):
        append_preference(_record("a.wav", "b.wav", "maybe"), path=tmp_path / "prefs.jsonl")


# ── pair selection ────────────────────────────────────────────────────


def _candidates(groups: int = 3, per_group: int = 4) -> list[AbCandidate]:
    return [
        AbCandidate(path=f"g{group}_c{index}.wav", group=f"group{group}")
        for group in range(groups)
        for index in range(per_group)
    ]


def test_select_pairs_is_cross_group_by_default():
    pairs = select_pairs(_candidates(), count=10, seed=1)
    assert len(pairs) == 10
    assert all(pair.first.group != pair.second.group for pair in pairs)


def test_select_pairs_excludes_already_judged():
    candidates = _candidates()
    first = select_pairs(candidates, count=5, seed=3)
    excluded = {pair.key for pair in first}
    second = select_pairs(candidates, count=5, seed=3, exclude_keys=excluded)
    assert not ({pair.key for pair in second} & excluded)


def test_select_pairs_shuffles_sides():
    """Blindness depends on A/B order not encoding the group."""
    pairs = select_pairs(_candidates(groups=2, per_group=12), count=24, seed=5)
    first_groups = {pair.first.group for pair in pairs}
    assert len(first_groups) == 2


# ── stream sourcing ───────────────────────────────────────────────────


def _write_stream_session(root, name="20260808T173600Z", rows=None, run_id=None):
    session = root / name
    (session / "rounds").mkdir(parents=True)
    (session / "manifest.json").write_text(json.dumps({"run_id": run_id or name}))
    lines = []
    for row in rows or []:
        wav = row.get("round_wav")
        if wav and row.pop("_write_wav", True):
            (session / wav).write_bytes(b"RIFF")
        lines.append(json.dumps(row))
    (session / "performance.jsonl").write_text("\n".join(lines) + "\n")
    return session


def _round(index, *, score=None, xp=".dora/xps/abc123", wav=None):
    return {
        "round_idx": index,
        "round_wav": wav if wav is not None else f"rounds/round_{index}.wav",
        "model_xp": xp,
        "score": {"onset_density": 4.0, "novelty_whitened": float(index)} if score is None else score,
    }


def test_load_stream_candidates_reads_rounds(tmp_path):
    session = _write_stream_session(tmp_path, rows=[_round(0), _round(1)])
    candidates = load_stream_candidates([session])
    assert len(candidates) == 2
    assert all(candidate.group == "abc123" for candidate in candidates)
    assert candidates[0].session_id == "20260808T173600Z"
    assert candidates[0].round_idx == 0
    assert candidates[0].metrics["onset_density"] == 4.0
    assert Path(candidates[0].path).is_absolute()


def test_load_stream_candidates_never_writes_to_the_session(tmp_path):
    session = _write_stream_session(tmp_path, rows=[_round(0), _round(1)])
    before = {path.relative_to(session): path.read_bytes() for path in session.rglob("*") if path.is_file()}
    load_stream_candidates([session], group_by="session")
    after = {path.relative_to(session): path.read_bytes() for path in session.rglob("*") if path.is_file()}
    assert after == before


# ── session loop ──────────────────────────────────────────────────────


def _session_pairs():
    return select_pairs(
        [
            AbCandidate(path="a.wav", group="left"),
            AbCandidate(path="b.wav", group="right"),
            AbCandidate(path="c.wav", group="left"),
            AbCandidate(path="d.wav", group="right"),
        ],
        count=3,
        seed=1,
    )


def test_session_records_the_question_and_writes_a_sidecar(tmp_path):
    store = tmp_path / "preferences.jsonl"
    answers = iter(["a", "s", "b"])
    records = run_ab_session(
        _session_pairs(),
        store_path=store,
        question="harder",
        seconds=8.0,
        session_id="20260810T120000Z",
        input_fn=lambda _prompt: next(answers),
        print_fn=lambda *_args: None,
        play_fn=lambda _path: True,
        sidecar={"sources": ["artifacts/streams/x"]},
    )
    assert [record.winner for record in records] == ["a", "b"]
    assert all(record.question == "harder" for record in records)
    assert all(record.seconds_played == 8.0 for record in records)
    assert all(record.session_id == "20260810T120000Z" for record in records)

    payload = json.loads((tmp_path / "sessions" / "20260810T120000Z_session.json").read_text())
    assert payload["question"] == "harder"
    assert payload["sources"] == ["artifacts/streams/x"]
    assert [row["outcome"] for row in payload["pairs"]] == ["a", "skipped", "b"]


@pytest.mark.parametrize("question", ["newer", "broke"])
def test_session_refuses_a_question_it_cannot_ask_honestly(question, tmp_path):
    with pytest.raises(ValueError):
        run_ab_session(
            _session_pairs(),
            store_path=tmp_path / "preferences.jsonl",
            question=question,
            input_fn=lambda _prompt: "a",
            print_fn=lambda *_args: None,
            play_fn=lambda _path: True,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a", "a"),
        ("A", "a"),
        (" b ", "b"),
        ("2", "b"),
        ("tie", "tie"),
        ("=", "tie"),
        ("r", "replay"),
        ("q", "quit"),
        ("nonsense", None),
        ("", None),
    ],
)
def test_parse_verdict(raw, expected):
    assert parse_verdict(raw) == expected
