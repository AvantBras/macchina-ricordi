from __future__ import annotations

import json
from pathlib import Path

from macchina_ricordi.models.catalog import build_catalog, format_catalog_text, gather_generations
from macchina_ricordi.sessions.read import facts_of, load_session
from macchina_ricordi.sessions.store import seal_session


def _generation(folder: Path) -> Path:
    samples = folder / "samples"
    samples.mkdir(parents=True)
    (samples / "000.wav").write_bytes(b"fake")
    (samples / "000_listening_report.json").write_text(
        json.dumps({"summary": {"global_summary": "dry clicks"}, "whole_clip": {"reward": {"score": 0.75}}}),
        encoding="utf-8",
    )
    (folder / "run.json").write_text(
        json.dumps(
            {
                "run_id": "20261006T090503Z",
                "request": {"mode": "unconditional", "duration_seconds": 8.0, "seed": 3},
                "generation": {"source": {"xp_path": "/r/.dora/xps/abcd1234"}},
                "model": {"label": "disco_run", "workspace": ".workspace_disco"},
                "output": {"prefix": "night"},
                "files": ["samples/000.wav"],
            }
        ),
        encoding="utf-8",
    )
    return folder


def test_a_take_reads_its_listening_score_where_the_report_keeps_it(tmp_path: Path) -> None:
    session = load_session(_generation(tmp_path / "run"))
    assert session.takes()[0].scores == {"listening_score": 0.75}
    assert session.facts["run_id"] == "20261006T090503Z" and session.facts["prefix"] == "night"


def test_sealed_sessions_keep_their_facts_and_older_seals_fall_back(tmp_path: Path) -> None:
    folder = _generation(tmp_path / "artifacts" / "sessions" / "2026-10-06" / "generate_20261006T090503Z")
    sealed = seal_session(folder)
    assert load_session(folder).facts == sealed.facts
    record = json.loads((folder / "session.json").read_text())
    record.pop("facts")
    (folder / "session.json").write_text(json.dumps(record))
    old = load_session(folder)
    assert old.facts == {} and facts_of(old)["model_label"] == "disco_run"


def test_the_catalog_lists_generations_and_duets_from_sessions(tmp_path: Path) -> None:
    _generation(tmp_path / "artifacts" / "sessions" / "2026-10-06" / "generate_20261006T090503Z")
    duet = tmp_path / "artifacts" / "sessions" / "2026-10-06" / "duet_20261006T090656Z"
    duet.mkdir(parents=True)
    (duet / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "macchina_ricordi.duet_session.v1",
                "mix": "mix.wav",
                "voices": {"a": {"source": {"xp_path": "/r/.dora/xps/abcd1234"}, "stem": "voice_a.wav"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".dora" / "xps").mkdir(parents=True)

    runs = gather_generations(tmp_path)
    assert [(g.run_id, g.prefix, g.model_label, g.listening_score) for g in runs] == [
        ("20261006T090503Z", "night", "disco_run", 0.75)
    ]
    catalog = build_catalog(tmp_path, include_lineage=False)
    assert [(e.kind, e.model_sigs, e.take_count) for e in catalog.other_sessions] == [("duet", ("abcd1234",), 2)]
    assert "duet_20261006T090656Z" in format_catalog_text(catalog)
