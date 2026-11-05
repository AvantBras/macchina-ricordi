from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from macchina_ricordi.instrument.replay import _default_replay_backend_factory


def source(digest: str):
    return SimpleNamespace(
        backend="audiocraft",
        model_id="published-model",
        device="cpu",
        generation={"source": {"kind": "bundle", "bundle_path": "/local/model", "weights_digest": digest}},
    )


def test_bundle_replay_selects_the_recorded_bundle_not_the_public_default(monkeypatch) -> None:
    bundle = SimpleNamespace(
        path=Path("/local/model"), config={"name": "published-model", "weights": {"sha256": "a" * 64}}
    )
    monkeypatch.setattr("macchina_ricordi.models.bundle.read_bundle", lambda path: bundle)
    monkeypatch.setattr("macchina_ricordi.musicgen.backend.build_audiocraft_backend", lambda **kwargs: kwargs)
    result = _default_replay_backend_factory(source("a" * 64))
    assert result["xp_path"] == bundle.path
    assert result["model_id"] == "published-model"


def test_bundle_replay_refuses_a_changed_snapshot(monkeypatch) -> None:
    bundle = SimpleNamespace(
        path=Path("/local/model"), config={"name": "published-model", "weights": {"sha256": "b" * 64}}
    )
    monkeypatch.setattr("macchina_ricordi.models.bundle.read_bundle", lambda path: bundle)
    with pytest.raises(ValueError, match="weights changed"):
        _default_replay_backend_factory(source("a" * 64))
