"""The desktop half: where files live, adding a model, and the panel as a server.

The panel is exercised over real HTTP on a free port, with a fake backend behind
it, so none of this needs ONNX Runtime, torch or an audio device.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from appliance.player import Player
from appliance.spec import ModelSpec


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(
        dim=16,
        num_heads=2,
        num_layers=2,
        n_q=4,
        card=32,
        hidden_scale=2.0,
        layer_scale=1e-4,
        delays=[0, 1, 2, 3],
        frame_rate=10.0,
        sample_rate=80,
        channels=1,
        segment_duration=4.0,
        name="stub",
    )


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("MACCHINA_RICORDI_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


# ---- the panel as a server ------------------------------------------------------------


@pytest.fixture
def panel(home, monkeypatch, spec):
    from appliance import panel as panel_module

    monkeypatch.setattr(panel_module, "load_player", lambda path, **kw: (Player(FakeBackend(spec)), spec))
    session = panel_module.Session([home / "boxes"])
    server = panel_module.make_server(session, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def call(path, body=None, headers=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=body, headers=headers or {}, method="GET" if body is None else "POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    yield call, session, server, thread, port
    server.shutdown()
    server.server_close()


class FakeBackend:
    def __init__(self, spec):
        self.spec = spec

    def start(self, batch, horizon):
        return None

    def step(self, tokens, offset, state):
        return np.random.default_rng(offset).standard_normal(tokens.shape + (self.spec.card,)), state

    def decode(self, codes):
        return np.full(codes.shape[-1] * 8, 0.1, dtype=np.float32)


def test_panel_refuses_a_file_that_is_not_a_model(panel):
    call, *_ = panel
    status, body = call("/api/models", b"hello", {"X-Filename": "hello.nrbox"})
    assert status == 400 and "not a macchina-ricordi model" in body["error"]


def test_panel_refuses_requests_from_other_sites(panel):
    call, _, _, thread, port = panel
    status, _ = call("/api/quit", b"{}", {"Origin": "https://example.com"})
    assert status == 403 and thread.is_alive()
    status, _ = call("/api/state")
    assert status == 200
    status, _ = call("/api/knobs", b'{"temperature": 1.5}', {"Origin": f"http://127.0.0.1:{port}"})
    assert status == 200
