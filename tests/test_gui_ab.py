from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from macchina_ricordi.gui.data import GuiIndex
from macchina_ricordi.gui.server import WRITE_HEADER, serve


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".dora" / "xps").mkdir(parents=True)
    return tmp_path


class TestWriteGuard:
    @pytest.fixture()
    def base_url(self, repo: Path):
        server = serve(GuiIndex(repo), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()

    def _post(self, url: str, body: dict, headers: dict | None = None) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def test_posts_without_the_header_or_from_another_host_are_refused(self, base_url: str) -> None:
        for headers in ({}, {WRITE_HEADER: "1", "Host": "evil.example:80"}):
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                self._post(f"{base_url}/api/ledger", {"id": "x", "fields": {}, "create": True}, headers)
            assert excinfo.value.code == 403
