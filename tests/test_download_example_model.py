from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path
from urllib.parse import urlparse

import pytest

from scripts import download_example_model as downloader


@pytest.fixture()
def files(monkeypatch):
    data = {name: f"test fixture: {name}".encode() for name in downloader.SHA256}
    monkeypatch.setattr(downloader, "SHA256", {name: hashlib.sha256(value).hexdigest() for name, value in data.items()})
    calls = []

    def fetch(request, *, timeout):
        name = Path(urlparse(request.full_url).path).name
        calls.append(name)
        assert downloader.REVISION in request.full_url
        return io.BytesIO(data[name])

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fetch)
    return data, calls


def test_download_checks_all_files_and_is_idempotent(tmp_path, files):
    data, calls = files
    output = tmp_path / "models" / "example"
    assert downloader.download(output) == output
    assert {path.name: path.read_bytes() for path in output.iterdir()} == data
    assert len(calls) == 4
    assert downloader.download(output) == output
    assert len(calls) == 4


def test_mismatched_download_is_not_installed(tmp_path, files):
    data, _ = files
    data["model.safetensors"] = b"corrupt transfer"
    output = tmp_path / "models" / "example"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader.download(output)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_different_existing_files_are_never_overwritten(tmp_path, files):
    _, calls = files
    output = tmp_path / "example"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"user weights")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader.download(output)
    assert (output / "model.safetensors").read_bytes() == b"user weights"
    assert calls == []


def test_incomplete_existing_folder_is_not_overwritten(tmp_path, files):
    _, calls = files
    output = tmp_path / "example"
    output.mkdir()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        downloader.download(output)
    assert calls == []


def test_symlink_destination_is_refused(tmp_path, files):
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "example"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink destination"):
        downloader.download(output)
    assert list(target.iterdir()) == []


def test_failed_network_transfer_cleans_staging(tmp_path, files, monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("test connection failure")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fail)
    output = tmp_path / "models" / "example"
    with pytest.raises(urllib.error.URLError):
        downloader.download(output)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []
