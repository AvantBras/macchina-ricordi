from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from macchina_ricordi.listening.archive import (
    ARCHIVE_SCHEMA_VERSION,
    build_archive_index,
    build_archive_portrait,
    build_novelty_report,
    build_similar_report,
    load_archive_index,
    render_archive_index_json,
)
from macchina_ricordi.listening.semantic import DEFAULT_SEMANTIC_PROBES


def _write_wav(path: Path, waveform: np.ndarray, *, sr: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sr)


def _clicks(*, duration: float = 2.0, sr: int = 16_000, offset: float = 0.2) -> np.ndarray:
    y = np.zeros(int(duration * sr), dtype=np.float32)
    click = np.hanning(64).astype(np.float32)
    for onset in np.arange(offset, duration - 0.1, 0.25):
        idx = int(onset * sr)
        y[idx : idx + len(click)] += click
    return y


def _noise(*, duration: float = 2.0, sr: int = 16_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.25, int(duration * sr)).astype(np.float32)


def _fast_kwargs() -> dict[str, object]:
    return {
        "sr": 16_000,
        "res_type": "soxr_qq",
        "window_seconds": 1.0,
        "window_hop_seconds": 1.0,
    }


class FakeSemanticBackend:
    name = "fake"
    version = "test"
    device = "cpu"
    model_path = None

    def score_audio_text(self, audio_path: str | Path, prompts: list[str]) -> list[float]:
        prompt_to_key = {definition.prompt: definition.key for definition in DEFAULT_SEMANTIC_PROBES}
        return [1.0 if prompt_to_key[prompt] == "metallic_scrape" else 0.0 for prompt in prompts]


class FakeCaptionerBackend:
    name = "fake-captioner"
    model_id = "fake/model"
    device = "cpu"
    dtype = "float32"
    quantization = "none"
    trust_remote_code = False

    def generate_caption(
        self,
        audio_path: str | Path,
        *,
        prompt: str,
        max_new_tokens: int,
        max_audio_seconds: float,
    ) -> str:
        return "A compact witness note about brittle clicks and unstable texture."


def test_archive_index_schema_and_top_level_keys(tmp_path: Path) -> None:
    _write_wav(tmp_path / "a.wav", _clicks())

    index = build_archive_index(tmp_path, **_fast_kwargs())
    payload = index.to_dict()

    assert list(payload) == ["schema", "analysis", "root", "entries", "groups", "warnings"]
    assert payload["schema"] == ARCHIVE_SCHEMA_VERSION
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["novelty_score"] == 1.0


def test_archive_index_includes_three_files_and_neighbors(tmp_path: Path) -> None:
    click_a = tmp_path / "click_a.wav"
    click_b = tmp_path / "click_b.wav"
    noise = tmp_path / "noise.wav"
    _write_wav(click_a, _clicks(offset=0.2))
    _write_wav(click_b, _clicks(offset=0.22))
    _write_wav(noise, _noise())

    index = build_archive_index(tmp_path, neighbors=2, **_fast_kwargs())

    assert [Path(entry.path).name for entry in index.entries] == ["click_a.wav", "click_b.wav", "noise.wav"]
    by_name = {Path(entry.path).name: entry for entry in index.entries}
    assert by_name["click_a.wav"].neighbors[0].path == str(click_b.resolve())
    assert by_name["click_a.wav"].neighbors[0].similarity > by_name["click_a.wav"].neighbors[1].similarity
    assert by_name["noise.wav"].novelty_score is not None


def test_archive_index_creates_missing_sidecars_by_default(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    _write_wav(audio, _clicks())

    index = build_archive_index(tmp_path, **_fast_kwargs())

    assert (tmp_path / "clip_listening_report.json").exists()
    assert (tmp_path / "clip_listening.md").exists()
    assert index.entries[0].report_path == str((tmp_path / "clip_listening_report.json").resolve())


def test_archive_index_with_semantic_probes_enriches_sidecars(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    _write_wav(audio, _clicks())

    index = build_archive_index(
        tmp_path,
        semantic_probes=True,
        semantic_probe_backend=FakeSemanticBackend(),
        **_fast_kwargs(),
    )
    payload = json.loads((tmp_path / "clip_listening_report.json").read_text(encoding="utf-8"))

    assert payload["semantic_probes"]["strong"][0]["key"] == "metallic_scrape"
    assert index.entries[0].semantic is not None
    assert index.entries[0].semantic["strong"] == ["metallic_scrape"]


def test_archive_index_with_captioner_witness_enriches_sidecars(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    _write_wav(audio, _clicks())

    index = build_archive_index(
        tmp_path,
        captioner_witness=True,
        captioner_backend_instance=FakeCaptionerBackend(),
        **_fast_kwargs(),
    )
    payload = json.loads((tmp_path / "clip_listening_report.json").read_text(encoding="utf-8"))

    assert payload["captioner_witness"]["backend"]["name"] == "fake-captioner"
    assert index.entries[0].captioner is not None
    assert index.entries[0].captioner["summary"].startswith("A compact witness")


def test_archive_read_only_skips_missing_sidecars(tmp_path: Path) -> None:
    _write_wav(tmp_path / "clip.wav", _clicks())

    index = build_archive_index(tmp_path, read_only=True, **_fast_kwargs())

    assert index.entries == []
    assert "missing listening sidecar" in index.warnings[0]


def test_archive_refresh_overwrites_stale_sidecar(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    report = tmp_path / "clip_listening_report.json"
    _write_wav(audio, _clicks())
    report.write_text(
        json.dumps(
            {
                "summary": {"global_summary": "stale"},
                "audio": {"duration_seconds": 1.0},
                "whole_clip": {"reward": {"score": 0.0}, "description": {}},
            }
        ),
        encoding="utf-8",
    )

    stale = build_archive_index(tmp_path, read_only=True, **_fast_kwargs())
    refreshed = build_archive_index(tmp_path, refresh=True, **_fast_kwargs())

    assert stale.entries[0].summary == "stale"
    assert refreshed.entries[0].summary != "stale"


def test_archive_novelty_ranks_outlier_highest(tmp_path: Path) -> None:
    _write_wav(tmp_path / "click_a.wav", _clicks(offset=0.2))
    _write_wav(tmp_path / "click_b.wav", _clicks(offset=0.22))
    _write_wav(tmp_path / "noise.wav", _noise())

    index = build_archive_index(tmp_path, neighbors=2, **_fast_kwargs())
    novelty = build_novelty_report(index, limit=1)

    assert Path(novelty["entries"][0]["path"]).name == "noise.wav"


def test_archive_corrupt_audio_warns_without_aborting(tmp_path: Path) -> None:
    _write_wav(tmp_path / "good.wav", _clicks())
    (tmp_path / "bad.wav").write_bytes(b"not audio")

    index = build_archive_index(tmp_path, **_fast_kwargs())

    assert len(index.entries) == 2
    assert any(entry.error for entry in index.entries)
    assert any("bad.wav" in warning for warning in index.warnings)


def test_archive_json_round_trips(tmp_path: Path) -> None:
    _write_wav(tmp_path / "clip.wav", _clicks())
    output = tmp_path / "index.json"

    index = build_archive_index(tmp_path, output_path=output, **_fast_kwargs())
    payload = json.loads(render_archive_index_json(index))
    loaded = load_archive_index(output)

    assert payload["schema"] == ARCHIVE_SCHEMA_VERSION
    assert loaded.schema == ARCHIVE_SCHEMA_VERSION
    assert loaded.entries[0].path == index.entries[0].path


def test_archive_musicgen_metadata_is_associated(tmp_path: Path) -> None:
    gen_dir = tmp_path / "artifacts" / "generated_audio"
    audio = gen_dir / "sample.wav"
    _write_wav(audio, _clicks())
    (gen_dir / "musicgen_20260516T000000Z.json").write_text(
        json.dumps(
            {
                "run_id": "20260516T000000Z",
                "request": {"temperature": 1.5, "top_k": 80, "top_p": 0.0, "cfg_coef": 3.0},
                "generation": {
                    "source": {
                        "xp_path": str(tmp_path / ".dora" / "xps" / "abc123"),
                        "checkpoint_path": str(tmp_path / ".dora" / "xps" / "abc123" / "checkpoint.th"),
                    }
                },
                "model": {"label": "demo", "workspace": ".ws"},
                "files": [str(audio)],
            }
        ),
        encoding="utf-8",
    )

    index = build_archive_index(gen_dir, repo_root=tmp_path, **_fast_kwargs())

    assert index.entries[0].musicgen is not None
    assert index.entries[0].musicgen["run_id"] == "20260516T000000Z"
    assert index.entries[0].musicgen["xp_path"].endswith("abc123")
    assert index.groups[0]["count"] == 1


def test_archive_new_run_metadata_is_associated(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "runs" / "2026-05-16" / "musicgen_20260516T000000Z"
    audio = run_dir / "samples" / "000.wav"
    _write_wav(audio, _clicks())
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "20260516T000000Z",
                "request": {"temperature": 1.5, "top_k": 80, "top_p": 0.0, "cfg_coef": 3.0},
                "generation": {"source": {"xp_path": str(tmp_path / ".dora" / "xps" / "abc123")}},
                "model": {"label": "demo", "workspace": ".ws"},
                "files": ["samples/000.wav"],
            }
        ),
        encoding="utf-8",
    )

    index = build_archive_index(run_dir, repo_root=tmp_path, **_fast_kwargs())

    assert index.entries[0].musicgen is not None
    assert index.entries[0].musicgen["run_id"] == "20260516T000000Z"
    assert index.entries[0].musicgen["model_label"] == "demo"


def test_archive_sweep_metadata_is_associated(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "artifacts" / "generated_audio" / "sweeps" / "demo"
    audio = sweep_dir / "combo_000" / "sample_000.wav"
    _write_wav(audio, _clicks())
    (sweep_dir / "index.csv").write_text(
        "\n".join(
            [
                "sweep_id,combo_id,combo_index,sample_index,path,xp_path,checkpoint_path,temperature,top_k,top_p,cfg_coef",
                f"demo,combo_000,0,0,{audio},{tmp_path / '.dora' / 'xps' / 'abc123'},ckpt,1.2,250,0.0,3.0",
            ]
        ),
        encoding="utf-8",
    )

    index = build_archive_index(sweep_dir, repo_root=tmp_path, **_fast_kwargs())

    assert index.entries[0].musicgen is not None
    assert index.entries[0].musicgen["sweep_id"] == "demo"
    assert index.entries[0].musicgen["combo_id"] == "combo_000"


def test_similar_report_uses_index_neighbors(tmp_path: Path) -> None:
    target = tmp_path / "click_a.wav"
    neighbor = tmp_path / "click_b.wav"
    _write_wav(target, _clicks(offset=0.2))
    _write_wav(neighbor, _clicks(offset=0.22))
    _write_wav(tmp_path / "noise.wav", _noise())
    index = build_archive_index(tmp_path, neighbors=2, **_fast_kwargs())

    report = build_similar_report(target, tmp_path, index=index, neighbors=1)

    assert report["target"] == str(target.resolve())
    assert report["neighbors"][0]["path"] == str(neighbor.resolve())


def test_portrait_groups_by_xp_path_and_signature(tmp_path: Path) -> None:
    gen_dir = tmp_path / "artifacts" / "generated_audio"
    xp_path = tmp_path / ".dora" / "xps" / "abc123"
    audio_a = gen_dir / "a.wav"
    audio_b = gen_dir / "b.wav"
    _write_wav(audio_a, _clicks(offset=0.2))
    _write_wav(audio_b, _clicks(offset=0.24))
    (gen_dir / "musicgen_20260516T000000Z.json").write_text(
        json.dumps(
            {
                "run_id": "20260516T000000Z",
                "request": {"temperature": 1.0, "top_k": 250, "top_p": 0.0, "cfg_coef": 3.0},
                "generation": {"source": {"xp_path": str(xp_path), "checkpoint_path": str(xp_path / "checkpoint.th")}},
                "files": [str(audio_a), str(audio_b)],
            }
        ),
        encoding="utf-8",
    )

    by_path = build_archive_portrait(str(xp_path), repo_root=tmp_path, **_fast_kwargs())
    by_sig = build_archive_portrait("abc123", repo_root=tmp_path, **_fast_kwargs())

    assert by_path.summary["sample_count"] == 2
    assert by_sig.summary["sample_count"] == 2
    assert by_sig.summary["representative_samples"]


def test_portrait_warns_when_no_audio_matches(tmp_path: Path) -> None:
    portrait = build_archive_portrait("missing", repo_root=tmp_path, **_fast_kwargs())

    assert portrait.summary["sample_count"] == 0
    assert portrait.warnings


@pytest.mark.parametrize("neighbors", [0, 1])
def test_archive_neighbor_count_is_respected(tmp_path: Path, neighbors: int) -> None:
    _write_wav(tmp_path / "a.wav", _clicks(offset=0.2))
    _write_wav(tmp_path / "b.wav", _clicks(offset=0.22))

    index = build_archive_index(tmp_path, neighbors=neighbors, **_fast_kwargs())

    assert all(len(entry.neighbors) == neighbors for entry in index.entries)
