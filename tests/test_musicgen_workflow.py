from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from macchina_ricordi.audio_files import iter_audio_files
from macchina_ricordi.data.prepare import prepare_musicgen_dataset, split_audio_files
from macchina_ricordi.musicgen.templates import DATASET_CONFIG_NAME, render_and_write_dataset_config
from macchina_ricordi.training.plan import MusicGenTrainingPlan, MusicGenXpReference, plan_musicgen_training
from macchina_ricordi.training.run import run_musicgen_training

# monkeypatch replaces subprocess.run module-wide; lazy imports (librosa -> scipy) also call it.
_real_subprocess_run = subprocess.run


def _write_training_workspace(
    workspace_dir: Path,
    *,
    dataset_name: str = DATASET_CONFIG_NAME,
    profile_name: str = "micro-unconditional",
    train_hash: str = "train-hash",
    valid_hash: str = "valid-hash",
) -> None:
    (workspace_dir / "egs" / dataset_name / "train").mkdir(parents=True)
    (workspace_dir / "egs" / dataset_name / "valid").mkdir(parents=True)
    (workspace_dir / "egs" / dataset_name / "train" / "data.jsonl.gz").write_text("{}", encoding="utf-8")
    (workspace_dir / "egs" / dataset_name / "valid" / "data.jsonl.gz").write_text("{}", encoding="utf-8")
    (workspace_dir / "logs").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "logs" / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": profile_name,
                "dataset_name": dataset_name,
                "train_files": [{"relative_path": "train.wav", "size_bytes": 1, "sha256": train_hash}],
                "valid_files": [{"relative_path": "valid.wav", "size_bytes": 1, "sha256": valid_hash}],
                "train_aggregate_sha256": train_hash,
                "valid_aggregate_sha256": valid_hash,
            }
        ),
        encoding="utf-8",
    )


def test_iter_audio_files_respects_recursive_flag_and_extensions(tmp_path: Path) -> None:
    root = tmp_path / "clips"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "top.wav").write_bytes(b"audio")
    (nested / "deep.flac").write_bytes(b"audio")
    (nested / "ignore.txt").write_text("skip", encoding="utf-8")

    non_recursive = list(iter_audio_files(root, recursive=False, extensions=("wav", "flac")))
    recursive = list(iter_audio_files(root, recursive=True, extensions=("wav", "flac")))

    assert non_recursive == [root / "top.wav"]
    assert recursive == [root / "top.wav", nested / "deep.flac"]


def test_split_audio_files_is_deterministic() -> None:
    files = [Path(f"clip_{index:03d}.wav") for index in range(10)]

    train_first, valid_first = split_audio_files(files, valid_fraction=0.2, seed=42)
    train_second, valid_second = split_audio_files(files, valid_fraction=0.2, seed=42)

    assert train_first == train_second
    assert valid_first == valid_second
    assert len(train_first) == 8
    assert len(valid_first) == 2


def test_prepare_musicgen_dataset_copies_files_writes_sidecars_and_runs_manifests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    (input_dir / "set_a").mkdir(parents=True)
    (input_dir / "set_b").mkdir(parents=True)
    for relative_path in [
        Path("set_a/clip_001.wav"),
        Path("set_a/clip_002.wav"),
        Path("set_b/clip_003.wav"),
    ]:
        full_path = input_dir / relative_path
        full_path.write_bytes(b"fake audio")

    commands: list[list[str]] = []

    def fake_run(command, check, cwd=None, env=None):  # type: ignore[no-untyped-def]
        commands.append(list(command))
        manifest_path = Path(command[-1])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Emulate the AudioCraft manifest builder: a gzip JSONL with per-clip
        # durations. 10s clips so the recorded slice length is deterministic.
        import gzip as _gzip

        with _gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
            for index in range(3):
                handle.write(json.dumps({"path": f"clip_{index}.wav", "duration": 10.0}) + "\n")
        return subprocess.CompletedProcess(command, 0)

    python_exe = tmp_path / "env" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr("macchina_ricordi.data.prepare.subprocess.run", fake_run)

    prepared = prepare_musicgen_dataset(
        input_dir=input_dir,
        workspace_dir=tmp_path / "workspace",
        musicgen_python=str(python_exe),
    )

    copied_audio = sorted(prepared.workspace_dir.glob("data/**/*.wav"))
    copied_sidecars = sorted(prepared.workspace_dir.glob("data/**/*.json"))

    assert prepared.train_count == 2
    assert prepared.valid_count == 1
    assert len(copied_audio) == 3
    assert copied_sidecars == []
    assert len(commands) == 2
    assert prepared.train_manifest.exists()
    assert prepared.valid_manifest.exists()

    prepare_log = prepared.workspace_dir / "logs" / "prepare_dataset.json"
    assert prepare_log.exists()
    prepare_payload = json.loads(prepare_log.read_text(encoding="utf-8"))
    assert "profile" not in prepare_payload
    assert prepare_payload["dataset_name"] == DATASET_CONFIG_NAME
    assert prepare_payload["clip_duration_stats"]["slice_length_s"] == 10.0
    assert prepare_payload["clip_duration_stats"]["median"] == 10.0
    assert prepared.dataset_fingerprint_path.exists()
    fingerprint = json.loads(prepared.dataset_fingerprint_path.read_text(encoding="utf-8"))
    assert fingerprint["profile"] is None
    assert fingerprint["dataset_name"] == DATASET_CONFIG_NAME
    assert fingerprint["train_aggregate_sha256"]
    assert fingerprint["valid_aggregate_sha256"]
    assert fingerprint["clip_duration_stats"]["slice_length_s"] == 10.0


def _write_wav_bytes(path: Path, waveform_float: list[float], sr: int) -> None:
    import wave

    import numpy as np

    arr = np.asarray(waveform_float, dtype=np.float32)
    clipped = np.clip(arr, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm16.tobytes())


def _write_loud_sine_wav(path: Path, *, duration_seconds: float = 5.0, sr: int = 8_000) -> None:
    import numpy as np

    t = np.arange(0, duration_seconds, 1.0 / sr, dtype=np.float32)
    waveform = 0.5 * np.sin(2.0 * np.pi * 220.0 * t)
    _write_wav_bytes(path, waveform.tolist(), sr)


def _write_silent_wav(path: Path, *, duration_seconds: float = 5.0, sr: int = 8_000) -> None:
    import numpy as np

    waveform = np.zeros(int(duration_seconds * sr), dtype=np.float32)
    _write_wav_bytes(path, waveform.tolist(), sr)


def test_prepare_musicgen_dataset_drops_mostly_silent_files(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    loud_paths = [input_dir / f"loud_{idx:02d}.wav" for idx in range(4)]
    silent_paths = [input_dir / f"silent_{idx:02d}.wav" for idx in range(2)]
    for path in loud_paths:
        _write_loud_sine_wav(path)
    for path in silent_paths:
        _write_silent_wav(path)

    def fake_run(command, check=False, cwd=None, env=None, **kwargs):  # type: ignore[no-untyped-def]
        if "audiocraft.data.audio_dataset" not in command:
            return _real_subprocess_run(command, check=check, cwd=cwd, env=env, **kwargs)
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    python_exe = tmp_path / "env" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr("macchina_ricordi.data.prepare.subprocess.run", fake_run)

    prepared = prepare_musicgen_dataset(
        input_dir=input_dir,
        workspace_dir=tmp_path / "workspace",
        musicgen_python=str(python_exe),
        valid_fraction=0.25,
        seed=0,
    )

    copied_audio = sorted(prepared.workspace_dir.glob("data/**/*.wav"))
    assert len(copied_audio) == 4
    assert prepared.train_count + prepared.valid_count == 4
    assert all("silent_" not in path.name for path in copied_audio)

    prepare_log = json.loads((prepared.workspace_dir / "logs" / "prepare_dataset.json").read_text(encoding="utf-8"))
    assert prepare_log["train_dropped_silent"] + prepare_log["valid_dropped_silent"] == 2
    assert prepare_log["max_silence_fraction"] == 0.85
    assert prepare_log["silence_frame_db"] == -35.0


def test_prepare_musicgen_dataset_writes_stable_fingerprint_without_absolute_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    (input_dir / "set_a").mkdir(parents=True)
    (input_dir / "set_b").mkdir(parents=True)
    for relative_path in [
        Path("set_a/clip_001.wav"),
        Path("set_a/clip_002.wav"),
        Path("set_b/clip_003.wav"),
    ]:
        (input_dir / relative_path).write_bytes(b"fake audio")

    def fake_run(command, check=False, cwd=None, env=None, **kwargs):  # type: ignore[no-untyped-def]
        if "audiocraft.data.audio_dataset" not in command:
            return _real_subprocess_run(command, check=check, cwd=cwd, env=env, **kwargs)
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    python_exe = tmp_path / "env" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr("macchina_ricordi.data.prepare.subprocess.run", fake_run)

    first = prepare_musicgen_dataset(
        input_dir=input_dir,
        workspace_dir=tmp_path / "workspace_one",
        musicgen_python=str(python_exe),
    )
    second = prepare_musicgen_dataset(
        input_dir=input_dir,
        workspace_dir=tmp_path / "workspace_two",
        musicgen_python=str(python_exe),
    )

    first_fingerprint = json.loads(first.dataset_fingerprint_path.read_text(encoding="utf-8"))
    second_fingerprint = json.loads(second.dataset_fingerprint_path.read_text(encoding="utf-8"))

    assert first_fingerprint == second_fingerprint
    assert all(
        str(tmp_path).replace("\\", "/") not in entry["relative_path"]
        for entry in [*first_fingerprint["train_files"], *first_fingerprint["valid_files"]]
    )


def test_render_and_write_dataset_config_uses_workspace_manifests(tmp_path: Path) -> None:
    audiocraft_dir = tmp_path / "audiocraft"
    workspace_dir = tmp_path / "workspace"
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "train").mkdir(parents=True)
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "valid").mkdir(parents=True)

    config_path = render_and_write_dataset_config(
        audiocraft_dir=audiocraft_dir,
        workspace_dir=workspace_dir,
    )

    payload = config_path.read_text(encoding="utf-8")
    assert config_path.name == f"{DATASET_CONFIG_NAME}.yaml"
    assert workspace_dir.joinpath("egs", DATASET_CONFIG_NAME, "train").as_posix() in payload
    assert workspace_dir.joinpath("egs", DATASET_CONFIG_NAME, "valid").as_posix() in payload


def test_plan_musicgen_training_builds_default_dora_command(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    audiocraft_dir = tmp_path / "audiocraft"
    python_dir = tmp_path / "env"
    python_exe = python_dir / "python.exe"
    dora_exe = python_dir / "dora.exe"

    _write_training_workspace(workspace_dir)
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")
    dora_exe.write_text("", encoding="utf-8")

    plan = plan_musicgen_training(
        audiocraft_dir=str(audiocraft_dir),
        workspace_dir=workspace_dir,
        musicgen_python=str(python_exe),
        overrides=["optim.epochs=3"],
        dry_run=True,
    )

    assert plan.dry_run is True
    assert plan.dataset_config_path.exists()
    assert plan.command[0] == str(dora_exe)
    assert "solver=musicgen/musicgen_base_32khz" in plan.command
    assert "optim.epochs=3" in plan.command
    assert plan.plan_record["command"] == plan.command
    assert not (workspace_dir / "logs" / "train_plan.json").exists()  # a plan is recorded by its run


def test_planning_leaves_older_run_records_alone(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    audiocraft_dir = tmp_path / "audiocraft"
    python_dir = tmp_path / "env"
    python_exe = python_dir / "python.exe"

    _write_training_workspace(workspace_dir)
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")
    (python_dir / "dora.exe").write_text("", encoding="utf-8")
    previous = workspace_dir / "logs" / "train_plan.json"
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_text('{"the": "last real launch"}', encoding="utf-8")

    plan = plan_musicgen_training(
        audiocraft_dir=str(audiocraft_dir),
        workspace_dir=workspace_dir,
        musicgen_python=str(python_exe),
        overrides=["optim.epochs=3"],
        dry_run=True,
    )

    assert "optim.epochs=3" in plan.command
    assert previous.read_text(encoding="utf-8") == '{"the": "last real launch"}'


def _setup_segment_guard_plan(
    tmp_path: Path,
    *,
    clip_durations: list[float],
    dataset_name: str = DATASET_CONFIG_NAME,
) -> tuple[Path, Path, Path]:
    """Build a plan-ready workspace whose train manifest has real gz durations."""
    import gzip as _gzip

    workspace_dir = tmp_path / "workspace"
    audiocraft_dir = tmp_path / "audiocraft"
    python_dir = tmp_path / "env"
    python_exe = python_dir / "python.exe"
    dora_exe = python_dir / "dora.exe"

    _write_training_workspace(workspace_dir, dataset_name=dataset_name)
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")
    dora_exe.write_text("", encoding="utf-8")

    manifest = workspace_dir / "egs" / dataset_name / "train" / "data.jsonl.gz"
    with _gzip.open(manifest, "wt", encoding="utf-8") as handle:
        for index, duration in enumerate(clip_durations):
            handle.write(json.dumps({"path": f"clip_{index}.wav", "duration": duration}) + "\n")

    return workspace_dir, audiocraft_dir, python_exe


def test_plan_musicgen_training_blocks_segment_duration_over_slice(tmp_path: Path) -> None:
    # micro-unconditional declares dataset.segment_duration=15; 10s clips => mismatch.
    workspace_dir, audiocraft_dir, python_exe = _setup_segment_guard_plan(tmp_path, clip_durations=[10.0] * 40)

    from macchina_ricordi.data.clip_stats import SegmentDurationMismatch

    with pytest.raises(SegmentDurationMismatch) as excinfo:
        plan_musicgen_training(
            audiocraft_dir=str(audiocraft_dir),
            workspace_dir=workspace_dir,
            profile_name="micro-unconditional",
            musicgen_python=str(python_exe),
            dry_run=True,
        )
    assert excinfo.value.recommended_segment_duration == 10
    assert "segment_duration=10" in str(excinfo.value)


def test_plan_musicgen_training_adds_auto_dataset_label(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "midwife_workspace"
    audiocraft_dir = tmp_path / "audiocraft"
    python_dir = tmp_path / "env"
    python_exe = python_dir / "python.exe"
    dora_exe = python_dir / "dora.exe"

    _write_training_workspace(workspace_dir, train_hash="aaa", valid_hash="bbb")
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")
    dora_exe.write_text("", encoding="utf-8")

    plan = plan_musicgen_training(
        audiocraft_dir=str(audiocraft_dir),
        workspace_dir=workspace_dir,
        musicgen_python=str(python_exe),
        dry_run=True,
    )

    labels = [override for override in plan.command if override.startswith("label=")]
    assert labels == [f"label=nr_midwife_workspace_{plan.dataset_fingerprint_digest}"]
    assert plan.auto_dataset_label is True
    assert plan.effective_label == f"nr_midwife_workspace_{plan.dataset_fingerprint_digest}"
    assert plan.xp_sig
    assert plan.xp_folder is not None
    persisted_plan = plan.plan_record
    assert persisted_plan["dataset_fingerprint_digest"] == plan.dataset_fingerprint_digest
    assert persisted_plan["xp_sig"] == plan.xp_sig


def test_plan_musicgen_training_rejects_existing_xp_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    audiocraft_dir = tmp_path / "audiocraft"
    python_dir = tmp_path / "env"
    python_exe = python_dir / "python.exe"
    dora_exe = python_dir / "dora.exe"
    xp_folder = tmp_path / "xps" / "abcd1234"
    _write_training_workspace(workspace_dir)
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)
    python_dir.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")
    dora_exe.write_text("", encoding="utf-8")
    xp_folder.mkdir(parents=True)
    (xp_folder / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps(
            {
                "profile": "other-profile",
                "dataset_name": DATASET_CONFIG_NAME,
                "workspace_dir": str(workspace_dir),
                "dataset_fingerprint_digest": "old",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.plan.resolve_training_xp_reference",
        lambda **kwargs: MusicGenXpReference(sig="abcd1234", folder=xp_folder),
    )

    with pytest.raises(RuntimeError, match="Refusing to reuse existing Dora XP"):
        plan_musicgen_training(
            audiocraft_dir=str(audiocraft_dir),
            workspace_dir=workspace_dir,
            musicgen_python=str(python_exe),
            dry_run=True,
        )


def test_run_musicgen_training_writes_failure_status(monkeypatch, tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    audiocraft_dir = tmp_path / "audiocraft"
    dataset_config = audiocraft_dir / "config" / "dset" / "audio" / "test.yaml"
    dataset_config.parent.mkdir(parents=True, exist_ok=True)
    dataset_config.write_text("{}", encoding="utf-8")
    (workspace_dir / "logs").mkdir(parents=True, exist_ok=True)

    def fake_run(command, check, cwd=None, env=None):  # type: ignore[no-untyped-def]
        raise subprocess.CalledProcessError(returncode=7, cmd=command)

    monkeypatch.setattr("macchina_ricordi.training.run.subprocess.run", fake_run)

    runtime_plan = MusicGenTrainingPlan(
        workspace_dir=workspace_dir,
        audiocraft_dir=audiocraft_dir,
        dataset_config_path=dataset_config,
        command=["dora", "run", "solver=test"],
        dry_run=False,
    )

    try:
        run_musicgen_training(runtime_plan, record_dir=tmp_path / "job")
    except RuntimeError as exc:
        assert "train_run_status.json" in str(exc)
    else:
        raise AssertionError("Expected run_musicgen_training to raise RuntimeError")

    status_path = tmp_path / "job" / "train_run_status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["returncode"] == 7
    assert payload["command"] == ["dora", "run", "solver=test"]


def test_run_musicgen_training_writes_xp_metadata(monkeypatch, tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    audiocraft_dir = tmp_path / "audiocraft"
    xp_folder = tmp_path / "xps" / "abcd1234"
    dataset_config = audiocraft_dir / "config" / "dset" / "audio" / "test.yaml"
    dataset_config.parent.mkdir(parents=True, exist_ok=True)
    dataset_config.write_text("{}", encoding="utf-8")
    (workspace_dir / "logs").mkdir(parents=True, exist_ok=True)

    def fake_run(command, check, cwd=None, env=None):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("macchina_ricordi.training.run.subprocess.run", fake_run)

    runtime_plan = MusicGenTrainingPlan(
        workspace_dir=workspace_dir,
        audiocraft_dir=audiocraft_dir,
        dataset_config_path=dataset_config,
        command=["dora", "run", "solver=test"],
        dry_run=False,
        xp_sig="abcd1234",
        xp_folder=xp_folder,
        train_metadata_path=xp_folder / "macchina_ricordi_train_metadata.json",
        train_metadata={
            "profile": "micro-unconditional",
            "dataset_name": DATASET_CONFIG_NAME,
            "workspace_dir": str(workspace_dir),
            "dataset_fingerprint_digest": "digest",
        },
    )

    assert run_musicgen_training(runtime_plan, record_dir=tmp_path / "job") == 0

    metadata = json.loads((xp_folder / "macchina_ricordi_train_metadata.json").read_text(encoding="utf-8"))
    assert metadata["profile"] == "micro-unconditional"
    assert metadata["dataset_fingerprint_digest"] == "digest"
    assert metadata["written_at"]
