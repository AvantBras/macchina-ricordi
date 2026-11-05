from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from macchina_ricordi.cli import main
from macchina_ricordi.data import datasets
from macchina_ricordi.models import record as models
from macchina_ricordi.training import jobs as train_jobs

# Stands in for `python -m macchina_ricordi.cli`: prints the arguments it got, and exits 5 when they mention "boom".
FAKE_PREFIX = (
    sys.executable,
    "-c",
    "import sys; print(' '.join(sys.argv[1:]), flush=True); sys.exit(5 if 'boom' in sys.argv else 0)",
    "macchina_ricordi.cli",
)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for module in (train_jobs, datasets, models):
        monkeypatch.setattr(module, "get_repo_root", lambda explicit_root=None: tmp_path)
    monkeypatch.setenv("AUDIOCRAFT_DORA_DIR", str(tmp_path / ".dora"))
    for split in ("train", "valid"):
        manifest = tmp_path / ".workspace_t" / "egs" / "ds" / split / "data.jsonl.gz"
        manifest.parent.mkdir(parents=True)
        with gzip.open(manifest, "wt") as handle:
            handle.write(json.dumps({"path": "a.wav", "duration": 10.0}) + "\n")
    xp = tmp_path / ".dora" / "xps" / "aaaa1111"
    xp.mkdir(parents=True)
    (xp / "checkpoint_5.th").write_bytes(b"")
    (xp / "history.json").write_text(json.dumps([{"valid": {"ce": 5.0}}] * 5))
    (xp / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps({"effective_label": "warm", "xp_sig": "aaaa1111"})
    )
    return tmp_path


def _add(*argv: str):  # type: ignore[no-untyped-def]
    return train_jobs.add_train_job(["--dataset", ".workspace_t", "--recipe", "reshape-nano", *argv])


def _sleeper(seconds: float = 30.0) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})", "macchina_ricordi.cli"])


class TestAdding:
    def test_a_job_keeps_train_runs_arguments_canonically(self, repo: Path) -> None:
        job = _add("--epochs", "30", "--run-label", "bz deep", "--from", "warm@5", "--override", "optim.lr=5e-5")
        assert job.state == "queued" and job.kind == "train" and job.origin == "cli"
        assert list(job.argv) == ["--recipe", "reshape-nano", "--dataset", ".workspace_t", "--epochs", "30",
                                  "--override", "optim.lr=5e-5", "--run-label", "bz deep", "--from", "warm@5"]  # fmt: skip
        assert job.request["epochs"] == 30 and job.request["init_from"] == "warm@5"
        assert job.describe() == "reshape-nano on .workspace_t from warm@5, 30 epochs"
        assert job.id.endswith("_bz_deep")
        assert job.command(FAKE_PREFIX)[-2:] == ["--job", job.id]
        assert json.loads((job.path / "job.json").read_text())["schema"] == "macchina_ricordi.train_job.v1"


class TestTheWorker:
    def test_runs_queued_jobs_in_order_and_records_how_each_ended(self, repo: Path) -> None:
        ok = _add("--run-label", "one")
        bad = _add("--run-label", "boom")
        resume = train_jobs.add_resume_job("warm", epochs=10)
        lines: list[str] = []
        assert train_jobs.run_queue(prefix=FAKE_PREFIX, poll_seconds=0.05, echo=lines.append) == 1

        states = {job.id: job for job in train_jobs.list_jobs()}
        assert (states[ok.id].state, states[ok.id].status["returncode"]) == ("completed", 0)
        assert (states[bad.id].state, states[bad.id].status["returncode"]) == ("failed", 5)
        assert states[resume.id].state == "completed" and states[resume.id].xp_sig == "aaaa1111"
        assert ok.log_path.read_text().startswith("macchina_ricordi.cli train run --recipe reshape-nano")
        assert "train resume warm --epochs 10" in resume.log_path.read_text()
        assert [line.split()[1] for line in lines if "start" in line] == ["start"] * 3
        assert lines[-1].split()[1:3] == ["completed", resume.id]


class TestStateAndStop:
    def test_a_job_whose_process_died_reads_as_failed(self, repo: Path) -> None:
        process = _sleeper()
        job = train_jobs.create_job("train", request={"dataset": ".workspace_t"}, started_by=process.pid)
        job = train_jobs.update_job(job, status="running", xp_sig="bbbb2222")
        assert job.state == "running" and train_jobs.training_sigs() == {"bbbb2222"}
        process.kill()
        process.wait()
        assert train_jobs.load_job(job.id).state == "failed" and train_jobs.training_sigs() == set()


class TestCli:
    def test_train_run_is_a_job_and_runs_a_queued_one(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from macchina_ricordi.training import plan as training_plan
        from macchina_ricordi.training import run as training_run

        recorded: list[Path] = []
        plan = SimpleNamespace(
            segment_warnings=(), audiocraft_dir=repo, dataset_config_path=repo / "c.yaml", xp_sig="eeee5555",
            xp_folder=repo / ".dora" / "xps" / "eeee5555", effective_label="lbl", init_from_checkpoint=None,
            command=["dora", "run", "optim.epochs=4"],
        )  # fmt: skip
        monkeypatch.setattr(training_plan, "plan_musicgen_training", lambda **kwargs: plan)
        monkeypatch.setattr(
            training_run, "run_musicgen_training", lambda plan, record_dir: recorded.append(record_dir) or 0
        )

        assert main(["train", "run", "--dataset", ".workspace_t"]) == 0
        direct = train_jobs.list_jobs()[-1]
        assert recorded == [direct.path]
        assert (direct.state, direct.xp_sig, direct.status["epochs"], direct.origin) == (
            "completed",
            "eeee5555",
            4,
            "cli",
        )

        queued = _add("--run-label", "q")
        assert main(["train", "run", *queued.argv, "--job", queued.id]) == 0
        assert train_jobs.load_job(queued.id).state == "completed" and recorded[-1] == queued.path
        assert main(["train", "run", *queued.argv, "--job", queued.id]) == 1
        assert "is completed, not queued" in capsys.readouterr().err

        assert main(["train", "run", "--dataset", ".workspace_t", "--dry-run"]) == 0
        assert len(train_jobs.list_jobs()) == 2  # a dry run records nothing
