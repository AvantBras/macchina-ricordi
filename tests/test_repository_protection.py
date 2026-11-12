"""Validate prepared protection payloads; these tests do not enable GitHub settings."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_main_protection_keeps_the_solo_maintainer_pr_path() -> None:
    policy = json.loads((ROOT / ".github/main-protection.json").read_text())
    assert policy["enforce_admins"] is True
    assert policy["allow_force_pushes"] is False
    assert policy["allow_deletions"] is False
    assert policy["lock_branch"] is False
    assert policy["required_conversation_resolution"] is True
    reviews = policy["required_pull_request_reviews"]
    assert reviews["required_approving_review_count"] == 0
    assert reviews["require_code_owner_reviews"] is False
    assert reviews["require_last_push_approval"] is False


def test_protection_requires_the_actual_cpu_checks_not_the_gpu_advisory() -> None:
    policy = json.loads((ROOT / ".github/main-protection.json").read_text())
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    test_job = jobs["tests"]
    expected = {
        test_job["name"].replace("${{ matrix.python }}", version)
        for version in test_job["strategy"]["matrix"]["python"]
    }
    expected.add(jobs["checks"]["name"])
    status = policy["required_status_checks"]
    assert status["strict"] is True
    checks = status["checks"]
    assert len(checks) == len(expected)
    assert {check["context"] for check in checks} == expected
    assert all(check["app_id"] == 15368 for check in checks)
    assert jobs["musicgen-lock"]["name"] not in expected
    assert jobs["musicgen-lock"]["continue-on-error"] is True


def test_external_fork_workflows_require_maintainer_approval() -> None:
    policy = json.loads((ROOT / ".github/fork-pr-approval.json").read_text())
    assert policy == {"approval_policy": "all_external_contributors"}
