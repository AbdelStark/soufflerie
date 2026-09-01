from __future__ import annotations

from pathlib import Path

from scripts.check_ci_policy import (
    check_ci_policy,
    check_pre_commit_policy,
    check_workflow_policy,
    find_unmarked_remote_tests,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_checked_in_ci_policy_is_complete_and_least_privilege() -> None:
    assert check_ci_policy(PROJECT_ROOT) == ()


def test_policy_rejects_floating_actions_and_pull_request_secrets(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    unsafe = source.replace(
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/checkout@v7",
        1,
    ).replace("permissions:\n  contents: read", "permissions:\n  contents: write", 1)
    unsafe += "\n# forbidden: ${{ secrets.REMOTE_TOKEN }}\n"
    workflow = tmp_path / "ci.yml"
    workflow.write_text(unsafe, encoding="utf-8")

    errors = check_workflow_policy(workflow)
    assert any("unpinned action" in error for error in errors)
    assert any("permissions must be exactly contents: read" in error for error in errors)
    assert any("must not reference" in error for error in errors)


def test_policy_requires_solver_profile_for_cpu_solver_jobs(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    incomplete = source.replace("uv sync --frozen --extra solver", "uv sync --frozen", 1)
    workflow = tmp_path / "ci.yml"
    workflow.write_text(incomplete, encoding="utf-8")

    errors = check_workflow_policy(workflow)
    assert any("must sync the locked solver extra" in error for error in errors)


def test_policy_rejects_unmarked_remote_sdk_and_subprocess_tests(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_remote_sdk.py").write_text(
        """\
import modal

def test_remote_call():
    modal.App.lookup("untrusted")
""",
        encoding="utf-8",
    )
    (tests_root / "test_remote_command.py").write_text(
        """\
import subprocess

def test_remote_command():
    subprocess.run(["modal", "run", "infra/app.py"], check=True)
""",
        encoding="utf-8",
    )
    errors = find_unmarked_remote_tests(tests_root)
    assert len(errors) == 2
    assert any("network/remote call" in error for error in errors)
    assert any("remote subprocess command" in error for error in errors)

    (tests_root / "test_remote_sdk.py").write_text(
        """\
import modal
import pytest

pytestmark = pytest.mark.remote

def test_remote_call():
    modal.App.lookup("reviewed")
""",
        encoding="utf-8",
    )
    (tests_root / "test_remote_command.py").write_text(
        """\
import subprocess
import pytest

@pytest.mark.remote
def test_remote_command():
    subprocess.run(["modal", "run", "infra/app.py"], check=True)
""",
        encoding="utf-8",
    )
    assert find_unmarked_remote_tests(tests_root) == ()


def test_policy_rejects_remote_capability_hidden_in_a_fixture(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_fixture.py").write_text(
        """\
import pytest
import requests

@pytest.fixture
def remote_payload():
    return requests.get("https://example.invalid")

def test_payload(remote_payload):
    assert remote_payload
""",
        encoding="utf-8",
    )

    errors = find_unmarked_remote_tests(tests_root)
    assert len(errors) == 1
    assert "helper or fixture remote_payload" in errors[0]


def test_pre_commit_policy_rejects_mutating_or_missing_gates(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    unsafe = source.replace("uv run ruff check", "uv run ruff check --fix", 1).replace(
        "      - id: ci-policy\n", "", 1
    )
    config = tmp_path / ".pre-commit-config.yaml"
    config.write_text(unsafe, encoding="utf-8")
    errors = check_pre_commit_policy(config)
    assert any("must not mutate" in error for error in errors)
    assert any("ci-policy" in error and "omits required hooks" in error for error in errors)
