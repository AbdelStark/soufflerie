from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from scripts.check_governance import check_governance

PROJECT_ROOT = Path(__file__).parents[2]
OWNER = "AbdelStark"
GOVERNANCE_FILES = {
    "LICENSE",
    "NOTICE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CITATION.cff",
    "MAINTAINERS.md",
}


def test_governance_contract_and_cli_pass() -> None:
    assert check_governance(PROJECT_ROOT) == ()
    result = subprocess.run(
        [sys.executable, "scripts/check_governance.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == "governance contracts: ok\n"
    assert result.stderr == ""


def test_governance_checker_fails_closed_when_a_required_policy_is_missing(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    shutil.copytree(
        PROJECT_ROOT,
        checkout,
        ignore=shutil.ignore_patterns(
            ".agents",
            ".git",
            ".hypothesis",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "dist",
        ),
    )
    (checkout / "SECURITY.md").unlink()
    assert check_governance(checkout) == ("missing required governance file: SECURITY.md",)


def test_license_citation_and_ownership_metadata_agree() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    citation = yaml.safe_load((PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert project["license"] == citation["license"] == "Apache-2.0"
    assert project["version"] == citation["version"] == "0.1.0"
    assert project["authors"] == citation["authors"] == [{"name": OWNER}]
    assert project["license-files"] == ["LICENSE", "NOTICE"]

    codeowners = (PROJECT_ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
    active_lines = [
        line.split()
        for line in codeowners.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert active_lines
    assert all(words[-1] == f"@{OWNER}" for words in active_lines)


@pytest.mark.integration
def test_source_distribution_contains_governance_and_templates(tmp_path: Path) -> None:
    result = subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    archive = next(tmp_path.glob("soufflerie-*.tar.gz"))
    with tarfile.open(archive, "r:gz") as source:
        names = {Path(name) for name in source.getnames()}
    prefix = Path("soufflerie-0.1.0")
    assert {prefix / name for name in GOVERNANCE_FILES} <= names
    assert prefix / ".github/CODEOWNERS" in names
    assert prefix / ".github/pull_request_template.md" in names
    assert prefix / ".github/ISSUE_TEMPLATE/bug.yml" in names
    assert prefix / ".github/ISSUE_TEMPLATE/config.yml" in names
    assert prefix / ".github/ISSUE_TEMPLATE/feature.yml" in names
    assert prefix / ".github/ISSUE_TEMPLATE/configuration.yml" in names
