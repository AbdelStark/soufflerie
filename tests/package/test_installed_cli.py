from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
OPTIONAL_RUNTIME_ROOTS = {
    "fastapi",
    "gradio",
    "imageio",
    "matplotlib",
    "modal",
    "physicsnemo",
    "torch",
    "warp",
}


@dataclass(frozen=True, slots=True)
class InstalledCli:
    executable: Path
    python: Path
    cwd: Path

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.executable), *arguments],
            cwd=self.cwd,
            capture_output=True,
            check=False,
            text=True,
        )


@pytest.fixture(scope="module")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> InstalledCli:
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is required for the installed-wheel CLI contract")
    root = tmp_path_factory.mktemp("installed-cli")
    distributions = root / "dist"
    environment = root / "venv"
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(distributions)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    subprocess.run(
        [uv, "venv", str(environment)],
        capture_output=True,
        check=True,
        text=True,
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheel = next(distributions.glob("soufflerie-*.whl"))
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        capture_output=True,
        check=True,
        text=True,
    )
    executable = environment / ("Scripts/soufflerie.exe" if os.name == "nt" else "bin/soufflerie")
    assert executable.is_file()
    return InstalledCli(executable=executable, python=python, cwd=root)


def test_clean_installed_cli_version_and_help(installed_cli: InstalledCli) -> None:
    version = installed_cli.run("version", "--json")
    assert version.returncode == 0
    assert version.stderr == ""
    payload = json.loads(version.stdout)
    assert payload == {
        "package": "soufflerie",
        "python": payload["python"],
        "schema_version": 1,
        "version": "0.1.0",
    }
    assert payload["python"].startswith("3.11.")

    for arguments in (
        ("--help",),
        ("solve", "--help"),
        ("dataset", "validate", "--help"),
        ("model", "inspect", "--help"),
        ("validate", "--help"),
        ("demo", "--help"),
        ("version", "--help"),
    ):
        result = installed_cli.run(*arguments)
        assert result.returncode == 0
        assert result.stdout.startswith("Usage: soufflerie")
        assert result.stderr == ""


def test_clean_base_install_reports_missing_optional_extra(installed_cli: InstalledCli) -> None:
    result = installed_cli.run(
        "solve", "--config", "case.yaml", "--output", "result", "--device", "cpu"
    )
    assert result.returncode == 5
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    error = json.loads(result.stderr)
    assert error["schema_version"] == 1
    assert error["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "soufflerie[solver]" in error["error"]["message"]


def test_clean_cli_import_does_not_load_optional_frameworks(installed_cli: InstalledCli) -> None:
    code = """
import json
import sys
before = set(sys.modules)
import soufflerie.cli
loaded = sorted({name.split('.')[0] for name in set(sys.modules) - before})
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [str(installed_cli.python), "-I", "-c", code],
        cwd=installed_cli.cwd,
        capture_output=True,
        check=True,
        text=True,
    )
    loaded = set(json.loads(result.stdout))
    assert loaded.isdisjoint(OPTIONAL_RUNTIME_ROOTS)
