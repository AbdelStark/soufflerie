from __future__ import annotations

import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.check_distribution import check_distributions

PROJECT_ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class BuiltDistributions:
    wheel: Path
    sdist: Path


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions:
    output = tmp_path_factory.mktemp("checked-distributions")
    result = subprocess.run(
        ["uv", "build", "--out-dir", str(output)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return BuiltDistributions(
        wheel=next(output.glob("soufflerie-*.whl")),
        sdist=next(output.glob("soufflerie-*.tar.gz")),
    )


def test_built_wheel_and_sdist_match_distribution_policy(
    built_distributions: BuiltDistributions,
) -> None:
    assert (
        check_distributions(
            (built_distributions.wheel, built_distributions.sdist), root=PROJECT_ROOT
        )
        == ()
    )


def test_distribution_policy_rejects_an_executable_payload_format(
    tmp_path: Path,
    built_distributions: BuiltDistributions,
) -> None:
    wheel = tmp_path / built_distributions.wheel.name
    shutil.copyfile(built_distributions.wheel, wheel)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("soufflerie/payload.pkl", b"not a safe package member")
    errors = check_distributions((wheel, built_distributions.sdist), root=PROJECT_ROOT)
    assert any("forbidden suffix '.pkl'" in error for error in errors)
