from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parents[2]


def _project() -> dict[str, Any]:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_python_and_build_contract() -> None:
    document = _project()
    project = document["project"]
    assert isinstance(project, dict)
    assert project["name"] == "soufflerie"
    assert project["requires-python"] == ">=3.11,<3.12"
    assert document["build-system"]["build-backend"] == "hatchling.build"


def test_runtime_profiles_match_rfc_0001() -> None:
    project = _project()["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert extras == {
        "solver": ["warp-lang==1.17.0"],
        "ml": [
            "nvidia-physicsnemo[cu12]==2.2.1",
            "tensorboard==2.21.0",
            "torch==2.10.0",
        ],
        "remote": ["modal==1.5.5"],
        "serve": ["fastapi==0.141.1", "gradio==6.26.0", "httpx==0.28.1"],
        "viz": ["imageio==2.37.4", "matplotlib==3.11.1", "pillow==12.3.0"],
    }


def test_dev_profile_is_exactly_pinned() -> None:
    groups = _project()["dependency-groups"]
    assert groups["dev"] == [
        "hypothesis==6.167.1",
        "mypy==2.3.1",
        "pre-commit==4.6.2",
        "pytest==9.1.1",
        "pytest-cov==7.1.0",
        "ruff==0.16.5",
    ]


def test_lock_contains_every_exact_direct_pin() -> None:
    document = _project()
    project = document["project"]
    groups = document["dependency-groups"]
    direct_requirements = [
        *project["dependencies"],
        *(
            requirement
            for extra in project["optional-dependencies"].values()
            for requirement in extra
        ),
        *(requirement for group in groups.values() for requirement in group),
    ]

    pin_pattern = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)")

    def canonical_name(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    direct_pins = {
        (canonical_name(match.group(1)), match.group(2))
        for requirement in direct_requirements
        if (match := pin_pattern.match(requirement))
    }
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_versions = {
        (canonical_name(package["name"]), package["version"]) for package in lock["package"]
    }

    assert direct_pins <= locked_versions
