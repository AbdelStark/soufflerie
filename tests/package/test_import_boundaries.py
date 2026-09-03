from __future__ import annotations

import ast
import importlib.resources
import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import soufflerie

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "soufflerie"
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
FOUNDATION_MODULES = {"config", "geometry", "observability", "schemas"}
DOMAIN_MODULES = {"datagen", "solver", "surrogate"}
TRAINING_MODULES = {"training"}
VALIDATION_MODULES = {"validation"}
APPLICATION_MODULES = {"cli", "demo", "service"}
LAYERS = {
    **dict.fromkeys(FOUNDATION_MODULES, 0),
    **dict.fromkeys(DOMAIN_MODULES, 1),
    **dict.fromkeys(TRAINING_MODULES, 2),
    **dict.fromkeys(VALIDATION_MODULES, 3),
    **dict.fromkeys(APPLICATION_MODULES, 4),
}


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(PACKAGE_ROOT)
    is_package = relative.name == "__init__.py"
    parts = relative.parent.parts if is_package else relative.with_suffix("").parts
    suffix = ".".join(parts)
    return (f"soufflerie.{suffix}" if suffix else "soufflerie", is_package)


def _internal_imports(path: Path) -> Iterator[str]:
    module_name, is_package = _module_name(path)
    package = module_name if is_package else module_name.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (name.name for name in node.names if name.name.startswith("soufflerie"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                target = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                if node.module is None:
                    yield from (f"{target}.{name.name}" for name in node.names)
                else:
                    yield target
            elif node.module == "soufflerie":
                yield from (f"soufflerie.{name.name}" for name in node.names)
            elif node.module and node.module.startswith("soufflerie."):
                yield node.module


def _top_level_module(module_name: str) -> str | None:
    parts = module_name.split(".")
    return parts[1] if len(parts) > 1 else None


def test_public_package_is_typed_and_versioned() -> None:
    assert "__version__" in soufflerie.__all__
    assert "CaseConfig" in soufflerie.__all__
    assert soufflerie.__version__ == "0.1.0"
    assert importlib.resources.files("soufflerie").joinpath("py.typed").is_file()


def test_base_import_does_not_load_optional_frameworks() -> None:
    code = """
import json
import sys

before = set(sys.modules)
import soufflerie
loaded = sorted({name.split('.')[0] for name in set(sys.modules) - before})
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_roots = set(json.loads(completed.stdout))
    assert loaded_roots.isdisjoint(OPTIONAL_RUNTIME_ROOTS)


def test_training_contract_import_does_not_load_optional_frameworks() -> None:
    code = """
import json
import sys

before = set(sys.modules)
import soufflerie.training
loaded = sorted({name.split('.')[0] for name in set(sys.modules) - before})
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_roots = set(json.loads(completed.stdout))
    assert loaded_roots.isdisjoint(OPTIONAL_RUNTIME_ROOTS)


def test_service_contract_import_does_not_load_optional_frameworks() -> None:
    code = """
import json
import sys

before = set(sys.modules)
import soufflerie.service
loaded = sorted({name.split('.')[0] for name in set(sys.modules) - before})
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_roots = set(json.loads(completed.stdout))
    assert loaded_roots.isdisjoint(OPTIONAL_RUNTIME_ROOTS)


def test_module_tree_matches_architecture() -> None:
    expected_modules = {
        "cli.py",
        "config.py",
        "geometry.py",
        "observability.py",
        "schemas.py",
    }
    expected_packages = {
        "datagen",
        "demo",
        "service",
        "solver",
        "surrogate",
        "training",
        "validation",
    }
    assert {path.name for path in PACKAGE_ROOT.glob("*.py")} >= expected_modules
    assert {path.parent.name for path in PACKAGE_ROOT.glob("*/__init__.py")} == expected_packages


def test_package_imports_follow_dependency_direction() -> None:
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        source_module, _ = _module_name(path)
        source_top = _top_level_module(source_module)
        # The package root is a public facade and may lazily reach inward.
        source_layer = LAYERS.get(source_top, 0) if source_top is not None else 3
        for target in _internal_imports(path):
            target_top = _top_level_module(target)
            target_layer = LAYERS.get(target_top, 0) if target_top is not None else 0
            if target_layer > source_layer:
                violations.append(f"{source_module} imports higher layer {target}")
            if (
                source_top in DOMAIN_MODULES
                and target_top in DOMAIN_MODULES
                and source_top != target_top
            ):
                violations.append(f"{source_module} imports peer domain {target}")

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                name.name == "infra" or name.name.startswith("infra.") for name in node.names
            ):
                violations.append(f"{source_module} imports infra")
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "infra" or node.module.startswith("infra."))
            ):
                violations.append(f"{source_module} imports infra")

    assert violations == []
