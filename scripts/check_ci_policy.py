"""Fail closed on unsafe, floating, remote, or incomplete pull-request CI policy."""

from __future__ import annotations

import argparse
import ast
import re
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).parents[1]
WORKFLOW_PATH = Path(".github/workflows/ci.yml")
PRE_COMMIT_PATH = Path(".pre-commit-config.yaml")
PINNED_ACTION = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")
REMOTE_BINARIES = {"curl", "gh", "modal", "rsync", "scp", "ssh", "wget"}
REMOTE_IMPORT_ROOTS = {"boto3", "google.cloud", "modal", "requests", "urllib.request"}
NETWORK_CALLS = {
    "socket.create_connection",
    "socket.socket.connect",
    "urllib.request.urlopen",
}
SUBPROCESS_CALLS = {
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
}
REQUIRED_JOBS = {"lint-type", "unit-contract", "integration"}
REQUIRED_COMMANDS = {
    "uv lock --check",
    "uv run ruff check .",
    "uv run ruff format --check .",
    "uv run mypy src tests",
    'uv run pytest -m "not remote" --cov=src/soufflerie --cov-report=term-missing '
    "--cov-fail-under=90",
    "uv run python scripts/validate_schemas.py",
    "uv run python scripts/render_validation.py --check tests/fixtures/report.json",
    "uv run python scripts/render_validation.py --check reports/validation.json",
    "uv run python scripts/check_docs.py",
    "uv build",
    "uv run python scripts/check_distribution.py dist/*",
}
REQUIRED_HOOKS = {
    "ci-policy",
    "docs-contracts",
    "governance-contracts",
    "lock-check",
    "ruff-check",
    "ruff-format-check",
    "schema-contracts",
    "strict-mypy",
    "validation-report",
}


def _mapping(value: object, *, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be a mapping")
        return None
    return value


def _sequence(value: object, *, label: str, errors: list[str]) -> Sequence[Any] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.append(f"{label} must be a sequence")
        return None
    return value


def _load_workflow(path: Path, errors: list[str]) -> Mapping[str, Any] | None:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        errors.append(f"{path}: invalid workflow YAML: {error}")
        return None
    return _mapping(loaded, label=str(path), errors=errors)


def _workflow_steps(
    jobs: Mapping[str, Any], errors: list[str]
) -> Iterable[tuple[str, int, Mapping[str, Any]]]:
    for job_name, raw_job in jobs.items():
        job = _mapping(raw_job, label=f"job {job_name!r}", errors=errors)
        if job is None:
            continue
        steps = _sequence(job.get("steps"), label=f"job {job_name!r} steps", errors=errors)
        if steps is None:
            continue
        for index, raw_step in enumerate(steps):
            step = _mapping(
                raw_step,
                label=f"job {job_name!r} step {index}",
                errors=errors,
            )
            if step is not None:
                yield str(job_name), index, step


def _normalized_commands(steps: Iterable[tuple[str, int, Mapping[str, Any]]]) -> set[str]:
    commands: set[str] = set()
    for _, _, step in steps:
        run = step.get("run")
        if not isinstance(run, str):
            continue
        commands.add(" ".join(run.split()))
        commands.update(line.strip() for line in run.splitlines() if line.strip())
    return commands


def check_workflow_policy(path: Path) -> tuple[str, ...]:
    """Validate one pull-request workflow without executing it."""

    errors: list[str] = []
    workflow = _load_workflow(path, errors)
    if workflow is None:
        return tuple(errors)
    raw = path.read_text(encoding="utf-8")
    normalized_raw = raw.casefold()
    if "pull_request_target" in normalized_raw:
        errors.append("CI workflow must not use pull_request_target")
    if "${{ secrets." in normalized_raw:
        errors.append("pull-request CI must not reference repository or environment secrets")

    triggers = _mapping(workflow.get("on"), label="workflow triggers", errors=errors)
    if triggers is not None:
        if "pull_request" not in triggers:
            errors.append("CI workflow must run for pull_request")
        unexpected = sorted(set(triggers) - {"pull_request", "push"})
        if unexpected:
            errors.append(f"CI workflow has unexpected triggers: {', '.join(unexpected)}")

    permissions = _mapping(workflow.get("permissions"), label="workflow permissions", errors=errors)
    if permissions is not None and permissions != {"contents": "read"}:
        errors.append("CI workflow permissions must be exactly contents: read")

    concurrency = _mapping(workflow.get("concurrency"), label="workflow concurrency", errors=errors)
    if concurrency is not None:
        group = concurrency.get("group")
        if not isinstance(group, str) or "github.event.pull_request.number" not in group:
            errors.append("CI concurrency must isolate pull-request branches")
        if concurrency.get("cancel-in-progress") != "true":
            errors.append("CI concurrency must cancel superseded branch runs")

    jobs = _mapping(workflow.get("jobs"), label="workflow jobs", errors=errors)
    if jobs is None:
        return tuple(errors)
    if set(jobs) != REQUIRED_JOBS:
        errors.append(
            "CI jobs must be exactly distinct lint-type, unit-contract, and integration jobs"
        )

    materialized_steps = tuple(_workflow_steps(jobs, errors))
    commands = _normalized_commands(materialized_steps)
    for required in sorted(REQUIRED_COMMANDS):
        if required not in commands:
            errors.append(f"CI workflow omits exact specification command: {required}")

    job_evidence: dict[str, set[str]] = {name: set() for name in jobs}
    for job_name, index, step in materialized_steps:
        uses = step.get("uses")
        if isinstance(uses, str):
            if uses.startswith("./"):
                pass
            elif not PINNED_ACTION.fullmatch(uses):
                errors.append(
                    f"job {job_name!r} step {index} uses an unpinned action reference: {uses}"
                )
            action = uses.split("@", maxsplit=1)[0]
            inputs = step.get("with")
            with_values = inputs if isinstance(inputs, Mapping) else {}
            if action == "actions/checkout":
                job_evidence[job_name].add("checkout")
                if with_values.get("persist-credentials") != "false":
                    errors.append(f"job {job_name!r} checkout must disable persisted credentials")
            elif action == "actions/setup-python":
                job_evidence[job_name].add("python")
                if with_values.get("python-version") != "3.11":
                    errors.append(f"job {job_name!r} must select Python 3.11 exactly")
            elif action == "astral-sh/setup-uv":
                job_evidence[job_name].add("uv")
                expected_uv = {
                    "version": "0.12.8",
                    "enable-cache": "true",
                    "cache-dependency-glob": "uv.lock",
                }
                for key, expected in expected_uv.items():
                    if with_values.get(key) != expected:
                        errors.append(f"job {job_name!r} setup-uv {key} must equal {expected!r}")

        run = step.get("run")
        if isinstance(run, str):
            normalized_run = " ".join(run.split())
            if normalized_run in {"uv sync --frozen", "uv sync --frozen --extra solver"}:
                job_evidence[job_name].add("sync")
            if normalized_run == "uv sync --frozen --extra solver":
                job_evidence[job_name].add("solver-extra")
            if "pytest" in run:
                environment = step.get("env")
                env_values = environment if isinstance(environment, Mapping) else {}
                if env_values.get("UV_OFFLINE") != "1":
                    errors.append(f"job {job_name!r} pytest step must set UV_OFFLINE=1")
                if "not remote" not in run:
                    errors.append(f"job {job_name!r} pytest step must exclude remote tests")
            for line in run.splitlines():
                try:
                    words = shlex.split(line)
                except ValueError:
                    continue
                if words and words[0] in REMOTE_BINARIES:
                    errors.append(
                        f"job {job_name!r} step {index} invokes remote command {words[0]!r}"
                    )

    for job_name, evidence in job_evidence.items():
        missing = sorted({"checkout", "python", "sync", "uv"} - evidence)
        if missing:
            errors.append(f"job {job_name!r} lacks setup evidence: {', '.join(missing)}")
        if job_name in {"integration", "unit-contract"} and "solver-extra" not in evidence:
            errors.append(f"job {job_name!r} must sync the locked solver extra")
        raw_job = jobs[job_name]
        if isinstance(raw_job, Mapping):
            if raw_job.get("runs-on") != "ubuntu-latest":
                errors.append(f"job {job_name!r} must use the reviewed CPU runner")
            if "environment" in raw_job or "secrets" in raw_job:
                errors.append(f"job {job_name!r} must not bind protected environments or secrets")
    return tuple(errors)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return None


def _contains_remote_marker(node: ast.AST) -> bool:
    return any(_dotted_name(child) == "pytest.mark.remote" for child in ast.walk(node))


def _literal_command(node: ast.AST) -> str | None:
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        first = node.elts[0]
        return (
            first.value
            if isinstance(first, ast.Constant) and isinstance(first.value, str)
            else None
        )
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            words = shlex.split(node.value)
        except ValueError:
            return None
        return words[0] if words else None
    return None


def _module_remote_evidence(tree: ast.Module) -> tuple[bool, set[str], set[str]]:
    module_marked = False
    remote_modules: set[str] = set()
    remote_callables: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            module_marked = _contains_remote_marker(node.value)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == root or alias.name.startswith(f"{root}.")
                    for root in REMOTE_IMPORT_ROOTS
                ):
                    remote_modules.add(alias.asname or alias.name.split(".", maxsplit=1)[0])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and any(
                node.module == root or node.module.startswith(f"{root}.")
                for root in REMOTE_IMPORT_ROOTS
            )
        ):
            for alias in node.names:
                remote_callables.add(alias.asname or alias.name)
    return module_marked, remote_modules, remote_callables


def _function_remote_reason(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    remote_modules: set[str],
    remote_callables: set[str],
) -> str | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func) or ""
        root = name.split(".", maxsplit=1)[0]
        if root in remote_modules or name in remote_callables or name in NETWORK_CALLS:
            return f"network/remote call {name!r}"
        if name in SUBPROCESS_CALLS and node.args:
            command = _literal_command(node.args[0])
            if command is not None and Path(command).name in REMOTE_BINARIES:
                return f"remote subprocess command {Path(command).name!r}"
    return None


def find_unmarked_remote_tests(tests_root: Path) -> tuple[str, ...]:
    """Return remote-capable tests that lack an explicit pytest remote marker."""

    errors: list[str] = []
    for path in sorted(tests_root.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            errors.append(f"{path}: unable to inspect test markers: {error}")
            continue
        module_marked, remote_modules, remote_callables = _module_remote_evidence(tree)
        path_is_remote = "remote" in path.relative_to(tests_root).parts
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_test = node.name.startswith("test_")
            marked = module_marked or (
                is_test
                and any(_contains_remote_marker(decorator) for decorator in node.decorator_list)
            )
            reason = "test is located under tests/remote"
            if not path_is_remote:
                discovered = _function_remote_reason(
                    node,
                    remote_modules=remote_modules,
                    remote_callables=remote_callables,
                )
                if discovered is None:
                    continue
                reason = discovered
            if not marked:
                relative = path.relative_to(tests_root.parent)
                subject = node.name if is_test else f"helper or fixture {node.name}"
                errors.append(
                    f"{relative}:{node.lineno}: {subject} lacks remote module marking ({reason})"
                )
    return tuple(errors)


def check_pre_commit_policy(path: Path) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        return (f"{path}: invalid pre-commit YAML: {error}",)
    config = _mapping(loaded, label=str(path), errors=errors)
    if config is None:
        return tuple(errors)
    if config.get("minimum_pre_commit_version") != "4.6.2":
        errors.append("pre-commit must require the locked 4.6.2 runner")
    repositories = _sequence(config.get("repos"), label="pre-commit repos", errors=errors)
    observed_hooks: set[str] = set()
    if repositories is not None:
        for index, raw_repository in enumerate(repositories):
            repository = _mapping(
                raw_repository,
                label=f"pre-commit repo {index}",
                errors=errors,
            )
            if repository is None:
                continue
            if repository.get("repo") not in {"local", "meta"}:
                errors.append(
                    "pre-commit hooks must use the locked environment or built-in meta hooks"
                )
            hooks = _sequence(
                repository.get("hooks"),
                label=f"pre-commit repo {index} hooks",
                errors=errors,
            )
            if hooks is None:
                continue
            for raw_hook in hooks:
                if isinstance(raw_hook, Mapping) and isinstance(raw_hook.get("id"), str):
                    observed_hooks.add(raw_hook["id"])
                    entry = raw_hook.get("entry")
                    if isinstance(entry, str) and "--fix" in entry:
                        errors.append(f"pre-commit hook {raw_hook['id']!r} must not mutate files")
    missing = sorted(REQUIRED_HOOKS - observed_hooks)
    if missing:
        errors.append(f"pre-commit omits required hooks: {', '.join(missing)}")
    return tuple(errors)


def check_ci_policy(root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Validate the complete local and GitHub pull-request trust boundary."""

    root = root.resolve()
    errors = [
        *check_workflow_policy(root / WORKFLOW_PATH),
        *check_pre_commit_policy(root / PRE_COMMIT_PATH),
        *find_unmarked_remote_tests(root / "tests"),
    ]
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    errors = check_ci_policy(args.root)
    if errors:
        for error in errors:
            print(f"CI policy error: {error}", file=sys.stderr)
        return 1
    print("CI policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
