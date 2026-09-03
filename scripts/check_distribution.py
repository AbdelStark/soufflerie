"""Validate Soufflerie wheel/sdist contents without extracting untrusted archives."""

from __future__ import annotations

import argparse
import email.policy
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Iterable, Mapping
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).parents[1]
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".ckpt", ".env", ".pickle", ".pkl", ".pt", ".pth", ".pyc"}
FORBIDDEN_PARTS = {
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
REQUIRED_WHEEL_MEMBERS = {
    "soufflerie/__init__.py",
    "soufflerie/py.typed",
    "soufflerie/configs/cases/cylinder-re100.yaml",
    "soufflerie/configs/service/demo-v1.yaml",
    "soufflerie/resources/model/bundle.json",
    "soufflerie/resources/model/model.safetensors.gz",
    "soufflerie/resources/model/resource.json",
    "soufflerie/schemas/v1/baseline-metadata.json",
    "soufflerie/schemas/v1/bundled-model-resource.json",
    "soufflerie/schemas/v1/bundled-model-smoke-result.json",
    "soufflerie/schemas/v1/provenance.json",
    "soufflerie/schemas/v1/training-epoch.json",
    "soufflerie/schemas/v1/training-checkpoint.json",
    "soufflerie/schemas/v1/training-selection.json",
    "soufflerie/schemas/v1/case-metrics.json",
    "soufflerie/schemas/v1/gate-result.json",
    "soufflerie/schemas/v1/metric-summary.json",
    "soufflerie/schemas/v1/ood-evaluation.json",
    "soufflerie/schemas/v1/openapi.json",
    "soufflerie/schemas/v1/plot-manifest.json",
    "soufflerie/schemas/v1/sensitivity-evaluation.json",
    "soufflerie/schemas/v1/validation-plot-data.json",
    "soufflerie/schemas/v1/validation-report.json",
    "soufflerie/service/__init__.py",
    "soufflerie/service/app.py",
    "soufflerie/service/contracts.py",
    "soufflerie/service/schema_registry.py",
    "soufflerie/training/__init__.py",
    "soufflerie/training/checkpoint.py",
    "soufflerie/training/loss.py",
    "soufflerie/training/loop.py",
    "soufflerie/training/metrics.py",
    "soufflerie/validation/gates.py",
    "soufflerie/validation/metrics.py",
    "soufflerie/validation/ood.py",
    "soufflerie/validation/plot_data.py",
    "soufflerie/validation/reporting.py",
    "soufflerie/validation/schema_registry.py",
    "soufflerie/validation/sensitivity.py",
    "soufflerie/training/runtime.py",
}
REQUIRED_SDIST_MEMBERS = {
    ".env.example",
    ".github/CODEOWNERS",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/configuration.yml",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    ".pre-commit-config.yaml",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "docs/api.md",
    "docs/validation/reporting.md",
    "LICENSE",
    "MAINTAINERS.md",
    "NOTICE",
    "README.md",
    "reports/validation-receipt.json",
    "reports/validation.json",
    "reports/validation.md",
    "reports/validation.plots.json",
    "reports/validation.plots/baseline-comparison.svg",
    "reports/validation.plots/divergence-compliance.svg",
    "reports/validation.plots/error-by-design.svg",
    "reports/validation.plots/head-vs-field.svg",
    "reports/validation.plots/ood-variance.svg",
    "reports/validation.plots/representative-fields.svg",
    "reports/validation.plots/sensitivity.svg",
    "reports/validation.plots/worst-fields.svg",
    "SECURITY.md",
    "SPEC.md",
    "pyproject.toml",
    "scripts/check.sh",
    "scripts/check_ci_policy.py",
    "scripts/check_distribution.py",
    "scripts/check_docs.py",
    "scripts/check_governance.py",
    "scripts/render_validation.py",
    "schemas/v1/provenance.json",
    "schemas/v1/openapi.json",
    "src/soufflerie/__init__.py",
    "src/soufflerie/py.typed",
    "tests/package/test_ci_policy.py",
    "tests/fixtures/report.json",
    "tests/fixtures/report.md",
    "tests/fixtures/report.plots.json",
    "tests/fixtures/report.plots/baseline-comparison.svg",
    "tests/fixtures/report.plots/divergence-compliance.svg",
    "tests/fixtures/report.plots/error-by-design.svg",
    "tests/fixtures/report.plots/head-vs-field.svg",
    "tests/fixtures/report.plots/ood-variance.svg",
    "tests/fixtures/report.plots/representative-fields.svg",
    "tests/fixtures/report.plots/sensitivity.svg",
    "tests/fixtures/report.plots/worst-fields.svg",
    "tests/service/test_health.py",
    "tests/service/test_openapi.py",
    "tests/service/test_schemas.py",
    "tests/validation/test_report.py",
    "uv.lock",
}


def _project_metadata(root: Path) -> tuple[str, str]:
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = document["project"]
    if not isinstance(project, Mapping):
        raise ValueError("pyproject.toml project metadata is not a table")
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError("pyproject.toml requires static project name and version")
    return name, version


def _safe_member_name(name: str) -> str | None:
    if "\x00" in name or "\\" in name:
        return "contains a NUL or alternate path separator"
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or str(path) != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return "is not a normalized relative POSIX path"
    if FORBIDDEN_PARTS.intersection(path.parts):
        return "contains a cache or repository-internal directory"
    if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        return f"uses forbidden suffix {path.suffix!r}"
    return None


def _check_archive_path(path: Path, errors: list[str]) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        errors.append(f"{path}: unable to stat distribution: {error}")
        return
    if size < 1 or size > MAX_ARCHIVE_BYTES:
        errors.append(f"{path}: distribution size {size} exceeds the bounded package policy")


def _check_wheel(path: Path, *, root: Path, name: str, version: str) -> tuple[str, ...]:
    errors: list[str] = []
    _check_archive_path(path, errors)
    expected_filename = f"{name}-{version}-py3-none-any.whl"
    if path.name != expected_filename:
        errors.append(f"{path}: expected pure-Python wheel filename {expected_filename!r}")
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as error:
        return (*errors, f"{path}: invalid wheel ZIP: {error}")
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            errors.append(f"{path}: wheel contains duplicate member names")
        for info in infos:
            reason = _safe_member_name(info.filename)
            if reason:
                errors.append(f"{path}: member {info.filename!r} {reason}")
            if info.file_size > MAX_MEMBER_BYTES:
                errors.append(f"{path}: member {info.filename!r} exceeds the byte cap")
        missing = sorted(REQUIRED_WHEEL_MEMBERS - set(names))
        if missing:
            errors.append(f"{path}: wheel omits required members: {', '.join(missing)}")
        forbidden_roots = ("tests/", "docs/", "infra/", "reports/")
        leaked = sorted(member for member in names if member.startswith(forbidden_roots))
        if leaked:
            errors.append(f"{path}: wheel leaks source-only members: {', '.join(leaked)}")

        dist_info = f"{name}-{version}.dist-info"
        metadata_name = f"{dist_info}/METADATA"
        wheel_name = f"{dist_info}/WHEEL"
        license_members = {
            f"{dist_info}/licenses/LICENSE": root.joinpath("LICENSE").read_bytes(),
            f"{dist_info}/licenses/NOTICE": root.joinpath("NOTICE").read_bytes(),
        }
        for member, expected_content in license_members.items():
            if member not in names:
                errors.append(f"{path}: wheel omits {member}")
            elif archive.read(member) != expected_content:
                errors.append(f"{path}: packaged {member} differs from reviewed source")
        if metadata_name not in names:
            errors.append(f"{path}: wheel omits core metadata")
        else:
            metadata = BytesParser(policy=email.policy.default).parsebytes(
                archive.read(metadata_name)
            )
            expected_headers = {
                "Name": name,
                "Version": version,
                "License-Expression": "Apache-2.0",
                "Author": "AbdelStark",
            }
            for header, expected_value in expected_headers.items():
                observed = metadata.get(header)
                if observed != expected_value:
                    errors.append(
                        f"{path}: METADATA {header} is {observed!r}, expected {expected_value!r}"
                    )
            if metadata.get("Requires-Python") not in {">=3.11,<3.12", "<3.12,>=3.11"}:
                errors.append(
                    f"{path}: METADATA Requires-Python does not match the Python 3.11 contract"
                )
            if sorted(metadata.get_all("License-File", ())) != ["LICENSE", "NOTICE"]:
                errors.append(f"{path}: METADATA License-File declarations are incomplete")
        if wheel_name not in names:
            errors.append(f"{path}: wheel omits WHEEL metadata")
        elif b"Root-Is-Purelib: true" not in archive.read(wheel_name):
            errors.append(f"{path}: wheel must declare a pure-Python root")
    return tuple(errors)


def _stripped_sdist_names(names: Iterable[str], *, prefix: str) -> set[str]:
    stripped: set[str] = set()
    for name in names:
        if name == prefix.rstrip("/"):
            continue
        if name.startswith(prefix):
            stripped.add(name.removeprefix(prefix))
    return stripped


def _inspect_sdist_archive(
    archive: tarfile.TarFile,
    *,
    path: Path,
    root: Path,
    name: str,
    version: str,
    errors: list[str],
) -> None:
    prefix = f"{name}-{version}/"
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        errors.append(f"{path}: source archive contains duplicate member names")
    for member in members:
        reason = _safe_member_name(member.name)
        if reason:
            errors.append(f"{path}: member {member.name!r} {reason}")
        if not (member.name == prefix.rstrip("/") or member.name.startswith(prefix)):
            errors.append(f"{path}: member {member.name!r} escapes the versioned root")
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            errors.append(f"{path}: member {member.name!r} has a forbidden archive type")
        if member.size > MAX_MEMBER_BYTES:
            errors.append(f"{path}: member {member.name!r} exceeds the byte cap")
    stripped = _stripped_sdist_names(names, prefix=prefix)
    missing = sorted(REQUIRED_SDIST_MEMBERS - stripped)
    if missing:
        errors.append(f"{path}: source archive omits required members: {', '.join(missing)}")
    if ".env" in stripped:
        errors.append(f"{path}: source archive contains a local .env file")
    for governed in ("LICENSE", "NOTICE", "CITATION.cff"):
        member_name = f"{prefix}{governed}"
        try:
            extracted = archive.extractfile(member_name)
            content = extracted.read() if extracted is not None else b""
        except (KeyError, OSError, tarfile.TarError) as error:
            errors.append(f"{path}: unable to read {governed}: {error}")
            continue
        if content != root.joinpath(governed).read_bytes():
            errors.append(f"{path}: packaged {governed} differs from reviewed source")


def _check_sdist(path: Path, *, root: Path, name: str, version: str) -> tuple[str, ...]:
    errors: list[str] = []
    _check_archive_path(path, errors)
    expected_filename = f"{name}-{version}.tar.gz"
    if path.name != expected_filename:
        errors.append(f"{path}: expected source archive filename {expected_filename!r}")
    try:
        with tarfile.open(path, "r:gz") as archive:
            _inspect_sdist_archive(
                archive,
                path=path,
                root=root,
                name=name,
                version=version,
                errors=errors,
            )
    except (OSError, tarfile.TarError) as error:
        return (*errors, f"{path}: invalid source archive: {error}")
    return tuple(errors)


def check_distributions(paths: Iterable[Path], *, root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Validate exactly one wheel and one sdist against package/repository policy."""

    root = root.resolve()
    materialized = tuple(path.resolve() for path in paths)
    wheels = tuple(path for path in materialized if path.suffix == ".whl")
    sdists = tuple(path for path in materialized if path.name.endswith(".tar.gz"))
    unknown = tuple(path for path in materialized if path not in {*wheels, *sdists})
    errors: list[str] = []
    if len(wheels) != 1 or len(sdists) != 1 or unknown:
        return (
            "distribution validation requires exactly one .whl and one .tar.gz "
            f"(wheels={len(wheels)}, sdists={len(sdists)}, unknown={len(unknown)})",
        )
    try:
        name, version = _project_metadata(root)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError, KeyError) as error:
        return (f"unable to load reviewed project metadata: {error}",)
    errors.extend(_check_wheel(wheels[0], root=root, name=name, version=version))
    errors.extend(_check_sdist(sdists[0], root=root, name=name, version=version))
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    errors = check_distributions(args.paths, root=args.root)
    if errors:
        for error in errors:
            print(f"distribution error: {error}", file=sys.stderr)
        return 1
    print("distribution contracts: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
