"""Validate repository governance, ownership, templates, and license metadata."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).parents[1]
OWNER = "AbdelStark"
PRIVATE_REPORT_URL = "https://github.com/AbdelStark/soufflerie/security/advisories/new"
REQUIRED_FILES = (
    "LICENSE",
    "NOTICE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CITATION.cff",
    "MAINTAINERS.md",
    ".github/CODEOWNERS",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/feature.yml",
    ".github/ISSUE_TEMPLATE/configuration.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
)
REQUIRED_SDIST_ENTRIES = {
    "/.github",
    "/CITATION.cff",
    "/CODE_OF_CONDUCT.md",
    "/CONTRIBUTING.md",
    "/LICENSE",
    "/MAINTAINERS.md",
    "/NOTICE",
    "/SECURITY.md",
}
ISSUE_FORMS = (
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/feature.yml",
    ".github/ISSUE_TEMPLATE/configuration.yml",
)
ISSUE_FORM_TYPES = {"markdown", "textarea", "input", "dropdown", "checkboxes", "upload"}
ISSUE_LABELS = {"type:bug", "type:feature"}
REQUIRED_CODEOWNER_SURFACES = (
    "/src/soufflerie/solver/",
    "/src/soufflerie/surrogate/",
    "/src/soufflerie/training/",
    "/src/soufflerie/service/",
    "/src/soufflerie/artifacts.py",
    "/infra/",
    "/docs/security/",
    "/.github/",
    "/pyproject.toml",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
FORM_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _mapping(value: object, *, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected a mapping")
        return None
    return value


def _load_yaml(path: Path, errors: list[str]) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        errors.append(f"{path}: invalid YAML: {error}")
        return None


def _check_license(root: Path, document: Mapping[str, Any], errors: list[str]) -> None:
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    required_license_text = (
        "Apache License",
        "Version 2.0, January 2004",
        "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
        "1. Definitions.",
        "9. Accepting Warranty or Additional Liability.",
        "END OF TERMS AND CONDITIONS",
        "APPENDIX: How to apply the Apache License to your work.",
    )
    for phrase in required_license_text:
        if phrase not in license_text:
            errors.append(f"LICENSE: missing canonical Apache-2.0 text {phrase!r}")

    notice = (root / "NOTICE").read_text(encoding="utf-8")
    if notice != "Soufflerie\nCopyright 2026 AbdelStark\n":
        errors.append("NOTICE: expected the factual project and maintainer attribution")

    project = _mapping(document.get("project"), path="pyproject.toml [project]", errors=errors)
    if project is None:
        return
    if project.get("license") != "Apache-2.0":
        errors.append("pyproject.toml: project.license must be the Apache-2.0 SPDX expression")
    if project.get("license-files") != ["LICENSE", "NOTICE"]:
        errors.append("pyproject.toml: project.license-files must include LICENSE and NOTICE")
    if project.get("authors") != [{"name": OWNER}]:
        errors.append("pyproject.toml: authors must retain the factual maintainer identity")

    hatch = _mapping(
        document.get("tool", {}).get("hatch", {}),
        path="pyproject.toml [tool.hatch]",
        errors=errors,
    )
    if hatch is None:
        return
    build = _mapping(hatch.get("build"), path="pyproject.toml [tool.hatch.build]", errors=errors)
    targets = _mapping(
        build.get("targets") if build else None,
        path="pyproject.toml [tool.hatch.build.targets]",
        errors=errors,
    )
    sdist = _mapping(
        targets.get("sdist") if targets else None,
        path="pyproject.toml [tool.hatch.build.targets.sdist]",
        errors=errors,
    )
    included = set(sdist.get("include", ())) if sdist else set()
    missing = sorted(REQUIRED_SDIST_ENTRIES - included)
    if missing:
        errors.append(f"pyproject.toml: sdist omits governance entries: {', '.join(missing)}")


def _check_citation(root: Path, errors: list[str]) -> None:
    citation = _mapping(
        _load_yaml(root / "CITATION.cff", errors),
        path="CITATION.cff",
        errors=errors,
    )
    if citation is None:
        return
    expected = {
        "cff-version": "1.2.0",
        "title": "Soufflerie",
        "type": "software",
        "repository-code": "https://github.com/AbdelStark/soufflerie",
        "version": "0.1.0",
        "license": "Apache-2.0",
    }
    for key, value in expected.items():
        if citation.get(key) != value:
            errors.append(f"CITATION.cff: {key} must equal {value!r}")
    if citation.get("authors") != [{"name": OWNER}]:
        errors.append("CITATION.cff: authors must use the verified maintainer identity")
    message = citation.get("message")
    if not isinstance(message, str) or "cite" not in message.casefold():
        errors.append("CITATION.cff: message must give citation instructions")


def _check_issue_form(path: Path, errors: list[str]) -> None:
    relative = str(path.relative_to(path.parents[2]))
    form = _mapping(_load_yaml(path, errors), path=relative, errors=errors)
    if form is None:
        return
    for key in ("name", "description", "body"):
        if key not in form:
            errors.append(f"{relative}: missing required top-level key {key!r}")
    if not isinstance(form.get("name"), str) or len(form["name"].strip()) <= 3:
        errors.append(f"{relative}: name must contain more than three characters")
    if not isinstance(form.get("description"), str) or not form["description"].strip():
        errors.append(f"{relative}: description must be a non-empty string")
    if form.get("assignees") != [OWNER]:
        errors.append(f"{relative}: must route to the active maintainer {OWNER}")
    labels = form.get("labels")
    if not isinstance(labels, list) or not labels or not set(labels) <= ISSUE_LABELS:
        errors.append(f"{relative}: labels must use existing repository labels")

    body = form.get("body")
    if not isinstance(body, list) or not body:
        errors.append(f"{relative}: body must be a non-empty list")
        return
    identifiers: set[str] = set()
    non_markdown = 0
    required_input = 0
    for index, raw_item in enumerate(body):
        item = _mapping(raw_item, path=f"{relative}: body[{index}]", errors=errors)
        if item is None:
            continue
        item_type = item.get("type")
        if item_type not in ISSUE_FORM_TYPES:
            errors.append(f"{relative}: body[{index}] has unsupported type {item_type!r}")
            continue
        if item_type != "markdown":
            non_markdown += 1
            identifier = item.get("id")
            if not isinstance(identifier, str) or FORM_ID.fullmatch(identifier) is None:
                errors.append(f"{relative}: body[{index}] requires a valid unique id")
            elif identifier in identifiers:
                errors.append(f"{relative}: duplicate body id {identifier!r}")
            else:
                identifiers.add(identifier)
            attributes = _mapping(
                item.get("attributes"),
                path=f"{relative}: body[{index}].attributes",
                errors=errors,
            )
            if attributes is not None and not isinstance(attributes.get("label"), str):
                errors.append(f"{relative}: body[{index}] requires a visible label")
            validations = item.get("validations")
            if isinstance(validations, Mapping) and validations.get("required") is True:
                required_input += 1
    if non_markdown == 0:
        errors.append(f"{relative}: body requires at least one non-markdown input")
    if required_input == 0:
        errors.append(f"{relative}: at least one input must be required")


def _check_templates(root: Path, errors: list[str]) -> None:
    codeowners = (root / ".github/CODEOWNERS").read_text(encoding="utf-8")
    for line in codeowners.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and f"@{OWNER}" not in stripped.split():
            errors.append(f".github/CODEOWNERS: path lacks the real owner: {stripped}")
    for surface in REQUIRED_CODEOWNER_SURFACES:
        if surface not in codeowners:
            errors.append(f".github/CODEOWNERS: missing ownership for {surface}")

    for relative in ISSUE_FORMS:
        _check_issue_form(root / relative, errors)

    chooser = _mapping(
        _load_yaml(root / ".github/ISSUE_TEMPLATE/config.yml", errors),
        path=".github/ISSUE_TEMPLATE/config.yml",
        errors=errors,
    )
    if chooser is not None:
        if chooser.get("blank_issues_enabled") is not False:
            errors.append("issue-template config must disable unstructured blank issues")
        links = chooser.get("contact_links")
        if not isinstance(links, list) or not any(
            isinstance(link, Mapping) and link.get("url") == PRIVATE_REPORT_URL for link in links
        ):
            errors.append("issue-template config must route private reports to the real repository")

    pull_request = (root / ".github/pull_request_template.md").read_text(encoding="utf-8")
    for phrase in (
        "Closes #",
        "## Problem",
        "## Solution",
        "## Validation",
        "## Caveats and evidence",
        "Signed-off-by",
    ):
        if phrase not in pull_request:
            errors.append(f"pull-request template: missing {phrase!r}")


def _check_policy_text(root: Path, errors: list[str]) -> None:
    contributing = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    normalized_contributing = " ".join(contributing.split())
    for phrase in (
        "Developer Certificate of Origin 1.1",
        "git commit -s",
        "Numerical golden files",
        'pytest -m "not remote"',
        "Every remote test must carry `remote`",
    ):
        if phrase not in normalized_contributing:
            errors.append(f"CONTRIBUTING.md: missing policy {phrase!r}")

    security = (root / "SECURITY.md").read_text(encoding="utf-8")
    normalized_security = " ".join(security.split())
    for phrase in (PRIVATE_REPORT_URL, "best-effort targets", "not a service-level agreement"):
        if phrase not in normalized_security:
            errors.append(f"SECURITY.md: missing accurate disclosure policy {phrase!r}")

    conduct = (root / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    normalized_conduct = " ".join(conduct.split())
    for phrase in (PRIVATE_REPORT_URL, "Contributor Covenant", "version 2.1"):
        if phrase not in normalized_conduct:
            errors.append(f"CODE_OF_CONDUCT.md: missing enforcement policy {phrase!r}")

    maintainers = (root / "MAINTAINERS.md").read_text(encoding="utf-8")
    if f"[@{OWNER}](https://github.com/{OWNER})" not in maintainers:
        errors.append("MAINTAINERS.md: active maintainer must use the verified GitHub identity")
    if "single-maintainer project" not in maintainers:
        errors.append("MAINTAINERS.md: staffing limitations must remain explicit")

    governed_text = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "SECURITY.md",
            "MAINTAINERS.md",
            ".github/pull_request_template.md",
        )
    ).casefold()
    for token in ("[insert", "todo", "tbd", "example.com", "<owner>", "@owner"):
        if token in governed_text:
            errors.append(f"governance text contains unresolved placeholder {token!r}")


def _check_markdown_links(root: Path, errors: list[str]) -> None:
    for relative in (
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "MAINTAINERS.md",
        ".github/pull_request_template.md",
    ):
        path = root / relative
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            destination = target.split(maxsplit=1)[0].strip("<>")
            if destination.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = destination.split("#", maxsplit=1)[0]
            if local and not (path.parent / local).resolve().is_file():
                errors.append(f"{relative}: broken relative link {target!r}")


def _check_pytest_markers(document: Mapping[str, Any], errors: list[str]) -> None:
    tool = document.get("tool")
    pytest_options: object = None
    if isinstance(tool, Mapping):
        pytest = tool.get("pytest")
        if isinstance(pytest, Mapping):
            pytest_options = pytest.get("ini_options")
    options = _mapping(
        pytest_options, path="pyproject.toml [tool.pytest.ini_options]", errors=errors
    )
    markers = options.get("markers", ()) if options else ()
    declared = {
        marker.split(":", maxsplit=1)[0].strip()
        for marker in markers
        if isinstance(marker, str) and ":" in marker
    }
    missing = sorted({"unit", "integration", "remote", "slow"} - declared)
    if missing:
        errors.append(f"pyproject.toml: missing documented pytest markers: {', '.join(missing)}")


def check_governance(root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Return every governance contract failure without mutating the repository."""

    root = root.resolve()
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"missing required governance file: {relative}")
    if errors:
        return tuple(errors)

    try:
        document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        return (f"pyproject.toml: unable to parse: {error}",)

    _check_license(root, document, errors)
    _check_citation(root, errors)
    _check_templates(root, errors)
    _check_policy_text(root, errors)
    _check_markdown_links(root, errors)
    _check_pytest_markers(document, errors)
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    errors = check_governance(args.root)
    if errors:
        for error in errors:
            print(f"governance error: {error}", file=sys.stderr)
        return 1
    print("governance contracts: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
