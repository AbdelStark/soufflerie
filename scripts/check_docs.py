"""Validate local documentation links, anchors, placeholders, and JSON evidence."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).parents[1]
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:name|id)=[\"']([^\"']+)[\"']\s*></a>", re.IGNORECASE)
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
INLINE_LINK = re.compile(r"\[([^]]+)\]\([^)]+\)")
INLINE_HTML = re.compile(r"<[^>]+>")
UNRESOLVED = re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b|\[INSERT|<INSERT", re.IGNORECASE)
CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
PUBLIC_CLAIM_FORBIDDEN = (
    re.compile(r"\bground truth\b", re.IGNORECASE),
    re.compile(r"\bengineering[- ]grade\b", re.IGNORECASE),
    re.compile(r"\blocal CUDA\b", re.IGNORECASE),
)
PUBLIC_DOC_ROOTS = (
    "README.md",
    "configs/README.md",
    "docs/cli.md",
    "docs/operations",
    "docs/security",
    "reports",
)


def markdown_documents(root: Path) -> tuple[Path, ...]:
    """Return the checked public/specification Markdown corpus."""

    candidates = {
        *(path for path in root.glob("*.md") if path.is_file()),
        *(path for path in (root / "docs").rglob("*.md") if path.is_file()),
        *(path for path in (root / "reports").rglob("*.md") if path.is_file()),
        *(path for path in (root / "configs").rglob("*.md") if path.is_file()),
        *(path for path in (root / ".github").rglob("*.md") if path.is_file()),
    }
    return tuple(sorted(candidates))


def _github_slug(heading: str) -> str:
    value = INLINE_LINK.sub(r"\1", heading)
    value = INLINE_HTML.sub("", value)
    value = value.replace("`", "").casefold().strip()
    punctuation = string.punctuation.replace("-", "").replace("_", "")
    value = value.translate(str.maketrans("", "", punctuation))
    return re.sub(r"[\s-]+", "-", value).strip("-")


def document_anchors(path: Path) -> frozenset[str]:
    """Approximate GitHub heading anchors and include explicit stable anchors."""

    text = path.read_text(encoding="utf-8")
    anchors = {match.casefold() for match in EXPLICIT_ANCHOR.findall(text)}
    counts: Counter[str] = Counter()
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match is None:
            continue
        base = _github_slug(match.group(1))
        if not base:
            continue
        duplicate = counts[base]
        counts[base] += 1
        anchors.add(base if duplicate == 0 else f"{base}-{duplicate}")
    return frozenset(anchors)


def _link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    # Markdown permits an optional quoted title after a whitespace separator.
    return target.split(maxsplit=1)[0]


def check_document(path: Path, *, root: Path) -> tuple[str, ...]:
    errors: list[str] = []
    relative = path.relative_to(root)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return (f"{relative}: unable to read UTF-8 documentation: {error}",)
    if text and not text.endswith("\n"):
        errors.append(f"{relative}: file must end with one newline")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.rstrip() != line:
            errors.append(f"{relative}:{line_number}: trailing whitespace")
        if any(marker in line for marker in CONFLICT_MARKERS):
            errors.append(f"{relative}:{line_number}: unresolved merge-conflict marker")
        if UNRESOLVED.search(line):
            errors.append(f"{relative}:{line_number}: unresolved documentation placeholder")

    for raw_target in LINK_PATTERN.findall(text):
        target = _link_target(raw_target)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            if parsed.scheme not in {"http", "https", "mailto"}:
                errors.append(f"{relative}: unsupported link scheme in {target!r}")
            continue
        decoded_path = unquote(parsed.path)
        if decoded_path:
            destination = (path.parent / decoded_path).resolve()
            try:
                destination.relative_to(root)
            except ValueError:
                errors.append(f"{relative}: link escapes repository root: {target!r}")
                continue
        else:
            destination = path
        if not destination.is_file():
            errors.append(f"{relative}: broken relative link target {target!r}")
            continue
        if parsed.fragment:
            fragment = unquote(parsed.fragment).casefold()
            if fragment not in document_anchors(destination):
                destination_relative = destination.relative_to(root)
                errors.append(
                    f"{relative}: missing anchor #{parsed.fragment} in {destination_relative}"
                )
    return tuple(errors)


def _is_public_claim_document(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return any(
        relative == prefix or relative.startswith(f"{prefix}/") for prefix in PUBLIC_DOC_ROOTS
    )


def check_public_claim_language(path: Path, *, root: Path) -> tuple[str, ...]:
    if not _is_public_claim_document(path, root):
        return ()
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in PUBLIC_CLAIM_FORBIDDEN:
            if pattern.search(line):
                errors.append(
                    f"{path.relative_to(root)}:{line_number}: reserved public claim language "
                    f"matches {pattern.pattern!r}"
                )
    return tuple(errors)


def check_json_documents(root: Path) -> tuple[str, ...]:
    errors: list[str] = []
    paths = sorted((root / "schemas").rglob("*.json")) + sorted((root / "reports").rglob("*.json"))
    for path in paths:
        try:
            json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {value}")
                ),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{path.relative_to(root)}: invalid finite JSON document: {error}")
    return tuple(errors)


def check_docs(root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Return every local docs/evidence failure without network access."""

    root = root.resolve()
    errors: list[str] = []
    for path in markdown_documents(root):
        errors.extend(check_document(path, root=root))
        errors.extend(check_public_claim_language(path, root=root))
    errors.extend(check_json_documents(root))
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    errors = check_docs(args.root)
    if errors:
        for error in errors:
            print(f"docs error: {error}", file=sys.stderr)
        return 1
    print("documentation contracts: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
