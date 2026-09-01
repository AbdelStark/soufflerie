"""Generate or verify the checked-in schema-v1 JSON documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from soufflerie.config import rendered_config_schema_documents
from soufflerie.observability import rendered_observability_schema_documents
from soufflerie.schemas import rendered_schema_documents

PROJECT_ROOT = Path(__file__).parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "schemas" / "v1"


def export(*, check: bool) -> tuple[str, ...]:
    documents = {
        **rendered_schema_documents(),
        **rendered_config_schema_documents(),
        **rendered_observability_schema_documents(),
    }
    stale: list[str] = []
    if check:
        actual_names = {path.name for path in OUTPUT_ROOT.glob("*.json")}
        expected_names = set(documents)
        stale.extend(sorted(actual_names ^ expected_names))
        for name, expected in documents.items():
            path = OUTPUT_ROOT / name
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(name)
        return tuple(sorted(set(stale)))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, content in documents.items():
        path = OUTPUT_ROOT / name
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return ()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when checked-in schemas differ")
    args = parser.parse_args()
    stale = export(check=args.check)
    if stale:
        parser.error(f"schema exports are stale: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
