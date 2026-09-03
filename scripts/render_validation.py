"""Render or verify deterministic validation Markdown, SVGs, and plot manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from soufflerie.errors import ArtifactIntegrityError
from soufflerie.validation import (
    PLOT_DEFINITIONS,
    check_validation_artifacts,
    load_validation_report,
    render_validation_artifacts,
    write_validation_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="canonical ValidationReport JSON input")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare all checked-in outputs without writing",
    )
    args = parser.parse_args()
    report_path: Path = args.report
    try:
        report = load_validation_report(report_path)
        artifacts = render_validation_artifacts(
            report,
            plot_directory=report_path.with_suffix(".plots").name,
        )
        if args.check:
            errors = check_validation_artifacts(report_path, artifacts)
            if errors:
                parser.error("; ".join(errors))
        else:
            write_validation_artifacts(report_path, artifacts)
    except ArtifactIntegrityError as error:
        parser.error(str(error))
    action = "verified" if args.check else "rendered"
    print(
        f"validation_report=PASS action={action} plots={len(PLOT_DEFINITIONS)} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
