"""Validate the digest-bound cylinder report and its rendered companion."""

from __future__ import annotations

import argparse
from pathlib import Path

from soufflerie.artifacts import safe_read_json
from soufflerie.errors import SoufflerieError
from soufflerie.solver.cylinder_acceptance import (
    CylinderAcceptanceReport,
    render_cylinder_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="schema-v1 cylinder report JSON")
    args = parser.parse_args()
    report_path = args.report.resolve()
    try:
        report = safe_read_json(
            report_path.parent,
            report_path.name,
            model=CylinderAcceptanceReport,
        )
    except (OSError, SoufflerieError, ValueError) as error:
        parser.error(f"invalid cylinder report: {error}")
    rendered_path = report_path.with_suffix(".md")
    try:
        rendered = rendered_path.read_text(encoding="utf-8")
    except OSError as error:
        parser.error(f"unable to read rendered companion {rendered_path}: {error}")
    if rendered != render_cylinder_report(report):
        parser.error(f"{rendered_path} does not match the typed JSON report")
    if not report.overall_passed:
        parser.error("cylinder acceptance report is red")
    print(
        "cylinder_acceptance=PASS "
        f"source={report.source_revision} "
        f"report_sha256={report.report_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
