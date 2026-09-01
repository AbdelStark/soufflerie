from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from pathlib import Path

from soufflerie.solver.numerical_gates import (
    generate_cpu_gate_summary,
    render_cpu_gate_summary,
)

DEFAULT_REPORT = Path("reports/solver/cpu-gates.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic CPU solver gate evidence")
    parser.add_argument(
        "--check",
        type=Path,
        metavar="PATH",
        help="compare generated evidence with PATH instead of printing it",
    )
    args = parser.parse_args()
    # Warp reports device/JIT status on stdout. Preserve that operator feedback
    # on stderr while keeping stdout valid JSON for generation pipelines.
    with redirect_stdout(sys.stderr):
        summary = generate_cpu_gate_summary()
    rendered = render_cpu_gate_summary(summary)
    if args.check is None:
        print(rendered, end="")
        return 0
    try:
        current = args.check.read_text(encoding="utf-8")
    except OSError as exc:
        parser.error(f"unable to read {args.check}: {exc}")
    if current != rendered:
        parser.error(
            f"{args.check} is stale for this exact platform; review the numerical rationale "
            "before updating the golden"
        )
    print(f"cpu_solver_gates=PASS report={args.check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
