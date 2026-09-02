"""Generate or verify the deterministic canonical design summary."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from soufflerie.config import SweepConfig, load_config
from soufflerie.datagen.design import generate_design_summary, render_design_summary

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sweeps" / "mvp-v1.yaml"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--check", type=Path, metavar="PATH")
    destination.add_argument("--output", type=Path, metavar="PATH")
    args = parser.parse_args()

    config = load_config(args.config, SweepConfig)
    rendered = render_design_summary(generate_design_summary(config))
    if args.check is not None:
        try:
            current = args.check.read_text(encoding="utf-8")
        except OSError as exc:
            parser.error(f"unable to read {args.check}: {exc}")
        if current != rendered:
            parser.error(f"{args.check} is stale for the locked canonical design")
        print(f"design_summary=PASS report={args.check}")
        return 0
    if args.output is not None:
        _write_atomic(args.output, rendered)
        print(f"design_summary=WRITTEN report={args.output}")
        return 0
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
