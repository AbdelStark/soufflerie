#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${repository_root}"

uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run python scripts/generate_bundled_model.py --check
UV_OFFLINE=1 uv run pytest -m "not remote" --cov=src/soufflerie --cov-report=term-missing --cov-fail-under=90
uv run python scripts/validate_schemas.py
uv run python scripts/render_validation.py --check tests/fixtures/report.json
uv run python scripts/check_docs.py
uv build
uv run python scripts/check_distribution.py dist/*
