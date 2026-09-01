# Contributing to Soufflerie

Thank you for helping make the simulation-to-surrogate workflow more
reproducible and inspectable. Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). Security vulnerabilities belong in the
[private reporting channel](SECURITY.md), not a public issue.

## Start from the specification

`SPEC.md` is the normative index. Before changing behavior, find the owning
specification, accepted RFC, and implementation issue. Open a focused issue when
none exists. A behavioral change that conflicts with the specification must
update that specification or add an accepted RFC in the same pull request; an
implementation must not silently relax a numerical, validation, safety, or
resource threshold.

Use the structured issue forms for bugs, features, and configuration changes.
Keep one shippable concern per branch and pull request, link the issue, and
explain the problem, solution, validation, and caveats.

## Development setup

Soufflerie v0.1 supports CPython `>=3.11,<3.12` and uses the committed lockfile.

```bash
git clone https://github.com/AbdelStark/soufflerie.git
cd soufflerie
uv sync --frozen
uv run soufflerie version
```

Never commit `.env`, credentials, local artifacts, caches, or remote-provider
state. Use only synthetic sentinel values in security tests.

## Validation

Run the narrowest relevant tests while developing, then the complete CPU-safe
gate before requesting review:

```bash
uv lock --check
uv run ruff check .
uv run mypy
uv run python scripts/export_schemas.py --check
uv run python scripts/check_governance.py
uv run pytest -m "not remote"
```

Tests without a marker must be deterministic and safe on a CPU-only pull
request runner. Use `unit` for isolated behavior, `integration` for bounded
multi-component paths, `slow` for intentionally longer local checks, and
`remote` for tests that may authenticate, allocate paid resources, or require a
GPU. Every remote test must carry `remote`; ordinary CI excludes it. Do not make
network access, credentials, or paid execution a prerequisite for the default
test suite.

## Numerical and generated evidence

Numerical golden files, schema exports, reports, plots, and benchmark summaries
are reviewed evidence, not snapshots to refresh until tests pass.

- Regenerate them only with the checked-in canonical script and configuration.
- Record the source revision, lock digest, command, device class, and affected
  artifact identities.
- Review semantic differences and include the generated diff in the pull
  request. Never hand-edit generated values.
- A tolerance, domain, seed, split, or release-gate change requires its owning
  specification/RFC update and explicit rationale.
- Do not replace a red result with a favorable fixture or describe a CPU smoke
  test as remote or release acceptance.

## Commit certification and licensing

This project uses the [Developer Certificate of Origin 1.1](https://developercertificate.org/)
instead of a contributor license agreement. Sign off every commit:

```bash
git commit -s
```

The `Signed-off-by` trailer certifies that you have the right to submit the
contribution under the repository's [Apache-2.0 license](LICENSE). Preserve
third-party notices and identify newly introduced third-party code or data,
including its license and origin, in the pull request.

## Pull-request review

A reviewable pull request:

- links its issue and owning specification/RFC;
- keeps public API, schemas, docs, and tests synchronized;
- includes exact validation commands and distinguishes narrow from full gates;
- calls out artifact invalidation, compatibility, security, cost, and evidence
  implications;
- updates user-visible release notes when the changelog exists and the change
  is user-visible; and
- contains only commits carrying a DCO sign-off.

The active maintainers and decision process are listed in
[MAINTAINERS.md](MAINTAINERS.md). Maintainer review is required for merge.
