# RFC-0013: Testing, packaging, and release

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

Soufflerie ships one typed Python 3.11 distribution under Apache-2.0 through CPU-only pull-request gates and separately recorded remote GPU acceptance. Release `v0.1.0` requires a clean source, exact lock, numerical/ML/service evidence, installed-artifact tests, governance documents, SBOM/provenance, and claim-to-evidence review.

## Motivation

The PRD defines a public repository, fresh-clone commands, checked-in validation evidence, a tag, README visuals, and honest limitations. Specs alone do not guarantee that wheels contain schemas/checkpoints, editable installs are not masking errors, CI avoids paid resources, or a release claim matches the validated source.

## Goals

- Define package metadata, dependency profiles, public CLI, and distribution contents.
- Split fast CPU pull-request validation from authenticated release acceptance.
- Make numerical, ML, service, security, docs, and package gates explicit.
- Ensure fresh-clone and installed-wheel workflows match documentation.
- Establish versioning, changelog, deprecation, contribution, security, and release ownership.

## Non-Goals

- Publishing v1.0 stability guarantees in the first release.
- Running paid GPU jobs for untrusted pull requests.
- Bundling the full training dataset or all training checkpoints in the wheel.
- Publishing optional build-log posts as a release gate.

## Proposed Design

`pyproject.toml` is the only packaging metadata source and follows RFC-0001. It declares distribution `soufflerie`, dynamic-free version `0.1.0` at release, Python range, Apache-2.0 expression, README, authors, classifiers, URLs, console script, package data, optional profiles, and dev group. `uv.lock` is committed and immutable in frozen builds. `src/soufflerie/py.typed`, `schemas/v1/*.json`, smoke configs, bundle metadata, and the small checkpoint/reference are package data with explicit inclusion tests.

Repository release/governance artifacts are:

```text
README.md
LICENSE
NOTICE
CHANGELOG.md
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
CITATION.cff
MAINTAINERS.md
.github/CODEOWNERS
.github/pull_request_template.md
.github/ISSUE_TEMPLATE/{bug,feature,config}.yml
.pre-commit-config.yaml
```

README order is limitations-aware: exact scope, three evidence-backed visuals, validation status/table with report link, install profiles, CPU quickstart, authenticated remote workflow, architecture, reproducibility/provenance, measured performance/cost, limitations, security/support, citation, license. It may say only what checked-in reports and benchmarks prove. GIF scripts and source artifact identities are documented.

Pull-request CI jobs are:

1. `lint-type`: lock check, Ruff lint/format, strict mypy, schema/docs link validation.
2. `unit-contract`: Python 3.11 CPU unit/property/contract/numerical smoke with coverage floor 90%.
3. `integration`: miniature CPU solver-to-report path, API/UI fixture tests, no network.
4. `package`: wheel/sdist build, content policy, clean-install CLI/import/bundled prediction.
5. `security`: secret scan, dependency vulnerability scan, safe-artifact fuzz/regressions.

Jobs use least GitHub permissions, pinned action commit SHAs, concurrency cancellation for superseded branch runs, caches keyed by lock digest, and no remote credentials. Test markers are `unit`, `integration`, `remote`, `slow`; unmarked tests default to CPU-safe. CI explicitly selects `not remote` and a policy test fails if a remote test lacks the marker.

The local gate is wrapped by `scripts/check.sh` and runs the commands in [`07-testing-strategy.md#ci-gates`](../spec/07-testing-strategy.md#ci-gates). The script fails fast only after preserving individual command output in CI; it never reports the full repository green from a subset. Docs validation checks internal anchors, issue/spec links where network-free, forbidden unresolved tokens, generated-report cleanliness, and public-claim references.

Remote release acceptance is manually dispatched only from a clean reviewed commit and records:

- real remote kernel smoke;
- canonical Poiseuille, mass, determinism, and cylinder evidence;
- canonical 1,000-case dataset manifest/statistics;
- three-seed training, baselines, selected bundle, and budget;
- full validation JSON/Markdown/plots and every gate;
- L40S solver/prediction and named CPU prediction benchmarks;
- deployed health, prediction, red-state fixture, solve/SSE, and browser smoke;
- total GPU seconds and dated cost calculation.

An acceptance index binds every output digest to the commit/lock/device. Failure leaves the release blocked; it never edits thresholds. Generated evidence is checked into `reports/` when reasonably sized, while large parents use immutable digests/release assets.

The release workflow builds from the annotated tag in a clean environment, reruns CPU gates, produces wheel/sdist, SHA-256 sums, CycloneDX SBOM, and build provenance, verifies artifacts in a new environment, then publishes the GitHub release. Package-index publication is optional for v0.1 but, if enabled, uses trusted publishing and the identical verified artifacts. The release notes include validation state and known limitations before favorable performance.

Fresh-clone acceptance executes:

```text
uv sync
uv run pytest -m "not remote"
modal run infra/sweep.py --config configs/sweeps/mvp-v1.yaml --n 8
modal deploy infra/serve.py
uv run soufflerie demo
```

Remote commands require separately documented authentication; local test/demo paths must not. The deployed URL is ephemeral evidence, not an uptime promise. The release owner checks repository visibility, license detection, tag/commit equality, asset checksums, and live endpoint smoke after publication.

Compatibility/deprecation follow [`09-release-and-versioning.md`](../spec/09-release-and-versioning.md). Every user-visible change receives a Keep-a-Changelog-style entry. Numerical/data/model/report schema changes also name artifact invalidation or migration. `main` remains releasable; branch protection and required checks are configured before implementation merges.

## Alternatives Considered

### Include remote GPU tests in every pull request

It would catch integration drift sooner but exposes paid credentials to workflow risk, adds nondeterministic capacity failures, and slows review. CPU contracts plus protected manual acceptance give a clearer trust boundary.

### Editable-install-only testing

It is faster but misses wheel metadata, package data, console scripts, and undeclared imports. A clean installed-wheel smoke is mandatory.

### Ship full dataset/checkpoints in Git

It makes cloning self-contained but bloats history and conflicts with the sub-2 GiB dataset scale. Manifests/digests/evidence ship; large artifacts use immutable external release storage.

### Delay governance/security files until later

The repository is public from the first release, so reporting, contribution, conduct, ownership, citation, and license discipline are part of v0.1 rather than polish.

## Drawbacks

- The release gate is heavier than the implementation size might suggest.
- Remote acceptance remains a manual, credentialed step.
- A 90% coverage floor may require deliberate fixture design for framework adapters.
- Exact action/dependency pins require regular maintenance.

## Migration / Rollout

1. Land packaging scaffold, governance, pre-commit, CPU CI, and branch protection before domain implementation.
2. Add contract/integration/package/security jobs as their surfaces land.
3. Add protected remote acceptance workflow after infrastructure smoke is proven.
4. Generate final README/reports/assets only from the release-candidate revision.
5. Tag and publish `v0.1.0` after the full gate and post-publish smoke.

No earlier releases exist. Future releases use the same artifact and changelog policy.

## Testing Strategy

- Validate `pyproject.toml`, lock/Python alignment, extras, project URLs, and license expression.
- Build wheel/sdist twice in clean environments and compare normalized contents/digests where reproducible.
- Assert required and forbidden distribution members; scan for secrets, caches, data dumps, and unsafe checkpoints.
- Install each profile and test console script, import isolation, schemas, and bundled prediction outside the checkout.
- Self-test CI policy: no remote invocation/secret on pull requests, pinned actions, least permissions, correct markers.
- Run docs anchor/claim/generated-artifact checks and README command smoke.
- Execute the full remote acceptance checklist against one immutable commit.
- Verify annotated tag, release assets, checksums, SBOM/provenance, and post-publish fresh clone.

## Open Questions

None for v0.1. Package-index publication is optional and does not block the GitHub release; the release maintainer records the decision in release notes before tagging.

## References

- [`prd.md#11-definition-of-done`](../../prd.md#11-definition-of-done)
- [`07-testing-strategy.md`](../spec/07-testing-strategy.md)
- [`09-release-and-versioning.md`](../spec/09-release-and-versioning.md)
- [RFC-0001](RFC-0001-package-and-runtime-boundaries.md)
- [RFC-0011](RFC-0011-remote-execution-and-persistence.md)
- [RFC-0012](RFC-0012-observability-provenance-and-security.md)
