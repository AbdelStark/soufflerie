# Implementation Tracker — 2026-09-01

Generated from the specification corpus in [PR #1](https://github.com/AbdelStark/soufflerie/pull/1), fixed at specification commit `0b184d49c13ff7a870f269c15928b87dccf8f41a`. Every implementable v0.1 unit is filed below. Each implementation issue is scoped to one shippable pull request; its body contains immutable spec links, mechanically verifiable acceptance criteria, and explicit dependencies.

<a id="summary"></a>
## Summary

- Milestone: [`v0.1`](https://github.com/AbdelStark/soufflerie/milestone/1)
- Implementation issues: 56
- RFC tracking issues: 13
- Total issues: 69
- Current state: 49 open, 20 closed
- Scope: PRD milestones M0-M5; stretch tracks excluded

Implementation issues by area: foundation 4, solver 4, geometry 5, datagen 5, surrogate 4, training 4, validation 4, API 5, UI 4, infrastructure/performance 6, observability 1, security 3, and release 7.

<a id="milestone-v01"></a>
## Milestone: v0.1

| # | Title | Area | Priority | Effort | RFC | Status |
|---|---|---|---|---|---|---|
| #2 | [foundation: scaffold the typed package and dependency profiles](https://github.com/AbdelStark/soufflerie/issues/2) | foundation | p0 | l | RFC-0001 | closed |
| #3 | [foundation: define canonical schemas and artifact identities](https://github.com/AbdelStark/soufflerie/issues/3) | foundation | p0 | l | RFC-0001 | closed |
| #4 | [foundation: implement strict configuration and schema export](https://github.com/AbdelStark/soufflerie/issues/4) | foundation | p0 | m | RFC-0001, RFC-0004 | closed |
| #5 | [cli: expose installed public commands and import isolation](https://github.com/AbdelStark/soufflerie/issues/5) | foundation | p1 | m | RFC-0001 | closed |
| #6 | [solver: implement D2Q9 configuration and NumPy oracle](https://github.com/AbdelStark/soufflerie/issues/6) | solver | p0 | m | RFC-0002 | closed |
| #7 | [solver: implement deterministic collision and pull-stream kernels](https://github.com/AbdelStark/soufflerie/issues/7) | solver | p0 | l | RFC-0002 | closed |
| #8 | [solver: add lifecycle, diagnostics, and time averaging](https://github.com/AbdelStark/soufflerie/issues/8) | solver | p0 | l | RFC-0002 | closed |
| #9 | [solver: verify channel, conservation, and determinism gates](https://github.com/AbdelStark/soufflerie/issues/9) | solver | p0 | l | RFC-0002 | closed |
| #10 | [geometry: implement ellipse SDF, masks, and preflight](https://github.com/AbdelStark/soufflerie/issues/10) | geometry | p0 | m | RFC-0003 | closed |
| #11 | [solver: implement channel inlet, outlet, walls, and sponge](https://github.com/AbdelStark/soufflerie/issues/11) | geometry | p0 | l | RFC-0003 | closed |
| #12 | [solver: implement obstacle bounce-back and force reduction](https://github.com/AbdelStark/soufflerie/issues/12) | geometry | p0 | l | RFC-0003 | closed |
| #13 | [solver: implement Strouhal and field-drag diagnostics](https://github.com/AbdelStark/soufflerie/issues/13) | geometry | p1 | m | RFC-0003 | closed |
| #14 | [solver: pass cylinder reference and grid-sensitivity acceptance](https://github.com/AbdelStark/soufflerie/issues/14) | geometry | p0 | l | RFC-0002, RFC-0003, RFC-0011 | closed |
| #15 | [datagen: implement deterministic LHS design and frozen splits](https://github.com/AbdelStark/soufflerie/issues/15) | datagen | p0 | m | RFC-0004 | closed |
| #16 | [datagen: implement safe run codecs and local artifact store](https://github.com/AbdelStark/soufflerie/issues/16) | datagen | p0 | l | RFC-0005 | closed |
| #17 | [datagen: implement leased resumable sweep state](https://github.com/AbdelStark/soufflerie/issues/17) | datagen | p0 | l | RFC-0005 | closed |
| #18 | [datagen: build and validate the immutable dataset manifest](https://github.com/AbdelStark/soufflerie/issues/18) | datagen | p0 | l | RFC-0005 | closed |
| #19 | [datagen: produce the canonical 1,000-case dataset](https://github.com/AbdelStark/soufflerie/issues/19) | datagen | p0 | l | RFC-0005, RFC-0011 | closed |
| #20 | [surrogate: implement leakage-safe preprocessing statistics](https://github.com/AbdelStark/soufflerie/issues/20) | surrogate | p0 | m | RFC-0006 | closed |
| #21 | [surrogate: implement the fixed FNO and drag head](https://github.com/AbdelStark/soufflerie/issues/21) | surrogate | p0 | l | RFC-0006 | closed |
| #22 | [surrogate: implement safe model bundle export and load](https://github.com/AbdelStark/soufflerie/issues/22) | surrogate | p0 | l | RFC-0006 | open |
| #23 | [surrogate: verify the bundled CPU inference contract](https://github.com/AbdelStark/soufflerie/issues/23) | surrogate | p1 | m | RFC-0006, RFC-0013 | open |
| #24 | [training: implement manifest loader and deterministic baselines](https://github.com/AbdelStark/soufflerie/issues/24) | training | p0 | m | RFC-0007 | open |
| #25 | [training: implement deterministic mixed-precision optimization](https://github.com/AbdelStark/soufflerie/issues/25) | training | p0 | l | RFC-0007 | open |
| #26 | [training: add checkpoint resume and validation selection](https://github.com/AbdelStark/soufflerie/issues/26) | training | p0 | l | RFC-0007 | open |
| #27 | [training: execute three canonical seeds and select the model](https://github.com/AbdelStark/soufflerie/issues/27) | training | p0 | l | RFC-0007, RFC-0011 | open |
| #28 | [validation: implement metrics and immutable gate evaluation](https://github.com/AbdelStark/soufflerie/issues/28) | validation | p0 | l | RFC-0008 | open |
| #29 | [validation: implement OOD ensemble and sensitivity probes](https://github.com/AbdelStark/soufflerie/issues/29) | validation | p0 | l | RFC-0008 | open |
| #30 | [validation: generate immutable JSON, Markdown, and plots](https://github.com/AbdelStark/soufflerie/issues/30) | validation | p0 | l | RFC-0008, RFC-0012 | open |
| #31 | [validation: run the canonical release evaluation](https://github.com/AbdelStark/soufflerie/issues/31) | validation | p0 | l | RFC-0008, RFC-0011 | open |
| #32 | [api: implement strict HTTP schemas, errors, and health](https://github.com/AbdelStark/soufflerie/issues/32) | api | p0 | m | RFC-0009 | open |
| #33 | [api: implement bounded prediction and field encoding](https://github.com/AbdelStark/soufflerie/issues/33) | api | p0 | l | RFC-0009 | open |
| #34 | [api: implement solve jobs and replayable SSE state](https://github.com/AbdelStark/soufflerie/issues/34) | api | p0 | l | RFC-0009 | open |
| #35 | [api: connect remote solves with admission and result comparison](https://github.com/AbdelStark/soufflerie/issues/35) | api | p0 | l | RFC-0009, RFC-0011, RFC-0012 | open |
| #36 | [api: verify contracts, load isolation, and hostile inputs](https://github.com/AbdelStark/soufflerie/issues/36) | api | p1 | m | RFC-0009, RFC-0012 | open |
| #37 | [ui: implement deterministic field and comparison rendering](https://github.com/AbdelStark/soufflerie/issues/37) | ui | p1 | m | RFC-0010 | open |
| #38 | [ui: build controls and persistent validation states](https://github.com/AbdelStark/soufflerie/issues/38) | ui | p1 | l | RFC-0010 | open |
| #39 | [ui: integrate solve comparison and local CPU demo](https://github.com/AbdelStark/soufflerie/issues/39) | ui | p1 | l | RFC-0010 | open |
| #40 | [docs: generate the three evidence-backed README visuals](https://github.com/AbdelStark/soufflerie/issues/40) | ui | p2 | m | RFC-0010, RFC-0013 | open |
| #41 | [infra: define the locked remote runtime and persistent volume](https://github.com/AbdelStark/soufflerie/issues/41) | infra | p0 | l | RFC-0011 | closed |
| #42 | [infra: expose idempotent remote solve and sweep entrypoints](https://github.com/AbdelStark/soufflerie/issues/42) | infra | p0 | l | RFC-0002, RFC-0005, RFC-0011 | closed |
| #43 | [infra: expose remote training and validation entrypoints](https://github.com/AbdelStark/soufflerie/issues/43) | infra | p0 | l | RFC-0007, RFC-0008, RFC-0011 | open |
| #44 | [infra: deploy the mounted GPU service and run end-to-end smoke](https://github.com/AbdelStark/soufflerie/issues/44) | infra | p0 | l | RFC-0009, RFC-0010, RFC-0011 | open |
| #45 | [infra: prove volume atomicity, fallback, and recovery](https://github.com/AbdelStark/soufflerie/issues/45) | infra | p0 | m | RFC-0011 | open |
| #46 | [perf: benchmark solver, training, inference, and resource budgets](https://github.com/AbdelStark/soufflerie/issues/46) | infra | p1 | l | RFC-0011, RFC-0013 | open |
| #47 | [observability: implement events, metrics, timers, and redaction](https://github.com/AbdelStark/soufflerie/issues/47) | observability | p0 | l | RFC-0012 | closed |
| #48 | [security: implement provenance, lineage, and safe readers](https://github.com/AbdelStark/soufflerie/issues/48) | security | p0 | l | RFC-0012 | closed |
| #49 | [security: enforce rate, budget, and readiness controls](https://github.com/AbdelStark/soufflerie/issues/49) | security | p0 | m | RFC-0012 | open |
| #50 | [security: add hostile-artifact, secret, and dependency gates](https://github.com/AbdelStark/soufflerie/issues/50) | security | p0 | m | RFC-0012, RFC-0013 | open |
| #51 | [docs: add license, governance, security, and citation files](https://github.com/AbdelStark/soufflerie/issues/51) | release | p1 | m | RFC-0013 | closed |
| #52 | [ci: enforce CPU lint, typing, tests, schemas, and docs](https://github.com/AbdelStark/soufflerie/issues/52) | release | p0 | l | RFC-0013 | closed |
| #53 | [release: verify distributions and installed-wheel behavior](https://github.com/AbdelStark/soufflerie/issues/53) | release | p0 | m | RFC-0013 | open |
| #54 | [ci: add protected security and remote acceptance workflows](https://github.com/AbdelStark/soufflerie/issues/54) | release | p0 | l | RFC-0011, RFC-0012, RFC-0013 | open |
| #55 | [docs: author evidence-bound README and fresh-clone guides](https://github.com/AbdelStark/soufflerie/issues/55) | release | p1 | l | RFC-0013 | open |
| #56 | [release: automate changelog, SBOM, provenance, and artifacts](https://github.com/AbdelStark/soufflerie/issues/56) | release | p1 | l | RFC-0013 | open |
| #57 | [release: pass acceptance and publish v0.1.0](https://github.com/AbdelStark/soufflerie/issues/57) | release | p0 | l | RFC-0013 | open |

<a id="rfc-coverage"></a>
## RFC coverage

Counts overlap when one shippable issue implements multiple RFC boundaries.

| RFC | Implementation issues |
|---|---:|
| RFC-0001 | 4 |
| RFC-0002 | 6 |
| RFC-0003 | 5 |
| RFC-0004 | 2 |
| RFC-0005 | 5 |
| RFC-0006 | 4 |
| RFC-0007 | 5 |
| RFC-0008 | 5 |
| RFC-0009 | 6 |
| RFC-0010 | 5 |
| RFC-0011 | 12 |
| RFC-0012 | 8 |
| RFC-0013 | 11 |

<a id="normative-surface-coverage"></a>
## Normative surface coverage

This matrix audits supporting-spec requirements that cut across RFC boundaries. Terminology-only content has documentation/claim-validation work rather than a separate code issue.

| Specification | Implementable concerns | Issues |
|---|---|---|
| [`00-overview.md`](../spec/00-overview.md) | v0.1 goals, success gates, risk resolution, contributor handoff | #2-#57 through the issue graph; terminal audit #57 |
| [`01-architecture.md`](../spec/01-architecture.md) | module layout, dependency direction, state, concurrency, execution planes | #2, #7, #17, #21, #41-#45, #52 |
| [`02-public-api.md`](../spec/02-public-api.md) | Python/CLI/HTTP/artifact surfaces and input limits | #3-#5, #22, #32-#39, #42-#44 |
| [`03-data-model.md`](../spec/03-data-model.md) | records, arrays, units, identities, schemas, invariants | #3, #6, #10, #16, #18, #20, #22, #30, #48 |
| [`04-error-model.md`](../spec/04-error-model.md) | typed failures, recovery, retries, HTTP and CLI mappings | #5, #8, #17, #26, #32, #34-#35, #42-#45 |
| [`05-observability.md`](../spec/05-observability.md) | events, metrics, traces, provenance, redaction, retention | #30, #35, #46-#50 |
| [`06-security.md`](../spec/06-security.md) | trust boundaries, artifact safety, secrets, abuse, vulnerability gates | #16, #22, #32, #35-#36, #45, #48-#50, #54 |
| [`07-testing-strategy.md`](../spec/07-testing-strategy.md) | numerical/ML/service/security/package test layers and CI | #9, #14, #23, #28-#31, #36, #45, #50, #52-#54, #57 |
| [`08-performance-budget.md`](../spec/08-performance-budget.md) | reference workloads, resource caps, profiling, regression policy | #14, #19, #27, #33, #44, #46 |
| [`09-release-and-versioning.md`](../spec/09-release-and-versioning.md) | dependencies, compatibility, governance, distributions, release | #2, #5, #23, #40, #50-#57 |
| [`10-glossary.md`](../spec/10-glossary.md) | canonical schema/API/docs terminology and claim language | #3, #5, #32, #55 |

<a id="tracking-issues"></a>
## Tracking Issues

- #58 [RFC-0001 tracking](https://github.com/AbdelStark/soufflerie/issues/58): Package and runtime boundaries (closed; #2-#5 complete)
- #59 [RFC-0002 tracking](https://github.com/AbdelStark/soufflerie/issues/59): D2Q9 lattice Boltzmann core
- #60 [RFC-0003 tracking](https://github.com/AbdelStark/soufflerie/issues/60): Geometry, boundaries, and force diagnostics
- #61 [RFC-0004 tracking](https://github.com/AbdelStark/soufflerie/issues/61): Experiment configuration and design of experiments
- #62 [RFC-0005 tracking](https://github.com/AbdelStark/soufflerie/issues/62): Dataset artifacts and resumable sweep lifecycle
- #63 [RFC-0006 tracking](https://github.com/AbdelStark/soufflerie/issues/63): FNO surrogate and checkpoints
- #64 [RFC-0007 tracking](https://github.com/AbdelStark/soufflerie/issues/64): Training, baselines, and reproducibility
- #65 [RFC-0008 tracking](https://github.com/AbdelStark/soufflerie/issues/65): Validation metrics and release gates
- #66 [RFC-0009 tracking](https://github.com/AbdelStark/soufflerie/issues/66): Inference and solve API
- #67 [RFC-0010 tracking](https://github.com/AbdelStark/soufflerie/issues/67): Interactive demo and visualization
- #68 [RFC-0011 tracking](https://github.com/AbdelStark/soufflerie/issues/68): Remote execution and persistence
- #69 [RFC-0012 tracking](https://github.com/AbdelStark/soufflerie/issues/69): Observability, provenance, and security
- #70 [RFC-0013 tracking](https://github.com/AbdelStark/soufflerie/issues/70): Testing, packaging, and release

<a id="cross-cutting-dependencies"></a>
## Cross-Cutting Dependencies

- #2 establishes the package/runtime boundary used by schemas (#3), configuration (#4), remote runtime (#41), and CPU CI (#52).
- The solver chain #6-#14 enables remote sweep execution (#42), the canonical dataset (#19), field validation (#28), and performance evidence (#46).
- Frozen design/data issues #15-#19 enable preprocessing/model/training #20-#27; model selection then enables OOD and release validation #29-#31.
- Provenance and safe-reader work #47-#48 feeds report publication (#30), remote resilience (#45), service readiness (#49), and security gates (#50).
- Service/API issues #32-#36 and UI issues #37-#40 converge in remote deployment #44; solve admission also depends on the remote solve adapter #42 and security controls #49.
- Protected acceptance #54 depends on CPU CI (#52), hostile-input/security gates (#50), and remote recovery evidence (#45).
- The evidence-bound README #55 depends on visuals (#40), canonical validation (#31), resilience (#45), and performance budgets (#46).
- Release publication #57 is blocked by the complete dataset (#19), trained models (#27), validation (#31), API tests (#36), deployment (#44), cylinder evidence (#14), documentation (#55), and release automation (#56).

<a id="maintenance"></a>
## Maintenance

GitHub is authoritative for live state. Update this table in the pull request that creates, closes, supersedes, or re-scopes an implementation issue. A new implementable specification requirement must add a shippable issue, its RFC label, applicable tracking checklist entry, and dependency links in the same change.
