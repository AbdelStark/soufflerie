# Deterministic validation reporting

The RFC-0008 reporter renders one already-evaluated `ValidationReport` into a
canonical JSON document, red-first Markdown, eight diagnostic SVG plots, and a
self-verifying plot manifest. Renderers copy metric, gate, and overall status
values from the report; they never reinterpret thresholds or turn a red report
green.

## Report contract

`ValidationReport` remains the single source of truth. Its content identity
covers the generator version, metrics and bootstrap summaries, all twelve gates,
OOD and sensitivity evidence, bounded plot data, and provenance. The plot-data
contract records:

- the representative test case closest to median velocity error;
- the worst test case by velocity error;
- bounded solver and surrogate velocity-magnitude grids for both cases;
- canonically ordered per-case design, drag, divergence, and compliance values;
- the selected model and the mean-field and nearest-design baseline aggregates.

Fields are limited to `128 x 128` scalar grids and per-case collections are
bounded at 10,000 entries. The current generator identifier is
`validation-report-v1`. Durable schemas are checked in as
[`validation-report.json`](../../schemas/v1/validation-report.json),
[`validation-plot-data.json`](../../schemas/v1/validation-plot-data.json), and
[`plot-manifest.json`](../../schemas/v1/plot-manifest.json).

## Lineage required for publication

Publication accepts exactly these direct parent roles:

```text
dataset
solver
ensemble_model_0
ensemble_model_1
ensemble_model_2
baseline_0
baseline_1
```

Every dataset, model, and baseline display ID must prefix its corresponding
full SHA-256 digest. OOD model digests must match the three ensemble parents,
and the sensitivity digest must match the selected model parent.
`validate_report_publication` then compares the report with independently
reviewed source revision, lock digest, configuration digest, package allowlist,
and complete parent map. Dirty, nondeterministic, missing, extra, or mismatched
evidence blocks publication.

## Generated artifacts

For an input such as `reports/validation.json`, the renderer produces:

```text
reports/validation.json
reports/validation.md
reports/validation.plots.json
reports/validation.plots/
  representative-fields.svg
  worst-fields.svg
  baseline-comparison.svg
  error-by-design.svg
  head-vs-field.svg
  divergence-compliance.svg
  ood-variance.svg
  sensitivity.svg
```

SVG is generated from fixed geometry, colors, fonts, ordering, and number
formatting without platform-native plotting state. This keeps artifacts
byte-identical across supported CI platforms. Every plot includes the report
ID and a textual green or red status. The manifest binds the exact file set,
titles, byte sizes, SHA-256 digests, report identity, and its own complete
contents.

Writers replace each file atomically and publish the plot manifest last as the
commit marker. Existing unrecognized files or symbolic-link targets fail
closed. Readers bound the JSON and rendered artifact sizes, validate the strict
schema and report identity, require canonical sorted JSON, and reject missing,
extra, stale, or modified output files.

## Regeneration

The checked fixture is deliberately synthetic and red. It tests presentation
and tamper behavior; it is not canonical model evidence. Verify it without
changing the worktree:

```bash
uv run pytest tests/validation/test_report.py
uv run python scripts/render_validation.py --check tests/fixtures/report.json
```

After reviewing an input report, regenerate its sibling artifacts with:

```bash
uv run python scripts/render_validation.py reports/validation.json
```

The canonical dataset/model evaluation in issue #31 will create
`reports/validation.json` and replace the explicit unevaluated notice in
[`reports/validation.md`](../../reports/validation.md). Until then, no fixture
metric or plot may be presented as release evidence.
