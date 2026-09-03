# Interactive prediction demo

The Soufflerie component tree is a reusable Gradio 6 interface over the same
version-one request and response contracts as `POST /predict`. It neither runs a
model in the browser nor invents a second prediction schema. A local or deployed
runtime supplies a `DemoPredictor` and the immutable model, dataset, and
validation-report identity shown to the user.

Install the locked UI and rendering profiles with:

```bash
uv sync --frozen --extra serve --extra viz
```

An adapter builds, mounts, or launches the component tree without changing it:

```python
from soufflerie.demo import DemoIdentity, build_demo

identity = DemoIdentity(
    model_id="7b6fd39c0ec78f452163",
    dataset_id="4aefbbe88a18d233249b",
    report_id="4995ef8f8456030f467d",
    validation_status="red",
    report_href="/validation/report",
)
demo = build_demo(predictor, identity)
```

`build_demo` constructs and queues the app but does not bind a socket, open a
browser, contact a remote service, or load Gradio during `import soufflerie.demo`.
Runtime adapters own launch and ASGI mounting. The predictor performs exactly
one attempt and returns a validated `PredictionResponse`; retry remains an
explicit user action.

## Controls and submission

The four independent public design values are visible:

| Control | Default | Bounds | Commit step |
| --- | ---: | ---: | ---: |
| Ellipse aspect ratio | `0.65` | `[0.50, 1.00]` | `0.01` |
| Rotation | `0 degrees` | `[0, 30] degrees` | `0.5 degrees` |
| Ellipse scale | `1.00` | `[0.75, 1.25]` | `0.01` |
| Reynolds number | `100` | `[40, 300]` | `1` |

Moving a slider updates its displayed value in the browser. Prediction is bound
only to the slider’s release/commit event, never its continuous input event.
The Predict current design button submits the same four values and becomes the
explicit Retry prediction action after a failure.

Each browser session receives monotonically increasing request numbers. A
150 ms server-side coalescing window discards superseded commits before they
reach the predictor. If a newer commit arrives while an older prediction is
already running, the older attempt may finish but its response is marked stale
and returns no component updates. The UI therefore implements last-commit-wins
without claiming that model or service execution was cancelled.

Session sequencing is held in an ordered, lock-protected registry capped at
1,024 browser sessions with a one-hour idle lifetime. The Gradio queue accepts
at most eight waiting events and permits eight prediction handlers; downstream
prediction admission remains owned by the runtime adapter.

## Persistent validation and results

The validation banner is the first visible state, before the heading, controls,
or results. Its wording is fixed:

- `Validated against all v0.1 gates` for a green report;
- `Surrogate unvalidated: one or more required gates failed` for a red report.

A red report remains usable but is labeled `VALIDATION BLOCKED` with text, a
border, and an exclamation mark. A green report is labeled `PASS` with text, a
border, and a check mark. Both states show the loaded model ID and a link to the
validation report. A response whose model, dataset, report, or validation status
does not match that loaded identity is rejected as an unavailable prediction.

Successful output includes the accessible three-field PNG, model-head and field
Cd estimates, inference and total request timing, artifact identities, and all
three per-case consistency checks. Every consistency row includes:

- the words `PASS` or `FAIL` and a distinct symbol;
- the measured value;
- the fixed threshold and units.

The head/field Cd gap passes at most 10 percent, divergence passes below 3
times the solver baseline, and obstacle velocity passes below a `0.01` ratio.
Color reinforces these states but never carries them alone. The output image has
alt text naming all three fields and the four committed parameter values.

Loading is announced through a live status region with the request number. A
failure is announced as an alert with sanitized retry guidance; exception text
is never rendered, and the previous successful fields, metrics, and consistency
cards remain visible.

## Responsive and keyboard contract

The four controls use native labeled Gradio sliders, preserving arrow-key and
focus behavior. A visible focus outline applies across the app. At viewports up
to 760 pixels wide, the four-control and result grids become one column; wider
viewports keep a four-control row and field/sidebar result layout. Validation
remains first in both layouts.

The checked browser configuration asserts component order, labels, bounds,
release-only bindings, private event APIs, focus styling, and both responsive
layouts. The framework-independent concurrency tests cover coalesced, in-flight,
failed, expired, and identity-mismatched responses:

```bash
uv run pytest tests/demo/test_ui.py -m 'not remote'
```
