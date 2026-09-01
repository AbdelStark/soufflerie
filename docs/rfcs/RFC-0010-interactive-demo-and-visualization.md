# RFC-0010: Interactive demo and visualization

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

The v0.1 client is a Gradio interface mounted with the service and reused by a local CPU command. It exposes four physical controls, predicts on committed control changes, renders fields through one deterministic plotting contract, and keeps validation and per-case failures visible in prediction and reference-solve comparison states.

## Motivation

The PRD calls for an immediate design loop, side-by-side reference solves, consistent visuals, and three honest repository GIFs. UI convenience must not alter fields, collapse validation into a favorable image, or obscure unsupported/failed states. The same rendering logic must generate API PNGs, reports, and documentation visuals.

## Goals

- Provide an understandable prediction loop on desktop and narrow screens.
- Keep validation status and consistency flags prominent and accessible.
- Compare solver and surrogate fields with common scales and explicit errors.
- Make plots and GIFs deterministic and reproducible from artifacts.
- Support remote GPU and local CPU/bundled-model modes through the same UI contract.

## Non-Goals

- A custom JavaScript application, arbitrary geometry editor, user sessions, or saved projects.
- Hiding warmup, queue, or solver latency behind fabricated progress.
- Visual claims beyond the measured two-dimensional fields.
- Client-side model inference.

## Proposed Design

The primary controls are shape aspect ratio, rotation degrees, scale, and Reynolds number with limits matching `PredictionRequest`. Defaults are `(0.65, 0, 1.0, 100)`. Although the PRD says “three sliders,” the ellipse domain has four independent values; v0.1 exposes all four rather than hiding scale. Slider changes update local labels continuously and trigger one prediction on release/commit. A 150 ms debounce collapses near-simultaneous committed changes; an in-flight older result is discarded by monotonically increasing client request sequence, not cancelled at the service.

The layout order is:

1. validation banner and model/report identity link;
2. four controls, Predict/retry state, and measured latency;
3. field panels and Cd estimates;
4. consistency panel with text/icon-independent states;
5. `Solve for real` action and job progress;
6. comparison fields, error fields, solver coefficients, and limitations.

The validation banner reads either `Validated against all v0.1 gates` or `Surrogate unvalidated: one or more required gates failed`. Red state uses text, border/icon shape, and color; it cannot be dismissed, collapsed, or placed below results. Each red consistency flag names measured value and threshold. “Solver reference” is used instead of “ground truth.”

```python
@dataclass(frozen=True, slots=True)
class RenderSpec:
    schema_version: Literal[1] = 1
    width_px: int = 1200
    height_px: int = 400
    dpi: int = 100
    colormap_velocity: Literal["viridis"] = "viridis"
    colormap_pressure: Literal["coolwarm"] = "coolwarm"
    colormap_vorticity: Literal["RdBu_r"] = "RdBu_r"
    colormap_error: Literal["magma"] = "magma"

def render_fields(fields: FlowFields, spec: RenderSpec) -> PngArtifact: ...
def render_comparison(
    prediction: FlowFields,
    reference: FlowFields,
    spec: RenderSpec,
) -> PngArtifact: ...
```

Velocity magnitude uses `[0, p99.5]` from the reference field for comparisons and its own finite p99.5 for standalone predictions; pressure proxy is `(rho-1)/3` with symmetric limits around zero; vorticity is second-order `dv/dx-du/dy` with symmetric p99 absolute limits. The obstacle is overlaid as a neutral opaque mask but underlying raw values remain in NPZ and metrics. Axes have equal physical aspect, flow-direction arrow, colorbar with lattice units, parameter annotation, model/report or solver provenance, and no interpolation that invents resolution.

Comparison plots use identical per-variable scales for solver and surrogate. Error is signed for scalar fields and vector magnitude for velocity, with its own labeled scale. Cd head, Cd field, solver Cd, percentage errors, divergence, and obstacle compliance appear as text. A failed solve preserves the prediction and shows sanitized failure/retry guidance.

The solve action shows true states `queued`, `running`, and terminal events. Progress comes only from server events; indeterminate execution uses an indeterminate indicator. Reconnection uses stored job ID and sequence within the browser process. The button disables while one job for the current session is active, without claiming cancellation.

`soufflerie demo` loads the bundled CPU bundle/report and mounts the same component tree against an in-process predictor. Reference solve is available locally only when the CPU solver profile and declared case budget are present; otherwise the button is visibly disabled with explanation. Remote deployment mounts Gradio into the FastAPI ASGI app without widening HTTP schemas.

README assets are generated by `scripts/render_readme_assets.py` from committed artifact references and a scene manifest. Required assets are solver vortex shedding, solver/surrogate/error comparison, and slider interaction. Each GIF includes parameters/provenance in adjacent README text, avoids implying real-time solver speed, uses bounded size, and has an accessible static fallback/alt text.

## Alternatives Considered

### Custom React client

It offers more interaction control but adds a build toolchain and duplicated schema client for a one-page educational interface. Gradio meets v0.1 scope and can be replaced behind the service later.

### Predict on every slider input event

It feels continuous but can flood a remote GPU and display responses out of order. Release/commit plus debounce provides a responsive bounded loop.

### Independent plot scales in comparison

They make each panel visually rich but can hide amplitude error. Shared reference-derived scales are required for honest comparison.

### Zeroing obstacle values for display and response

Mask overlay improves readability, but mutating raw predictions would hide compliance failures. Only rendering overlays the obstacle.

## Drawbacks

- Four controls depart from the PRD’s shorthand “three sliders,” though they expose every required parameter.
- Gradio constrains fine-grained accessibility and responsive behavior.
- Base64 plot responses duplicate server rendering work.
- Static GIFs require regeneration when render contracts change.

## Migration / Rollout

1. Implement pure deterministic rendering and golden/image-statistic tests.
2. Build UI against fixture responses for green/red/error/job states.
3. Connect local in-process predictor and installed CLI smoke.
4. Mount with the service, integrate SSE solve progress, and run browser tests.
5. Generate README assets from accepted artifacts after validation/deployment evidence exists.

Rendering schema changes regenerate plots/assets and receive a changelog entry. A future UI may replace Gradio while retaining HTTP and visibility contracts.

## Testing Strategy

- Golden-test render metadata, dimensions, labels, colormap/range calculations, obstacle overlay, and byte digest under the locked plotting stack.
- Test constant/zero/non-finite field handling and comparison shared scales.
- Browser-test all slider bounds, debounce, stale-response rejection, loading/error/retry, narrow/wide layouts, and keyboard labels.
- Assert red validation is first, persistent, textually explicit, and not color-only.
- Simulate every solve event, reconnect, failure, expiry, and unavailable-local-solver state.
- Run local installed-wheel demo smoke and remote mounted-app smoke.
- Regenerate README assets in CI and require a clean diff for their manifest/metadata.

## Open Questions

None for v0.1. A custom client or additional geometry controls requires a new RFC owned by the service/UI maintainer.

## References

- [`prd.md#65-api-ui`](../../prd.md#65-api-ui)
- [`05-observability.md#provenance`](../spec/05-observability.md#provenance)
- [RFC-0008](RFC-0008-validation-and-release-gates.md)
- [RFC-0009](RFC-0009-inference-and-solve-api.md)
