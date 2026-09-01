# RFC-0004: Experiment configuration and design of experiments

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-09-01
- Target milestone: v0.1

## Summary

Versioned strict YAML defines sweep and run policy, while deterministic maximin Latin hypercube sampling generates exactly 1,000 valid ellipse/Reynolds design points. A fixed seed and design-point hash assign immutable `600/200/200` train/validation/test splits before any solver execution.

## Motivation

The surrogate evidence is invalid if snapshots or reruns leak across splits, if failed cases are silently replaced non-deterministically, or if configuration coercion changes the sampled domain. The PRD fixes the parameters, method, count, and split sizes; [`03-data-model.md#data-invariants`](../spec/03-data-model.md#data-invariants) requires immutable identities.

## Goals

- Define strict, reviewable YAML schemas for cases, sweeps, and experiment references.
- Generate space-filling samples reproducibly without global RNG state.
- Validate numerical/geometry feasibility before remote submission.
- Freeze split membership by design point with exact counts.
- Make config and design identities independent of YAML formatting.

## Non-Goals

- Adaptive or active-learning sampling.
- Multiple shape families in v0.1.
- Random snapshot-level splitting.
- Automated reduction of the sample count after failures.

## Proposed Design

`configs/sweeps/mvp-v1.yaml` parses into:

```python
class Range(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    minimum: float
    maximum: float

class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1] = 1
    name: str
    seed: int
    samples: Literal[1000]
    shape_family: Literal["ellipse"]
    aspect_ratio: Range
    rotation_deg: Range
    scale: Range
    reynolds: Range
    grid: GridSpec
    run: RunSchedule
    split_counts: tuple[Literal[600], Literal[200], Literal[200]]
```

Canonical values are exactly: aspect ratio `[0.3,1.0]`, rotation degrees `[0,30]`, scale `[0.75,1.25]`, Reynolds `[40,300]`, sample count `1000`, split counts `(600,200,200)`, and a checked-in 64-bit seed. YAML uses explicit numbers; environment interpolation, anchors, aliases, unknown keys, implicit strings-to-numbers, and non-finite values are rejected. YAML sequences normalize only to the immutable tuple fields declared by the model. `config_digest` derives from validated canonical JSON.

Sampling uses a local NumPy `Generator(PCG64(seed))`. For dimension `d=4` and `n=1000`, each axis receives one jittered sample per stratum `(k+u)/n`, permutations are independent per dimension, and 32 deterministic candidates are generated from child seeds. The candidate maximizing minimum pairwise Euclidean distance in normalized `[0,1]^4` space is selected; ties use candidate index. Endpoints are not forced. Values map linearly to physical ranges and are serialized at IEEE-754 double precision before `CaseConfig` derivation.

```python
@dataclass(frozen=True, slots=True)
class DesignPoint:
    index: int
    aspect_ratio: float
    rotation_deg: float
    scale: float
    reynolds: float
    design_id: str
    split: Split

def sample_design(config: SweepConfig) -> tuple[DesignPoint, ...]: ...
def assign_splits(points: Sequence[UnsplitDesignPoint], seed: int) -> tuple[DesignPoint, ...]: ...
```

Every sample passes RFC-0002 lattice and RFC-0003 geometry preflight. Because the fixed public domain is designed to be feasible, a preflight failure fails configuration generation; it does not resample around an invalid region. This prevents an undocumented change in sampling density.

Split assignment computes `sha256("split-v1" || seed_bytes || canonical_design_json)` for each point, sorts ascending by full digest then `design_id`, assigns the first 600 train, next 200 validation, and final 200 test, and persists the assignment before solver execution. Execution order may differ but membership cannot. Reruns and snapshots inherit the design point's split.

`design_id` hashes physical parameters and design schema, excluding grid/run controls. `case_id` also includes numerical controls. A numerical rerun of one design point therefore has a new case ID but may only replace the canonical run through an explicit dataset revision while retaining split.

Design summaries report per-dimension min/max/mean/quantiles, pairwise correlation, nearest-neighbor distances, split counts, and preflight outcome. Plots are generated from the manifest, not hand-edited notebook state. A notebook MAY explore statistics but a script owns reproducible output.

## Alternatives Considered

### Cartesian grid

It gives transparent coverage but grows combinatorially and creates axis-aligned repetition at 1,000 points. Latin hypercube better covers four continuous variables under the fixed budget.

### Pure random sampling

It is simple but can leave large gaps and uneven one-dimensional marginals. Stratified LHS has deterministic marginal coverage; maximin candidate selection improves spacing without changing scope.

### Split after solver completion

It could balance only successful cases but makes execution failures influence evaluation membership and invites leakage/replacement bias. Precomputed immutable splits are required.

### Hash threshold split

It is stable but does not guarantee exact `600/200/200` counts. Sort-by-hash gives both stability and exact counts.

## Drawbacks

- Candidate maximin selection is more expensive than one LHS draw, though trivial at 1,000 points.
- Linear scaling may under-sample regimes with nonlinear physical transitions.
- Exact split counts couple membership to the full design set; adding points creates a new dataset design version.
- Rejecting infeasible domains instead of resampling can block the sweep early.

## Migration / Rollout

1. Land strict config models and canonicalization with checked-in examples.
2. Implement LHS candidates, deterministic selection, identities, and split assignment.
3. Add numerical/geometry preflight and design summary generation.
4. Freeze `mvp-v1.yaml`, its digest, sample digest, and split digest before remote execution.
5. Any domain/count/seed change creates a new named config and dataset identity; existing manifests remain readable.

## Testing Strategy

- Reject unknown keys, strings for numbers, booleans, NaN/infinity, reversed ranges, wrong sample/split counts, and YAML aliases that violate policy.
- Golden-test canonical JSON and config digest independent of key order/formatting.
- Reproduce all 1,000 points and split identities bitwise under the lock.
- Assert one point per stratum per dimension and all values within ranges.
- Verify candidate tie-breaking and absence of global RNG mutation.
- Assert exact split counts and no design/case overlap across splits.
- Run preflight over the full generated design without remote calls.
- Snapshot deterministic summary statistics and update only with design-version review.

## Open Questions

None for v0.1. Alternative designs for future data scaling require a new RFC owned by the ML maintainer.

## References

- [`prd.md#62-datagen-sweep-runner`](../../prd.md#62-datagen-sweep-runner)
- [`03-data-model.md#identities`](../spec/03-data-model.md#identities)
- [RFC-0002](RFC-0002-d2q9-lbm-core.md)
- [RFC-0003](RFC-0003-geometry-boundaries-and-forces.md)
- McKay, Beckman, and Conover, “A Comparison of Three Methods for Selecting Values of Input Variables,” 1979.
