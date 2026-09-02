# Data model

<a id="canonical-types"></a>
## Canonical types

Shared scalar records are frozen dataclasses or strict Pydantic v2 models. Array-bearing records use dataclasses with runtime shape/dtype validation.

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

SchemaVersion = Literal[1]
Split = Literal["train", "validation", "test"]
RunState = Literal["pending", "running", "succeeded", "failed"]

class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ShapeParams(StrictFrozenModel):
    aspect_ratio: float = Field(ge=0.5, le=1.0)
    rotation_deg: float = Field(ge=0.0, le=30.0)
    scale: float = Field(ge=0.75, le=1.25)


class GridSpec(StrictFrozenModel):
    schema_version: SchemaVersion = 1
    nx: int = Field(ge=3)
    ny: int = Field(ge=3)
    axis_order: Literal["yx"] = "yx"


class CaseConfig(StrictFrozenModel):
    schema_version: SchemaVersion = 1
    shape: ShapeParams
    reynolds: float
    nx: int
    ny: int
    steps: int
    warmup_steps: int
    inlet_velocity_lu: float
    seed: int

@dataclass(frozen=True, slots=True)
class FlowFields:
    u: npt.NDArray[np.float32]
    v: npt.NDArray[np.float32]
    rho: npt.NDArray[np.float32]
    sdf: npt.NDArray[np.float32]
    obstacle_mask: npt.NDArray[np.bool_]

@dataclass(frozen=True, slots=True)
class SolverResult:
    case_id: str
    fields: FlowFields
    cd: float
    cl_mean: float
    strouhal: float | None
    force_steps: npt.NDArray[np.int64]
    cd_history: npt.NDArray[np.float32]
    cl_history: npt.NDArray[np.float32]
    diagnostics: SolverDiagnostics
    provenance: Provenance
```

The shared implementation additionally defines:

- `GridSpec`: `nx`, `ny`, fixed `yx` axis order, and explicit lattice spacing/time units;
- `ArrayDescriptor`: dtype, shape, unit, C order, finite-only, and
  `allow_pickle=false` declarations;
- `SolverDiagnostics`: progress/sample counts, initial/final mass and derived
  drift, density/speed bounds, convergence/validity, and typed messages;
- `Provenance`: exact source/lock/config identities, environment and dtype
  policy, direct parent digests, seeds, determinism, aware timestamps, and GPU
  seconds;
- `ArtifactRef`: artifact type, full digest, matching 20-character display ID,
  byte size, and normalized artifact-root-relative URI.

These scalar records are strict, frozen Pydantic models: unknown fields and
implicit string/boolean coercions fail. Array-bearing `FlowFields` and
`SolverResult` remain frozen dataclasses whose construction performs runtime
array and coherence checks. The full manifest, checkpoint, validation, and HTTP
models are fixed by RFC-0005, RFC-0006, RFC-0008, and RFC-0009.

JSON Schema draft 2020-12 documents generated from the shared scalar records
are checked in under `schemas/v1/`. `scripts/export_schemas.py --check` compares
the generated contract byte-for-byte with those files. Unknown integer schema
versions raise `SchemaVersionError`; no schema migration beyond version 1 is
implemented.

<a id="units-and-coordinates"></a>
## Units and coordinates

- Array order is `(ny, nx)`; `x` increases with columns and flow direction, `y` increases with rows.
- Solver values use lattice units. One timestep and one lattice spacing equal one.
- `u` and `v` are lattice velocities; `rho` is density normalized around `1.0`.
- The reference diameter `D_lu` is the unscaled ellipse major-axis diameter in lattice cells; `scale` multiplies both axes.
- `rotation_deg` is counter-clockwise in degrees at public/config boundaries and converted once to radians internally.
- Drag and lift coefficients use `0.5 * rho_ref * U_ref^2 * D_lu` per unit span.
- Strouhal is `f_lu * D_lu / U_ref`.

<a id="array-contracts"></a>
## Array contracts

`FlowFields` arrays MUST be C-contiguous and have identical `(ny, nx)` shapes. Solver arrays are fp32; dataset field arrays are fp16 after time averaging and deterministic area downsampling to `(320, 256)`. The mask is boolean. Values MUST be finite. `rho > 0`; obstacle SDF values are negative, the zero contour is the boundary, and fluid values are positive.

Inference accepts and returns batch-first tensors:

```text
input:  float32[B, 2, 320, 256]  # clipped normalized SDF, normalized Re plane
output: float32[B, 3, 320, 256]  # normalized mean u, v, rho-1
latent: float32[B, 64, 320, 256]
cd:     float32[B, 1]
```

Device transfers and dtype casts occur at adapter boundaries. Model code MUST NOT silently cast fp64 input, move devices, or change memory layout.

<a id="identities"></a>
## Stable identities

Canonical JSON uses UTF-8, sorted keys, no insignificant whitespace, decimal
numbers normalized through the typed model (including `-0.0` to `0.0`), UTC
microsecond timestamps, and no NaN/infinity. Array bytes and array descriptors
are hashed separately rather than silently converting arrays to lists. IDs are
lowercase SHA-256 prefixes:

- `case_id = sha256(canonical CaseConfig)[:20]`
- `dataset_id = sha256(canonical manifest rows + schema metadata)[:20]`
- `model_id = sha256(bundle metadata + weights digest)[:20]`
- `report_id = sha256(report JSON excluding report_id)[:20]`

Prefixes are presentation identifiers; full digests remain in metadata and collision checks compare full values.

<a id="schema-evolution"></a>
## Schema evolution

All durable artifacts start at integer `schema_version: 1`. Readers reject unknown major integers with `SchemaVersionError`. Adding optional fields with defined defaults is backward-compatible within schema 1. Removing fields, changing units, dtype, shape, identity canonicalization, split semantics, or metric definitions requires schema 2 plus an explicit migration tool or a documented no-migration decision in a new RFC.

<a id="data-invariants"></a>
## Named invariants

- `DM-1 SHAPE`: every field array shares the declared shape.
- `DM-2 FINITE`: persisted numeric data contains no NaN or infinity.
- `DM-3 DTYPE`: solver state is fp32; curated fields are fp16; metrics accumulate in fp64.
- `DM-4 IDENTITY`: artifact bytes and canonical metadata verify against full SHA-256 digests.
- `DM-5 NO_PICKLE`: no public or remote artifact requires executable Python deserialization.
- `DM-6 SPLIT`: a `case_id` occurs in exactly one split and split membership never changes for a `dataset_id`.
- `DM-7 PROVENANCE`: every child artifact names all direct parent digests.
- `DM-8 UNITS`: every scalar/array field uses the units declared in this document; conversions occur only at named boundaries.

<a id="executable-invariants"></a>
## Executable invariant map

| Invariant | Authoritative executor |
|---|---|
| DM-1 SHAPE | `validate_array`, `FlowFields`, `SolverResult` |
| DM-2 FINITE | strict scalar fields plus `validate_array` and result constructors |
| DM-3 DTYPE | exact `ArrayDescriptor`/`validate_array` dtype comparison; no implicit cast |
| DM-4 IDENTITY | `canonical_sha256`, `verify_sha256`, and `ArtifactRef` prefix/full-digest coherence |
| DM-5 NO_PICKLE | `ArrayDescriptor.allow_pickle=false` and object-dtype rejection |
| DM-6 SPLIT | `validate_split_membership`, `build_manifest`, and `load_manifest` exact cardinality/membership checks |
| DM-7 PROVENANCE | strict `Provenance.parent_sha256` plus `validate_parent_digests` |
| DM-8 UNITS | fixed `GridSpec` units, `ArrayDescriptor.unit`, and `validate_field_units` |
