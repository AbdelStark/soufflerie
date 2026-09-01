# Data model

<a id="canonical-types"></a>
## Canonical types

Shared scalar records are frozen dataclasses or strict Pydantic v2 models. Array-bearing records use dataclasses with runtime shape/dtype validation.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import numpy.typing as npt

SchemaVersion = Literal[1]
Split = Literal["train", "validation", "test"]
RunState = Literal["pending", "running", "succeeded", "failed"]

@dataclass(frozen=True, slots=True)
class ShapeParams:
    aspect_ratio: float       # minor / major axis, [0.3, 1.0]
    rotation_deg: float       # counter-clockwise, [0, 30]
    scale: float              # reference-diameter multiplier, [0.75, 1.25]

@dataclass(frozen=True, slots=True)
class CaseConfig:
    schema_version: SchemaVersion
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

The full manifest, checkpoint, validation, and HTTP models are fixed by RFC-0005, RFC-0006, RFC-0008, and RFC-0009. JSON schemas generated from Pydantic models are checked in under `schemas/v1/` and compared in CI.

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

`FlowFields` arrays MUST be C-contiguous and have identical `(ny, nx)` shapes. Solver arrays are fp32; dataset field arrays are fp16 after time averaging and deterministic area downsampling to `(128, 256)`. The mask is boolean. Values MUST be finite. `rho > 0`; obstacle SDF values are negative, the zero contour is the boundary, and fluid values are positive.

Inference accepts and returns batch-first tensors:

```text
input:  float32[B, 2, 128, 256]  # clipped normalized SDF, normalized Re plane
output: float32[B, 3, 128, 256]  # normalized mean u, v, rho-1
latent: float32[B, 64, 128, 256]
cd:     float32[B, 1]
```

Device transfers and dtype casts occur at adapter boundaries. Model code MUST NOT silently cast fp64 input, move devices, or change memory layout.

<a id="identities"></a>
## Stable identities

Canonical JSON uses UTF-8, sorted keys, no insignificant whitespace, decimal numbers normalized through the typed model, and no NaN/infinity. IDs are lowercase SHA-256 prefixes:

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
