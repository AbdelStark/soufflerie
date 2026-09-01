# Glossary

<a id="canonical-terms"></a>
## Canonical terms

- **Artifact:** Immutable bytes plus typed metadata, digest, schema version, and provenance.
- **Bundle:** A deployable surrogate consisting of safe weights, architecture config, preprocessing statistics, and metadata.
- **Case:** One fully specified geometry, Reynolds number, grid, step schedule, and seed.
- **Case ID:** Content-derived identity of a canonical `CaseConfig`.
- **Curated run:** Solver output admitted to a dataset after all run-level integrity and numerical checks.
- **Design point:** The physical tuple `(aspect_ratio, rotation_deg, scale, reynolds)` independent of snapshots or execution attempts.
- **Field Cd:** Drag coefficient estimated from predicted mean fields using the declared control-volume balance.
- **Gate:** A deterministic predicate over a metric and threshold. A required red gate makes overall validation red.
- **Head Cd:** Drag coefficient emitted by the learned scalar head.
- **Lattice unit (lu):** Dimensionless solver unit with lattice spacing and timestep equal to one.
- **LUPS:** Lattice updates per second, measured over synchronized solver kernel time.
- **Manifest:** The sole versioned index of curated run artifacts and immutable split membership.
- **Model ID:** Content-derived identity of a model bundle.
- **OOD probe:** A deliberately unsupported input used to test uncertainty response, never a supported prediction claim.
- **Prediction:** Surrogate fields, two drag estimates, consistency flags, validation state, latency, and provenance for one supported case.
- **Reference solve:** A numerical solver run for the same user-selected case; “reference” does not mean engineering ground truth.
- **Remote plane:** Authenticated runtime that owns CUDA execution and persistent large artifacts.
- **Report ID:** Content-derived identity of one validation report.
- **Required gate:** A gate whose failure sets the overall report state to red.
- **Solver result:** Time-averaged fields, force histories, derived coefficients, diagnostics, and provenance from one case.
- **Split:** Design-point-disjoint train, validation, or test partition.
- **Validation state:** `green` only when every required gate passes; otherwise `red`.

<a id="symbols"></a>
## Symbols

| Symbol | Meaning |
|---|---|
| `f_i` | D2Q9 population in lattice direction `i` |
| `c_i` | D2Q9 discrete lattice velocity |
| `w_i` | D2Q9 equilibrium weight |
| `rho` | Lattice density |
| `u, v` | Streamwise and transverse lattice velocity |
| `nu` | Lattice kinematic viscosity |
| `tau` | BGK relaxation time, `3*nu + 0.5` |
| `omega` | BGK relaxation rate, `1/tau` |
| `Re` | Reynolds number, `U_ref * D_lu / nu` |
| `Cd, Cl` | Drag and lift coefficient per unit span |
| `St` | Strouhal number, `f_lu * D_lu / U_ref` |
| `SDF` | Signed distance function; negative inside the obstacle |
| `D_lu` | Unscaled reference major-axis diameter in lattice cells |

<a id="reserved-language"></a>
## Reserved language

Use **solver** rather than “ground truth” except when explaining why it is not ground truth. Use **validated against the declared gates** rather than “accurate” or “trusted.” Use **remote GPU** rather than implying local CUDA. Use **supported domain** only for `Re in [40, 300]` and declared ellipse ranges. Do not describe OOD probes as supported extrapolation.
