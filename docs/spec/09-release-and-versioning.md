# Release and versioning

<a id="version-policy"></a>
## Version policy

The distribution, CLI, and public Python API use semantic versioning. Before `1.0`, minor releases may change experimental APIs only with migration notes; patch releases remain backward-compatible. Durable artifact and HTTP schemas use independent integer versions as defined in [`03-data-model.md`](03-data-model.md#schema-evolution). Model and dataset identities are content-derived, not semantic versions.

The first release is `v0.1.0`; the GitHub milestone is named `v0.1`. A signed annotated tag points to the verified release commit.

<a id="compatibility-policy"></a>
## Compatibility policy

- Public Python exports, CLI commands, config keys, and HTTP response fields require changelog entries for user-visible changes.
- Removing or changing a public surface requires a deprecation warning for at least one minor release and 90 days, unless a security issue requires immediate removal.
- New optional fields are compatible; readers ignore only fields explicitly allowed by schema rules. Config and request models reject unknown fields to catch mistakes.
- Artifact unit/dtype/shape changes are schema-breaking even if filenames stay the same.
- Model checkpoints are supported only with their bundled code/config schema and declared package version range.

<a id="dependency-policy"></a>
## Dependency policy

Python is constrained to `>=3.11,<3.12` for v0.1 to match the declared environment and eliminate cross-minor numerical drift. Direct dependency constraints are recorded in `pyproject.toml`; exact transitive versions are recorded in `uv.lock`. The initial compatibility set is:

| Profile | Constraint | Rationale |
|---|---|---|
| Solver | `warp-lang==1.17.0` | Kernel/runtime API and CPU fallback |
| ML remote | `nvidia-physicsnemo[cu12]==2.2.1`, `torch==2.10.0` | FNO/checkpoint and CUDA 12 runtime contract; later PyTorch releases require CUDA 13 and conflict with the `cu12` profile |
| Remote operations | `modal==1.5.5` | App/image/volume API contract |
| Service | `fastapi==0.141.1`, `gradio==6.26.0` | OpenAPI and mounted UI contract |
| Schemas | `pydantic==2.13.5`, `pydantic-settings==2.15.0` | Strict models and settings |
| Data | `numpy==2.2.6`, `scipy==1.17.1`, `pandas==2.3.3`, `pyarrow==23.0.1`, `PyYAML==6.0.3` | Arrays, FFT, tabulation, Parquet, strict config input; NumPy, pandas, and PyArrow remain below the RAPIDS ceilings required by the ML profile |
| CLI/viz | `typer==0.27.2`, `matplotlib==3.11.1`, `imageio==2.37.4` | Installed CLI and deterministic rendering |
| Artifact/service support | `safetensors==0.8.0`, `httpx==0.28.1`, `pillow==12.3.0`, `tensorboard==2.21.0` | Safe weights, HTTP tests, image encoding, training event compatibility |
| Development | `pytest==9.1.1`, `pytest-cov==7.1.0`, `hypothesis==6.167.1`, `ruff==0.16.5`, `mypy==2.3.1`, `pre-commit==4.6.2` | Test, coverage, properties, formatting/lint, typing, local hooks |
| Release tooling | `uv==0.12.8`, `build==1.6.0`, `cyclonedx-bom==7.3.1`, `pip-audit==2.10.1` | Lock/install/build, standards build, SBOM, vulnerability audit |

Compatibility is verified by the lock/build jobs; if the resolver proves an exact set incompatible on Python 3.11, the first implementation issue updates this table and spec in the same reviewed change rather than forcing installation. GPU, ML, remote, service, and visualization packages live in named extras or dependency groups; importing the core package does not load them. The remote image installs the locked full profile. Automated updates are reviewed with numerical, artifact, and API regression evidence.

<a id="release-artifacts"></a>
## Release artifacts

Each release publishes source distribution, wheel, checksums, changelog, SBOM, build provenance, validation evidence index, and the bundled small checkpoint or an immutable release-asset reference. The wheel includes `py.typed`, schema JSON, default/smoke configs, and bundled-model metadata. The source distribution additionally includes tests, docs, examples, scripts, license, notice, citation, contribution, conduct, security, and governance files.

Large datasets and training checkpoints do not ship inside the wheel. Their manifests, digests, license metadata, and retrieval instructions do.

<a id="release-gate"></a>
## Release gate

1. Freeze a clean source revision and dependency lock.
2. Pass every CPU CI gate and installed-wheel/package-content test.
3. Run remote solver, dataset, training, validation, performance, and deployment acceptance on that revision.
4. Check in the generated validation report, plots, manifest statistics, and provenance index without hand-editing generated values.
5. Confirm README claims exactly match evidence and every red gate is prominent.
6. Build artifacts from the tagged commit, scan dependencies/secrets, generate SBOM/provenance, and verify checksums in a clean environment.
7. Publish `v0.1.0`, then execute fresh-clone quickstarts and the live endpoint smoke.

No release may be marked green from CPU smoke tests alone.

<a id="repository-governance"></a>
## Repository governance

`main` is releasable. Pull requests require linked specification/issue, tests, changelog when user-visible, and review of generated artifacts. `CODEOWNERS` assigns solver, ML, service/security, and release paths. `CONTRIBUTING.md` documents setup, test markers, numerical-golden policy, issue workflow, and DCO/sign-off choice. `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, `CHANGELOG.md`, Apache-2.0 `LICENSE`, and `NOTICE` ship before v0.1.

<a id="deprecation"></a>
## Deprecation process

A deprecation identifies replacement, first deprecated version, earliest removal version/date, and migration example. Python emits `DeprecationWarning`; CLI emits one stderr warning per invocation; HTTP adds `Deprecation` and `Sunset` headers where applicable. Tests cover both old and replacement paths throughout the window.
