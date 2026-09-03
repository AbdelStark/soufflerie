# Soufflerie synthetic CPU smoke FNO

A deterministic untrained fixture for installed-wheel bundle and CPU inference smoke.

## Identity

- Model ID: `ce9b9a70fa9ba548f8b0`
- Dataset ID: `40f9dc98d92b65c80575`
- Dataset SHA-256: `40f9dc98d92b65c8057508efd08db7c3f49d59e3cec8cd526328f749ecce3d72`
- Experiment ID: `2a3bf06a29fd567b7fa2`
- Architecture: `fno2d-v1`
- Selected epoch: `1`
- Training seed: `0`
- Weights SHA-256: `c65e1768098f97a68f0f0c79f13608c4246cffe8fc0b2ed802356cf3416a0c8c`
- Source revision: `509115ce8cb50d224eb3705f2b276ab3da41e647`
- License: `Apache-2.0`

## Intended use

- Installed-wheel integrity and CPU inference contract testing.

## Validation gates

| Gate | Status | Threshold | Measured | Evidence SHA-256 |
| --- | --- | --- | --- | --- |
| Scientific validation | not_evaluated | Requires a separately trained and RFC-0008-validated model | not evaluated | not available |

## Limitations

- Synthetic zero-weight fixture with fixed output biases; not flow-accuracy evidence.
- Do not use this fixture for scientific prediction or release-quality claims.
