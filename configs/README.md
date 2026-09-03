# Configuration examples

Every YAML file in this directory is parsed as one UTF-8, schema-v1 document
with strict scalar types and no unknown keys. Duplicate keys, anchors, aliases,
environment placeholders, non-finite numbers, and unsafe YAML tags are rejected.
Comments and key ordering never participate in `config_digest`; validated typed
content is canonicalized before hashing.

The canonical sweep and case examples are executable inputs. Training pins the
published canonical dataset identity. Validation and service examples retain
20-character all-zero or low-numbered model/report identity sentinels until the
canonical three-seed training run publishes its index. Downstream entrypoints
must resolve every identity to a verified artifact before work starts; sentinel
examples do not claim completed validation or deployment. The validation smoke
uses only 100 deterministic bootstrap resamples, but it still requires real
dataset/model/baseline parents and evaluates the fixed test/OOD/sensitivity
memberships rather than fabricating reduced evidence.

Run `uv run python scripts/validate_schemas.py` after changing a model or example.
Regenerate schema JSON with `uv run python scripts/export_schemas.py`.

The canonical sweep records its validated config, design, and split SHA-256
digests in comments at the top of [`sweeps/mvp-v1.yaml`](sweeps/mvp-v1.yaml).
Comments are outside the config identity boundary. The reproducible statistics
and preflight evidence live in
[`reports/data/design-mvp-v1.json`](../reports/data/design-mvp-v1.json).
