# Configuration examples

Every YAML file in this directory is parsed as one UTF-8, schema-v1 document
with strict scalar types and no unknown keys. Duplicate keys, anchors, aliases,
environment placeholders, non-finite numbers, and unsafe YAML tags are rejected.
Comments and key ordering never participate in `config_digest`; validated typed
content is canonicalized before hashing.

The canonical sweep and case examples are executable inputs. Training,
validation, and service examples use 20-character all-zero or low-numbered
artifact identity sentinels because the dataset, model, and report artifacts do
not exist yet. Downstream entrypoints must resolve those identities to verified
artifacts before work starts; these examples do not claim completed training,
validation, or deployment.

Run `uv run python scripts/validate_schemas.py` after changing a model or example.
Regenerate schema JSON with `uv run python scripts/export_schemas.py`.

The canonical sweep records its validated config, design, and split SHA-256
digests in comments at the top of [`sweeps/mvp-v1.yaml`](sweeps/mvp-v1.yaml).
Comments are outside the config identity boundary. The reproducible statistics
and preflight evidence live in
[`reports/data/design-mvp-v1.json`](../reports/data/design-mvp-v1.json).
