# Remote training

Remote training runs the three seeds declared by one `TrainingConfig` against
the exact published dataset reference in `reports/dataset/sweep-summary.json`.
Run it only from a clean reviewed commit:

```bash
uv run --extra remote modal run infra/train.py \
  --config configs/training/fno-v1.yaml \
  --output reports/training/index.json
```

The local boundary loads the dataset `ArtifactRef`, binds it to the config,
fixed FNO architecture, full source revision, full lock digest, and explicit
L40S or A10G class, and stages canonical JSON below 16 KiB. The three seed
workers share no optimizer state and Modal admits at most three containers.
Each worker has a 75-minute timeout and provider retries are disabled.

Every epoch consumes the complete 600-row train split, evaluates only the
200-row validation split, appends authoritative JSONL metrics, and atomically
publishes private model/optimizer/scheduler/scaler/RNG state. A volume commit
occurs only after the epoch record, validation score, latest checkpoint, and
best-checkpoint pointer agree. Resume starts at the next epoch boundary and
requires exact experiment, dataset, config, architecture, source, lock, seed,
CUDA device/name/capability, and precision identity. Any difference fails with
`REMOTE_TRAIN_RESUME_MISMATCH` or the stricter `TRAIN-13 RESUME` domain error;
the worker never changes GPU or precision silently.

After all seeds commit completion evidence, every worker independently calls
the domain `freeze_validation_selection` function. Infrastructure neither
defines the score nor chooses a model. The frozen decision selects each seed's
earliest minimum validation score, then the deployable seed by score and seed
tie-break. Only then is each best checkpoint decoded and exported as a safe
weights-only model bundle. Test data is structurally absent from this stage.

`reports/training/index.json` is the small digest-bound handoff to validation.
It contains the immutable request, three ordered receipts, three distinct
model references, selection identity/deployable seed, baseline identities,
solver-run lineage digest, timing, GPU seconds, peak allocated/reserved bytes,
device name/class, precision, source, lock, and full parent digests. A red or
missing later validation report does not alter these training artifacts.

The persistent layout is:

```text
/data/soufflerie/v1/
  requests/training/<request-sha256>.json
  training/<experiment-id>/<seed>/
    epochs.jsonl
    validation-metrics.json
    latest.json
    best.json
    completion.json
    receipt.json
    checkpoints/<checkpoint-id>/...
  training/<experiment-id>/selection.json
  models/<model-id>/...
```

Do not edit, copy over, or delete any path referenced by the index. Rerunning
the identical command verifies and resumes the same experiment. Changing the
data, architecture, config, seed set, source, or lock creates a new experiment
identity rather than overwriting the prior run.
