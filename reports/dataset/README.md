# Canonical dataset evidence

Issue #19's authenticated `canonical-lhs-v1` sweep published the complete
1,000-case dataset on 2026-09-02. The authoritative artifact is
`/data/soufflerie/v1/datasets/4aefbbe88a18d233249b` in the Modal volume
`soufflerie-data`. Large run archives and the binary manifest remain in that
immutable store; this directory retains their digest-bound JSON evidence.

## Identity

| Record | Value |
|---|---|
| Dataset ID | `4aefbbe88a18d233249b` |
| Dataset SHA-256 | `4aefbbe88a18d233249bae3dbebc529142a0e9d5dbaaf33dae474c82f121fa28` |
| Manifest SHA-256 | `4a11f40a18f253554ca04cd3e4564cf95f0efc06cf19082bc0dddb60a37ae504` |
| Statistics SHA-256 | `ed9f0388a0539f106ae67efb1e43507130e1ba7679111ef801c5b8cdee70470e` |
| Metadata SHA-256 / `COMMITTED` marker | `d4a6eb7bf9c1fa72fbbe703fe3367f129983eee62c085aea3f11dec7ee59c480` |
| Sweep digest | `5140cf2c5d6f36d097c8716267afed6258470df916a45c2b28f451194db9ce0a` |
| Terminal evidence SHA-256 | `678a232f0fec9a9e3c99394ecc6c2ae1b43b0ddaada035156c639bf9c700a42a` |
| Config SHA-256 | `04ffaf7dcb027482d82c921fa617a429872a06e5d0a92545e3cfaf724a011333` |
| Design SHA-256 | `352a060bdb7ef2ff3e9432d7eff4333d6d3c9bd9aca33c6f29f6f56a307250c1` |
| Split SHA-256 | `406a35bdef46cd10e77efc7fc6b301ffbb089783c2454dc86c8308a093e9d027` |
| Source revision | `c2552fc728e02326f96949d23995b5dfcda1f13e` |
| Lock SHA-256 | `181b61f84e84aa57e5a373373de9b556c033d9510ceb98dfc78110a0e38bbb90` |
| Requested device | `L40S` |

The checked [`metadata.json`](metadata.json),
[`statistics.json`](statistics.json), and
[`sweep-summary.json`](sweep-summary.json) are byte-exact copies of the
published sidecars and terminal output. The terminal summary itself has
SHA-256 `f6592f9bbd479194ff9c5e81cf0c0e3bd9eb95bd77c57b2b3c7f7d9e163d55ec`.

## Acceptance

- All 1,000 intended cases succeeded; pending, running, and failed counts are
  zero. The manifest binds 1,000 unique verified parent-run digests.
- Exact split counts are 600 train, 200 validation, and 200 test.
- Referenced payload is 765,964,584 bytes (730.481 MiB), or 35.668% of the
  strict 2 GiB gate. The committed manifest bundle is 242,947 bytes, including
  the 167,115-byte Parquet manifest.
- Cumulative claimed case attempts are 1,000. Case retries are zero and the
  failure-code ledger is `{}`. The publication invocation submitted zero new
  cases because all 1,000 durable successes were verified and skipped.
- The builder fully reopened every parent archive, required `solver_valid`,
  reproduced the canonical design and splits, and published only after every
  invariant passed. No case was replaced and no partial manifest was exposed.
- Cd has mean `1.0736794715990872` over
  `[0.4405976454168558, 1.9761562782526017]`; mean Cl has mean
  `-0.12774578857450744` over
  `[-0.5631589628756046, 0.19060945197939871]`.
- Strouhal has 0 values and 1,000 nulls. This is explicit release evidence,
  not omitted data: RFC-0005 makes Strouhal nullable, and the v0.1 surrogate
  targets mean fields and Cd rather than Strouhal.

## Execution, resume, and cost

The durable compute app `ap-IZFwvUeBW3ILNjy10hn0kv` ran from 11:26:48 to
13:36:05 CEST. Its local parent disconnected while L40S capacity was queued,
but detached workers continued from leased state and committed all 1,000
cases. The implementation ceiling was 100 workers; Modal enforced 10 observed
concurrent GPU tasks for this workspace. Capacity changed wall time, never the
frozen sample count, domain, or gates.

The resume/publication app `ap-Ylny21vHOwxvEkdNLEgAUE` ran from 13:36:50 to
14:09:37 CEST. It submitted no solves, reopened all completed artifacts, built
the dataset, committed the four-member bundle, reloaded it, and verified it
again. The terminal invocation reports 1,946.272940806 seconds (32.438 minutes);
the observed durable lifecycle from compute-app creation through publication
was 9,769 seconds (2:42:49), including provider queueing and the handoff.

Solver provenance totals 26,786.943724711953 L40S-seconds, or 7.440817701
GPU-hours. At Modal's official 2026-09-02 L40S rate of
[$0.000542 per second](https://modal.com/pricing), estimated GPU compute is:

```text
26,786.943724711953 s * $0.000542/s = $14.518523499
```

This is a GPU-only estimate. CPU, memory, image build, volume storage, and
network charges are excluded because the untagged workspace billing report
cannot attribute them to these apps without mixing unrelated use.

A preceding app, `ap-wTNrrVN527q0YbBTvllseh`, was preempted before state
initialization. It claimed no canonical case and emitted no run artifact, so it
correctly does not appear as a case failure or retry.

## Reproduce the checks

The accepted release command was:

```bash
uv run --extra remote modal run -d -q infra/sweep.py \
  --config configs/sweeps/mvp-v1.yaml \
  --output /tmp/soufflerie-c2552fc-sweep-summary.json
```

Download and independently validate the committed manifest:

```bash
uv run --extra remote modal volume get soufflerie-data \
  /soufflerie/v1/datasets/4aefbbe88a18d233249b \
  /tmp/soufflerie-dataset

uv run soufflerie dataset validate \
  --manifest /tmp/soufflerie-dataset/4aefbbe88a18d233249b/manifest.parquet
```

The observed validator result was:

```text
dataset valid: 4aefbbe88a18d233249b (1000 cases)
```

The synthetic manifest under `tests/fixtures/` remains contract data and is
not canonical run evidence.
