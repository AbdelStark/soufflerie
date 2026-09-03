# Public service admission

The HTTP application constructs one process-local `AdmissionController` before
serving requests. It fails before prediction or solve work when a request is
outside the reviewed public budget. The controller is deliberately not an
authentication or multi-tenant boundary.

## Fixed limits

The maximum v0.1 policy is 60 prediction requests per minute per client, two
solve admissions per hour per client, 20 solve admissions per UTC day globally,
two running solves, eight queued solves, and a 180-second solve deadline.
`ServiceConfig` may lower any limit but its strict schema rejects values above
those maxima.

Per-client controls are continuously refilled token buckets. A denied response
is `429 RATE_LIMITED` with an integer `Retry-After`. Global solve count and GPU
budget reset at 00:00 UTC; their denial is `429 BUDGET_EXHAUSTED` with
`Retry-After` to that boundary. Each new solve conservatively reserves its full
configured deadline from `solve_gpu_seconds_per_day`, so concurrent admissions
cannot promise more GPU time than the daily ceiling. Idempotent replay and
conflicting reuse are resolved by the job manager before this new-work
reservation, and capacity rejection does not consume it.

Client bucket maps are capped at 4,096 entries per operation and expire inactive
entries after that operation's full rate window. When the map is full, an unseen
client receives a bounded 429 until the earliest entry expires. Only an HMAC-SHA256
digest is retained; aggregate snapshots contain counts and reserved seconds but
never source addresses or client digests.

## Client address trust

By default, the TCP peer is the client identity and every `X-Forwarded-For`
header is ignored. Set `SOUFFLERIE_TRUSTED_PROXIES` to a comma-separated list of
at most 16 exact IP addresses or CIDRs only when those networks sanitize and
append the header. The resolver starts at the immediate peer, removes only
allowlisted hops from right to left, and selects the first untrusted address.
Malformed, duplicate, oversized, or untrusted forwarding input falls back to
the peer identity.

`SOUFFLERIE_CLIENT_HMAC_KEY` is a provider-managed secret encoded as exactly 64
lowercase hexadecimal characters. If it is absent, the process generates an
ephemeral 32-byte key. Since rate state is process-local, both the state and an
ephemeral identifier reset on restart. Neither form is emitted in responses,
events, metrics, or health.

## Kill switch and readiness

`SOUFFLERIE_SOLVE_ENABLED=false` closes new solve admission with
`503 SOLVE_DISABLED`; setting it true cannot override a disabled `ServiceConfig`.
`SOUFFLERIE_SOLVE_GPU_SECONDS_PER_DAY` may lower, but never raise, the checked
configuration ceiling. The repository example leaves both controls at zero.

Readiness precedence is fixed:

1. artifact identity/integrity failure closes prediction and solve;
2. dependency, device, or warmup failure closes prediction and solve;
3. kill-switch or daily-budget closure leaves prediction ready and closes solve;
4. validation red remains ready and visible;
5. otherwise both configured operations are ready.

Public `/health` remains an allowlist and reports prediction readiness. It does
not reveal client state, proxy topology, budget counters, or the reason a solve
guard is closed. In-memory limits apply per process and are not a claim of
distributed enforcement.

## Remote solve execution

`SolveJobManager` calls one `RemoteSolveExecutor` only after idempotency,
capacity, and admission have accepted new work. The executor obtains the
request-bound prediction first, then delegates to the provider-neutral mounted
volume backend. The fixed public numerical policy is the v0.1 dataset policy:
`512x640`, 20,000 steps, 10,000 warmup steps, inlet velocity `0.05`, and seed
`20260901`. Physical shape and Reynolds number come only from the validated
public request.

The backend encodes one canonical remote request with `service-<job_id>` as its
attempt token, invokes the remote worker once, and passes the HTTP correlation
ID unchanged. Provider retries stay disabled. If the 180-second manager deadline
cancels the invocation, the Modal adapter requests cancellation with container
termination. Disconnecting an SSE client still does not cancel the job.

Successful remote calls return only an `ArtifactRef`. The mounted reader reloads
the volume, opens the reference through `LocalRunArtifactStore`, and checks the
case, design, split, clean source revision, lock digest, selected device, config
digest, and seed before the shared field projector runs. The response binds the
run/provenance digests and computes comparison metrics through the same fp64
validation reductions used offline. Any mismatch becomes a typed terminal
failure; no path, provider detail, or unverified payload reaches the client.
