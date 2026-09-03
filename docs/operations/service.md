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
