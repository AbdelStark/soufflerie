# Security

<a id="security-objectives"></a>
## Security objectives

Soufflerie protects maintainer credentials and GPU budget, prevents untrusted inputs or artifacts from executing code, preserves artifact integrity, and avoids overstating model trust. It does not provide confidential user storage or a multi-tenant security boundary.

<a id="trust-boundaries"></a>
## Trust boundaries

1. **Public client to service.** All bytes, headers, numeric values, and reconnect cursors are untrusted.
2. **Local maintainer to remote execution provider.** Local configuration is trusted after schema validation; transport, capacity, and worker lifetime can fail.
3. **Worker to persistent artifact store.** Objects may be partial, stale, duplicated, corrupted, or from an incompatible schema.
4. **Artifact store to model/solver process.** Metadata is untrusted until size, type, digest, and schema checks pass.
5. **Repository to release consumer.** Source, wheels, checkpoints, generated reports, and public claims must be traceable and free of credentials.

<a id="threat-model"></a>
## Threat model

| Threat | Control |
|---|---|
| Oversized, NaN, or adversarial request causes resource exhaustion | Strict Pydantic parsing, 16 KiB body limit, closed numeric ranges, queue bounds, request and job deadlines |
| Public solve abuse consumes GPU budget | Per-client and global rate limits, separate solve capacity, daily GPU-time budget, deployment kill-switch |
| Credential leaks through repository or logs | Ignored `.env`, committed `.env.example` names only, provider secret store, redaction, secret scanning |
| Malicious pickle/checkpoint executes code | NPZ `allow_pickle=False`; weights-only safe tensor format; JSON metadata; no generic Python object loading |
| Corrupt or swapped artifact changes evidence | SHA-256 digests, parent lineage, schema/type checks, atomic publication, startup identity match |
| Validation state hidden or forged in UI | Report signature-by-digest, server-derived status, immutable response field, visible banner contract and UI tests |
| Server-side request forgery/path traversal | No user-supplied URLs or paths; artifact keys derive from validated IDs; resolved paths constrained to configured roots |
| Dependency compromise | Exact lockfile, hashes where supported, dependency audit, least-dependency runtime profiles, reviewed updates |
| Cross-origin drive-by solve submission | Explicit allowlist CORS, no wildcard with credentials, non-simple JSON request, rate limit |
| Information disclosure via health/errors | Allowlisted health fields, generic unexpected errors, correlation IDs, no environment or stack traces |

<a id="secrets"></a>
## Secrets handling

Remote tokens are created and stored through provider-supported authentication outside Git. `.env` is local-only; `.env.example` contains variable names and safe comments without sample credentials. Processes read only named settings, never dump the environment, and fail closed when required credentials are absent. CI has no remote execution credentials. Production-like deploy jobs use a protected environment and least-privileged token. Rotation invalidates prior tokens and triggers a redaction audit of logs and history.

<a id="artifact-safety"></a>
## Artifact safety

- NPZ archives use primitive numeric arrays and `allow_pickle=False`.
- Model weights use `safetensors`; optimizer state, if serialized through a framework format, is private training state and never loaded from an untrusted source or shipped as a public checkpoint.
- Parquet readers project declared columns and enforce row/byte limits.
- Archives cannot contain absolute paths, parent traversal, symbolic links, or executable files.
- A model bundle is read only after its manifest and every member digest verify.
- The service starts unready if bundle and validation identities differ.

<a id="service-controls"></a>
## Service controls

The v0.1 demo has no user accounts. Prediction may be public; reference solve is public only when rate limit and budget guard are active. Defaults are 60 predictions/minute/client, 2 solves/hour/client, 20 solves/day globally, 2 concurrent solves, and a 180-second solve deadline. Operator configuration may lower these values, never exceed them without a reviewed config change. Client identification is a privacy-preserving keyed hash of the normalized source address retained only for the rate-limit window.

If the daily GPU budget, validation-identity check, or artifact-integrity check fails, solve admission closes. Prediction closes only when its model/report integrity fails; a red validation report does not close it and must remain visible.

<a id="vulnerability-management"></a>
## Vulnerability management

`SECURITY.md` names supported versions, private reporting instructions, acknowledgment expectations, and response targets without promising unavailable staffing. CI runs dependency and secret scanning. Release commits must be clean, and built artifacts receive provenance plus checksums. Security fixes may break compatibility when exploitation risk outweighs deprecation; the changelog must say so.

<a id="security-non-goals"></a>
## Security non-goals

The service does not accept user files, arbitrary SDFs, code, URLs, plugins, or model uploads. It stores no user profile or result history beyond bounded ephemeral job state. Adding any of those capabilities requires a new threat-model RFC.
