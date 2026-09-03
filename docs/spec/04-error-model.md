# Error model

<a id="principles"></a>
## Principles

Errors are typed, stable enough for automation, and safe to display. The system fails before expensive work where possible, never retries deterministic invalid input, and never converts a validation failure into success. Exceptions chain internal causes for logs while public responses expose only stable codes and sanitized messages.

<a id="taxonomy"></a>
## Taxonomy

```python
class SoufflerieError(Exception):
    code: str
    retryable: bool

class ConfigurationError(SoufflerieError): ...       # CONFIG_INVALID
class DomainError(SoufflerieError): ...              # CASE_OUT_OF_DOMAIN
class NumericalStabilityError(SoufflerieError): ...  # SOLVER_UNSTABLE
class NonConvergenceError(SoufflerieError): ...       # SOLVER_NOT_CONVERGED
class ArtifactIntegrityError(SoufflerieError): ...   # ARTIFACT_INTEGRITY
class SchemaVersionError(SoufflerieError): ...       # SCHEMA_UNSUPPORTED
class DependencyUnavailableError(SoufflerieError): ... # DEPENDENCY_UNAVAILABLE
class DeviceUnavailableError(SoufflerieError): ...   # DEVICE_UNAVAILABLE
class CapacityError(SoufflerieError): ...             # CAPACITY_EXHAUSTED
class RateLimitError(SoufflerieError): ...            # RATE_LIMITED
class BudgetExhaustedError(SoufflerieError): ...      # BUDGET_EXHAUSTED
class SolveDisabledError(SoufflerieError): ...        # SOLVE_DISABLED
class RemoteExecutionError(SoufflerieError): ...      # REMOTE_EXECUTION
class ValidationGateError(SoufflerieError): ...       # VALIDATION_RED
class InternalInvariantError(SoufflerieError): ...    # INTERNAL_INVARIANT
```

`retryable` is fixed by the error instance and cause: invalid configuration, unsupported schema, integrity mismatch, and numerical instability are not retryable unchanged; expired leases, transient remote failures, and capacity timeouts may be.

<a id="failure-matrix"></a>
## Failure matrix

| Failure | Detection | System response | Recovery |
|---|---|---|---|
| Invalid or unsupported case | Strict parse/preflight | Reject before allocation | Correct input |
| Lattice tau/Mach bound violation | Derived-config validation | Reject case; record reason | Change resolution or config through controlled experiment |
| Non-finite lattice state | Per-interval finite/reasonable-range check | Stop run, publish failed state without run artifact | Diagnose; never admit to manifest |
| Mass drift or convergence gate failure | Post-run diagnostics | Preserve diagnostic artifact; mark invalid | Solver change or longer declared run |
| Preempted sweep worker | Lease expiry | Requeue same `case_id` | Idempotent retry |
| Divergent duplicate result | Digest comparison | Fail sweep integrity | Investigate nondeterminism |
| Partial artifact upload | Temp-object/commit marker check | Reader ignores temp state | Retry publication |
| Checksum/schema mismatch | Reader preflight | Refuse deserialization | Restore matching artifact |
| Missing optional dependency | Lazy adapter import | Explain required extra | Install declared extra |
| Validation gate red | Report evaluator | Serve prediction with red state/banner | Improve model/data; gate unchanged |
| Remote capacity unavailable | Admission timeout | `503 CAPACITY_EXHAUSTED` | Bounded jittered retry |
| Public solve timeout | Job watchdog | Cancel, terminal failed event | Retry later if policy permits |
| Unknown internal exception | Boundary handler | Correlation ID + generic `500` | Inspect sanitized logs |

<a id="http-errors"></a>
## HTTP errors

```json
{
  "schema_version": 1,
  "error": {
    "code": "CASE_OUT_OF_DOMAIN",
    "message": "reynolds must be between 40 and 300",
    "retryable": false,
    "correlation_id": "019..."
  }
}
```

Mapping: parse/domain `422`; request size/media `413/415`; unknown job `404`; capacity/rate limit `429` or `503` with `Retry-After`; integrity/dependency/device/readiness `503`; unexpected invariant `500`. Validation-red predictions remain `200` because the response is valid and explicitly carries `validation_status="red"`.

<a id="cli-exit-codes"></a>
## CLI exit codes

| Code | Meaning |
|---:|---|
| `0` | Command completed and all requested gates passed |
| `2` | CLI usage or configuration invalid |
| `3` | Domain or numerical case invalid |
| `4` | Artifact or schema invalid |
| `5` | Dependency or device unavailable |
| `6` | Validation completed with a red required gate |
| `7` | Remote or capacity failure after retry policy |
| `70` | Unexpected internal error |

<a id="retry-policy"></a>
## Retry policy

Only idempotent operations retry automatically. Remote calls use at most three attempts, full-jitter exponential delay capped at 30 seconds, and a total deadline declared by the caller. Training does not automatically restart from an arbitrary point; it resumes only from a verified checkpoint whose experiment identity matches. API clients receive retry guidance but the server never recursively submits duplicate solve jobs.
