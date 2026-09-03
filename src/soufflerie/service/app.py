"""Thin FastAPI trust boundary for the version-one service contracts."""

from __future__ import annotations

from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from soufflerie.config import ServiceConfig
from soufflerie.errors import (
    ArtifactIntegrityError,
    CapacityError,
    ConfigurationError,
    DependencyUnavailableError,
    DeviceUnavailableError,
    DomainError,
    InternalInvariantError,
    JobNotFoundError,
    NonConvergenceError,
    NumericalStabilityError,
    RemoteExecutionError,
    SchemaVersionError,
    SoufflerieError,
    ValidationGateError,
)
from soufflerie.observability import new_correlation_id
from soufflerie.service.contracts import (
    MAX_REQUEST_BODY_BYTES,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    PublicError,
    PublicErrorCode,
    PublicErrorResponse,
    ReadinessProbe,
    SolveAccepted,
    SolveEvent,
    SolveStatus,
    Uuid7,
    assess_health,
)

_BODY_LIMIT_PATHS = frozenset({"/predict", "/solve"})
_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    413: {"model": PublicErrorResponse, "description": "Request body exceeds 16 KiB."},
    415: {"model": PublicErrorResponse, "description": "Content type is not JSON."},
    422: {"model": PublicErrorResponse, "description": "Request violates the closed schema."},
    500: {"model": PublicErrorResponse, "description": "Unexpected internal failure."},
    503: {"model": PublicErrorResponse, "description": "Required capacity is unavailable."},
}
_LOOKUP_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": PublicErrorResponse, "description": "Solve job is unknown or expired."},
    422: _ERROR_RESPONSES[422],
    500: _ERROR_RESPONSES[500],
    503: _ERROR_RESPONSES[503],
}
_SSE_RESPONSES: dict[int | str, dict[str, object]] = {
    200: {
        "description": "Replayable SSE frames whose data is one SolveEvent JSON object.",
        "content": {
            "text/event-stream": {
                "schema": {
                    "type": "string",
                    "x-event-data-schema": {"$ref": "#/components/schemas/SolveEvent"},
                }
            }
        },
    },
    **{
        status: {
            "description": details["description"],
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/PublicErrorResponse"}}
            },
        }
        for status, details in _LOOKUP_ERROR_RESPONSES.items()
    },
}

_ERROR_STATUS: tuple[tuple[type[SoufflerieError], int, str], ...] = (
    (JobNotFoundError, 404, "solve job is unknown or expired"),
    (DomainError, 422, "request is outside the supported domain"),
    (ConfigurationError, 422, "request configuration is invalid"),
    (SchemaVersionError, 422, "request schema version is unsupported"),
    (CapacityError, 503, "service capacity is temporarily unavailable"),
    (ArtifactIntegrityError, 503, "artifact integrity verification failed"),
    (DependencyUnavailableError, 503, "required service dependency is unavailable"),
    (DeviceUnavailableError, 503, "requested service device is unavailable"),
    (RemoteExecutionError, 503, "remote execution is temporarily unavailable"),
    (NumericalStabilityError, 503, "reference solve was numerically unstable"),
    (NonConvergenceError, 503, "reference solve did not converge"),
    (ValidationGateError, 503, "requested operation requires green validation"),
    (InternalInvariantError, 500, "internal service invariant failed"),
)


def _response(
    *,
    status_code: int,
    code: PublicErrorCode,
    message: str,
    retryable: bool,
    correlation_id: str,
) -> JSONResponse:
    payload = PublicErrorResponse(
        error=PublicError(
            code=code,
            message=message,
            retryable=retryable,
            correlation_id=correlation_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers={"X-Correlation-ID": correlation_id},
    )


def _correlation_id(request: Request) -> str:
    value = getattr(request.state, "correlation_id", None)
    return value if isinstance(value, str) else new_correlation_id()


async def _send_with_correlation(
    send: Send,
    message: Message,
    *,
    correlation_id: str,
) -> None:
    if message["type"] == "http.response.start":
        headers = [
            (name, value)
            for name, value in message.get("headers", [])
            if name.lower() != b"x-correlation-id"
        ]
        headers.append((b"x-correlation-id", correlation_id.encode("ascii")))
        message = {**message, "headers": headers}
    await send(message)


class RequestBoundaryMiddleware:
    """Generate correlation IDs and bound JSON bodies before route parsing."""

    def __init__(self, app: ASGIApp, *, body_limit: int = MAX_REQUEST_BODY_BYTES) -> None:
        self.app = app
        self.body_limit = body_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        correlation_id = new_correlation_id()
        scope.setdefault("state", {})["correlation_id"] = correlation_id

        async def send_with_correlation(message: Message) -> None:
            await _send_with_correlation(send, message, correlation_id=correlation_id)

        if scope["method"] != "POST" or scope["path"] not in _BODY_LIMIT_PATHS:
            await self.app(scope, receive, send_with_correlation)
            return

        headers = cast(list[tuple[bytes, bytes]], scope.get("headers", []))
        content_types = [value for name, value in headers if name.lower() == b"content-type"]
        media_type = (
            content_types[0].split(b";", maxsplit=1)[0].strip().lower()
            if len(content_types) == 1
            else b""
        )
        if media_type != b"application/json":
            response = _response(
                status_code=415,
                code="UNSUPPORTED_MEDIA_TYPE",
                message="content type must be application/json",
                retryable=False,
                correlation_id=correlation_id,
            )
            await response(scope, receive, send_with_correlation)
            return

        content_lengths = [value for name, value in headers if name.lower() == b"content-length"]
        if len(content_lengths) > 1:
            response = _response(
                status_code=422,
                code="REQUEST_INVALID",
                message="request has invalid framing metadata",
                retryable=False,
                correlation_id=correlation_id,
            )
            await response(scope, receive, send_with_correlation)
            return
        if content_lengths:
            try:
                declared_length = int(content_lengths[0].decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                declared_length = -1
            if declared_length < 0:
                response = _response(
                    status_code=422,
                    code="REQUEST_INVALID",
                    message="request has invalid framing metadata",
                    retryable=False,
                    correlation_id=correlation_id,
                )
                await response(scope, receive, send_with_correlation)
                return
            if declared_length > self.body_limit:
                response = _response(
                    status_code=413,
                    code="REQUEST_TOO_LARGE",
                    message=f"request body must not exceed {self.body_limit} bytes",
                    retryable=False,
                    correlation_id=correlation_id,
                )
                await response(scope, receive, send_with_correlation)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.body_limit:
                response = _response(
                    status_code=413,
                    code="REQUEST_TOO_LARGE",
                    message=f"request body must not exceed {self.body_limit} bytes",
                    retryable=False,
                    correlation_id=correlation_id,
                )
                await response(scope, receive, send_with_correlation)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_body() -> Message:
            nonlocal replayed
            if replayed:
                return {"type": "http.request", "body": b"", "more_body": False}
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_body, send_with_correlation)


class EventStreamResponse(StreamingResponse):
    """OpenAPI media declaration for a stream of typed SSE data records."""

    media_type = "text/event-stream"


def _request_error(error: RequestValidationError) -> tuple[PublicErrorCode, str]:
    issues = error.errors()
    if any(isinstance(issue.get("ctx", {}).get("error"), SchemaVersionError) for issue in issues):
        return ("SCHEMA_UNSUPPORTED", "request schema version is unsupported")
    domain_error_types = {
        "finite_number",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
    }
    if any(
        issue.get("type") in domain_error_types
        and any(
            part in {"aspect_ratio", "rotation_deg", "scale", "reynolds"}
            for part in issue.get("loc", ())
        )
        for issue in issues
    ):
        return ("CASE_OUT_OF_DOMAIN", "request is outside the supported domain")
    return ("REQUEST_INVALID", "request body does not match the closed schema")


def create_app(
    *,
    config: ServiceConfig,
    readiness: ReadinessProbe,
    package_version: str,
) -> FastAPI:
    """Create the contract service without loading model or remote runtimes."""

    health = assess_health(config, readiness, package_version=package_version)
    app = FastAPI(
        title="Soufflerie HTTP API",
        summary="Strict prediction and asynchronous reference-solve contracts.",
        description=(
            "Version-one educational wind-tunnel API. A red validation report remains visible "
            "and serviceable; integrity or identity failures make prediction unavailable."
        ),
        version=package_version,
        openapi_version="3.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.add_middleware(RequestBoundaryMiddleware)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        code, message = _request_error(error)
        return _response(
            status_code=422,
            code=code,
            message=message,
            retryable=False,
            correlation_id=_correlation_id(request),
        )

    @app.exception_handler(SoufflerieError)
    async def soufflerie_error_handler(request: Request, error: SoufflerieError) -> JSONResponse:
        for error_type, status_code, message in _ERROR_STATUS:
            if isinstance(error, error_type):
                return _response(
                    status_code=status_code,
                    code=cast(PublicErrorCode, error.code),
                    message=message,
                    retryable=error.retryable,
                    correlation_id=_correlation_id(request),
                )
        return _response(
            status_code=500,
            code="SOUFFLERIE_ERROR",
            message="service operation failed",
            retryable=error.retryable,
            correlation_id=_correlation_id(request),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code: PublicErrorCode = "METHOD_NOT_ALLOWED" if error.status_code == 405 else "NOT_FOUND"
        message = "HTTP method is not allowed" if error.status_code == 405 else "route not found"
        return _response(
            status_code=error.status_code,
            code=code,
            message=message,
            retryable=False,
            correlation_id=_correlation_id(request),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
        del error
        return _response(
            status_code=500,
            code="INTERNAL_ERROR",
            message="unexpected internal service error",
            retryable=False,
            correlation_id=_correlation_id(request),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        operation_id="health_v1",
        summary="Return allowlisted liveness and readiness identities",
        responses={500: _ERROR_RESPONSES[500]},
    )
    async def get_health() -> HealthResponse:
        return health

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        operation_id="predict_v1",
        summary="Predict fields and consistency for one supported design",
        responses=_ERROR_RESPONSES,
    )
    async def predict(request: PredictionRequest) -> PredictionResponse:
        del request
        if health.readiness != "ready":
            raise DependencyUnavailableError("service is not ready")
        raise DependencyUnavailableError("prediction adapter is not configured")

    @app.post(
        "/solve",
        response_model=SolveAccepted,
        status_code=202,
        operation_id="solve_submit_v1",
        summary="Admit one bounded asynchronous reference solve",
        responses=_ERROR_RESPONSES,
    )
    async def submit_solve(request: PredictionRequest) -> SolveAccepted:
        del request
        if not config.solve_enabled:
            raise DependencyUnavailableError("reference solves are disabled")
        raise DependencyUnavailableError("reference solve adapter is not configured")

    @app.get(
        "/solve/{job_id}",
        response_model=SolveStatus,
        operation_id="solve_status_v1",
        summary="Poll one retained reference-solve state",
        responses=_LOOKUP_ERROR_RESPONSES,
    )
    async def solve_status(job_id: Uuid7) -> SolveStatus:
        raise JobNotFoundError(f"unknown solve job {job_id}")

    @app.get(
        "/solve/{job_id}/events",
        response_model=SolveEvent,
        response_class=EventStreamResponse,
        operation_id="solve_events_v1",
        summary="Open the replayable server-sent solve event stream",
        responses=_SSE_RESPONSES,
    )
    async def solve_events(job_id: Uuid7) -> SolveEvent:
        raise JobNotFoundError(f"unknown solve job {job_id}")

    return app


__all__ = ["EventStreamResponse", "RequestBoundaryMiddleware", "create_app"]
