"""
FastAPI application entrypoint.

The two exception handlers below are where the "internal vs. user-facing
error" policy (app/core/errors.py) is actually enforced at the API boundary:

- `AppError` (and subclasses) were deliberately raised somewhere in the code
  with both a safe `user_message` and a detailed `internal_detail`. We log the
  detail and return only the safe message.
- Anything else (`Exception`) is unexpected — a bug, an unhandled third-party
  exception. We log it with full traceback for debugging, but the HTTP
  response is a generic, fixed message. The caller never sees a stack trace,
  file path, or exception string that might contain internal details.

DEPLOYMENT HARDENING (Phase 7) WIRING:
- `require_api_key` is attached to the ingestion/retrieval/query routers via
  `dependencies=[...]` on `include_router` — centralized here rather than
  scattered per-route, and easy to audit ("which routes require auth" is
  answered by reading this file, not grepping every router module).
  `/health` deliberately has NO auth dependency: health checks (load
  balancers, container orchestrators) must be reachable without a key.
- `RateLimitMiddleware` and `CORSMiddleware` are both no-ops at their
  respective defaults (`RATE_LIMIT_ENABLED=false`, `CORS_ALLOWED_ORIGINS=""`)
  — see app/core/config.py.
- `validate_production_config()` runs at startup and refuses to boot if
  `APP_ENV=production` with unsafe settings — this is what makes "Phase 7
  is the first point security starts" impossible: production simply won't
  start with auth off, rather than silently running insecurely.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.ingestion import router as ingestion_router
from app.api.query import router as query_router
from app.api.retrieval import router as retrieval_router
from app.api.schemas import ErrorResponse
from app.core.auth import require_api_key
from app.core.config import settings, validate_production_config
from app.core.errors import AppError
from app.core.rate_limit import RateLimitMiddleware
from app.observability import get_tracer

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("rag")

problems = validate_production_config()
if problems:
    raise RuntimeError(
        "Refusing to start with APP_ENV=production and unsafe configuration:\n"
        + "\n".join(f"  - {p}" for p in problems)
    )

@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Langfuse exports spans asynchronously in the background (see
    # app/observability/langfuse_tracer.py); flushing on shutdown means a
    # clean server stop doesn't rely solely on the SDK's own atexit hook to
    # deliver the last few requests' traces. A no-op when LoggingTracer is
    # active. get_tracer() constructs a new client object here, but Langfuse's
    # SDK keys its actual background resources by public key, so this still
    # flushes the same buffered data any other tracer instance created during
    # this process's lifetime — confirmed by reading the SDK's resource-manager
    # source, not assumed.
    get_tracer().flush()


app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0", lifespan=lifespan)

app.add_middleware(RateLimitMiddleware)

if settings.cors_allowed_origins.strip():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

_auth_dep = [Depends(require_api_key)]
app.include_router(ingestion_router, dependencies=_auth_dep)
app.include_router(retrieval_router, dependencies=_auth_dep)
app.include_router(query_router, dependencies=_auth_dep)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    trace_id = str(uuid.uuid4())
    logger.warning(
        "app_error code=%s trace_id=%s path=%s detail=%s",
        exc.error_code, trace_id, request.url.path, exc.internal_detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error_code=exc.error_code, message=exc.user_message, trace_id=trace_id).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    trace_id = str(uuid.uuid4())
    logger.exception("unexpected_error trace_id=%s path=%s", trace_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="internal_error",
            message="An unexpected error occurred. Please try again.",
            trace_id=trace_id,
        ).model_dump(),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "env": settings.app_env}
