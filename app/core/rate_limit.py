"""
In-memory rate limiting — deployment-only (Phase 7), single-process only.

WHAT THIS IS: a sliding-window request counter per client (API key if
present, else IP address), held in a plain dict in process memory.

WHAT THIS DELIBERATELY IS NOT: a distributed rate limiter. If this app ever
runs as more than one process/instance, each instance enforces its own
independent limit — a client could get `N * instance_count` requests through
rather than `N`. Fixing that needs a shared store (Redis is the standard
choice) coordating counts across instances. That's explicitly NOT added here
— per the approved plan's constraint against reaching for Redis/Kafka/
Temporal-style infrastructure before there's a real multi-instance
deployment to justify it. This limitation is deliberate and documented, not
an oversight — see DEPLOYMENT.md.

Why in-memory is still worth having now: it's genuine, real insurance against
a single-instance deployment being trivially hammered (accidentally or not)
into a large API bill, at zero infrastructure cost. That's the actual problem
Phase 7 needs to solve today; distributed correctness is a problem for
whenever this app actually runs as more than one instance.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        # client_key -> deque of request timestamps within the current window
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        # NOTE: falls back to request.client.host, which is the DIRECT TCP
        # peer — behind a reverse proxy/load balancer, that's the proxy's
        # IP for every client, collapsing everyone onto one shared budget.
        # Fixing that means trusting an X-Forwarded-For (or similar) header,
        # which is itself unsafe to trust blindly (a client can set it) unless
        # the proxy is known to overwrite rather than append it. Not handled
        # here — deployments behind a proxy should prefer API-key-based
        # limiting (this branch is unaffected) or configure their proxy to
        # set a trusted forwarded-for header before revisiting this. See
        # DEPLOYMENT.md.
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key}"
        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled or request.url.path == "/health":
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        window = settings.rate_limit_window_seconds
        limit = settings.rate_limit_requests_per_window

        timestamps = self._requests[key]
        while timestamps and now - timestamps[0] > window:
            timestamps.popleft()

        if len(timestamps) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "rate_limited",
                    "message": f"Rate limit exceeded: {limit} requests per {window}s.",
                    "trace_id": None,
                },
            )

        timestamps.append(now)
        return await call_next(request)
