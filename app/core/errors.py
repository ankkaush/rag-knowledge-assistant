"""
Error hierarchy: internal exceptions vs. user-facing errors.

The core idea (from the approved plan, security-from-Phase-1 requirement):
every exception raised inside the app carries full diagnostic detail for logging,
but nothing that reaches an HTTP response is allowed to contain that detail —
no stack traces, no API keys, no raw document content, no internal file paths.

How this is enforced structurally:
- `AppError` and its subclasses are the ONLY exceptions that are allowed to cross
  the API boundary as a deliberate, categorized error. Each has a `user_message`
  (safe to return) and an `internal_detail` (logged, never returned).
- Anything that is NOT an AppError (a bug, an unexpected third-party exception)
  is caught by a catch-all handler at the API layer, logged with full detail,
  and converted to a generic "internal_error" response. The caller never sees
  the raw exception.
"""
from __future__ import annotations


class AppError(Exception):
    """Base class for all deliberately-raised, categorized application errors."""

    error_code: str = "internal_error"
    status_code: int = 500

    def __init__(self, user_message: str, internal_detail: str | None = None):
        super().__init__(internal_detail or user_message)
        self.user_message = user_message
        self.internal_detail = internal_detail or user_message


class ValidationError(AppError):
    """Input failed validation — permanent error, never retried."""

    error_code = "bad_request"
    status_code = 400


class NotFoundError(AppError):
    error_code = "not_found"
    status_code = 404


class UpstreamUnavailableError(AppError):
    """
    A downstream dependency (DB, embedding API, LLM API) could not be reached
    after retries were exhausted, or failed permanently.
    """

    error_code = "upstream_unavailable"
    status_code = 503


class IngestionError(AppError):
    """Document loading/cleaning/chunking failed for a reason specific to that document."""

    error_code = "ingestion_failed"
    status_code = 422


class AuthenticationError(AppError):
    """Missing or invalid API key. Deployment-only (Phase 7) — see app/core/auth.py."""

    error_code = "unauthorized"
    status_code = 401


class RateLimitExceededError(AppError):
    """Too many requests from this client in the current window. See app/core/rate_limit.py."""

    error_code = "rate_limited"
    status_code = 429
