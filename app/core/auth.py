"""
API key authentication — deployment-only (Phase 7).

Deliberately simple: a static set of accepted keys, sent via the `X-API-Key`
header, compared with a constant-time comparison. This is NOT a user/session
system (no accounts, no per-user quotas beyond what rate_limit.py adds
per-key) — appropriate for a single-operator deployment where the goal is
"don't let strangers spend my API budget," not multi-tenant access control.
A real multi-user product would need something considerably more than this;
that's out of scope for what this project needs.

`require_api_key` is a FastAPI dependency. When `API_AUTH_ENABLED=false`
(the default — see app/core/config.py) it's a no-op, so every existing
route/test that doesn't set a header keeps working unchanged.
"""
from __future__ import annotations

import hmac

from fastapi import Header

from app.core.config import settings
from app.core.errors import AuthenticationError


def _accepted_keys() -> set[str]:
    return {k.strip() for k in settings.api_keys.split(",") if k.strip()}


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.api_auth_enabled:
        return

    if x_api_key is None:
        raise AuthenticationError(user_message="Missing X-API-Key header.")

    # Constant-time comparison against each accepted key — a plain `in` check
    # on a set would leak timing information about key length/content via
    # Python's string equality short-circuiting. Low-stakes here (single-
    # operator API keys, not user passwords) but costs nothing to do right.
    if not any(hmac.compare_digest(x_api_key, key) for key in _accepted_keys()):
        raise AuthenticationError(user_message="Invalid API key.")
