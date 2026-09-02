"""
LangfuseTracer: the real observability backend.

RELIABILITY GUARANTEE (agreed before Phase 1 began, applied here to a new
external dependency): Langfuse being unreachable, misconfigured, or slow must
NEVER fail or delay the user-facing request. Every call into the Langfuse SDK
below is wrapped in its own try/except — if starting a span fails, we fall
back to a no-op span handle and log a warning; the calling pipeline code
never sees the failure and proceeds normally. This is deliberately asymmetric
with how Tracer.span() treats PIPELINE exceptions (which it records and
re-raises, never swallows) — tracing-infrastructure failures and
business-logic failures are different failure domains, and only one of them
should ever be allowed to break a request.

Langfuse's Python SDK (v4+) is itself built on OpenTelemetry:
`start_as_current_observation` opens an OTel span and pushes it onto the
current OTel context, so a nested call made while still inside an outer
span's `with` block automatically becomes a CHILD span — no explicit
parent-passing needed. This is the concrete mechanism behind "OTel is the
underlying standard, Langfuse is a specific tool built on it," named as a
concept to understand in the project plan.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from app.observability.base import SpanHandle, Tracer

logger = logging.getLogger("rag.trace")


class _NoOpSpanHandle(SpanHandle):
    """Used only when Langfuse itself fails to start a span — the fallback of last resort."""

    def set_input(self, value: Any) -> None:
        pass

    def set_output(self, value: Any) -> None:
        pass

    def set_error(self, message: str) -> None:
        pass

    def set_usage(self, **_kwargs: Any) -> None:
        pass


class _LangfuseSpanHandle(SpanHandle):
    def __init__(self, lf_span: Any):
        self._lf_span = lf_span

    def set_input(self, value: Any) -> None:
        self._safe_update(input=value)

    def set_output(self, value: Any) -> None:
        self._safe_update(output=value)

    def set_error(self, message: str) -> None:
        self._safe_update(level="ERROR", status_message=message[:1000])

    def set_usage(
        self, *, input_tokens: int | None = None, output_tokens: int | None = None, model: str | None = None
    ) -> None:
        usage_details = {}
        if input_tokens is not None:
            usage_details["input"] = input_tokens
        if output_tokens is not None:
            usage_details["output"] = output_tokens
        kwargs: dict[str, Any] = {}
        if usage_details:
            kwargs["usage_details"] = usage_details
        if model:
            kwargs["model"] = model
        if kwargs:
            self._safe_update(**kwargs)

    def _safe_update(self, **kwargs: Any) -> None:
        try:
            self._lf_span.update(**kwargs)
        except Exception:
            logger.warning("langfuse_span_update_failed fields=%s", list(kwargs), exc_info=True)


class LangfuseTracer(Tracer):
    def __init__(self, public_key: str, secret_key: str, host: str):
        from langfuse import Langfuse

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", input: Any = None, metadata: dict | None = None):
        lf_context_manager = None
        handle: SpanHandle = _NoOpSpanHandle()
        try:
            lf_context_manager = self._client.start_as_current_observation(
                name=name, as_type=as_type, input=input, metadata=metadata,
            )
            lf_span = lf_context_manager.__enter__()
            handle = _LangfuseSpanHandle(lf_span)
        except Exception:
            logger.warning("langfuse_span_start_failed name=%s", name, exc_info=True)
            lf_context_manager = None

        try:
            yield handle
        except Exception as exc:
            handle.set_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if lf_context_manager is not None:
                try:
                    lf_context_manager.__exit__(None, None, None)
                except Exception:
                    logger.warning("langfuse_span_end_failed name=%s", name, exc_info=True)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.warning("langfuse_flush_failed", exc_info=True)
