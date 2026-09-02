"""
LoggingTracer: the default Tracer when Langfuse isn't configured.

Makes trace/span behavior directly observable via standard logging, with zero
external dependency — used by every test in this project (so tracing can
never make a test flaky or network-dependent) and as the sane local-dev
default. "No Langfuse keys configured" is a normal, expected state, not a
degraded one — the pipeline's actual behavior (what gets called, in what
order, with what input/output) is identical either way; only the tracing
BACKEND differs.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any

from app.observability.base import SpanHandle, Tracer

logger = logging.getLogger("rag.trace")


class _LoggingSpanHandle(SpanHandle):
    def __init__(self) -> None:
        self.output: Any = None
        self.error: str | None = None
        self.usage: dict | None = None

    def set_input(self, value: Any) -> None:
        logger.debug("span_input input=%r", value)

    def set_output(self, value: Any) -> None:
        self.output = value

    def set_error(self, message: str) -> None:
        self.error = message

    def set_usage(
        self, *, input_tokens: int | None = None, output_tokens: int | None = None, model: str | None = None
    ) -> None:
        self.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "model": model}


class LoggingTracer(Tracer):
    @contextmanager
    def span(self, name: str, *, as_type: str = "span", input: Any = None, metadata: dict | None = None):
        span_id = uuid.uuid4().hex[:8]
        start = time.monotonic()
        logger.info("span_start name=%s id=%s type=%s input=%r metadata=%r", name, span_id, as_type, input, metadata)
        handle = _LoggingSpanHandle()
        try:
            yield handle
        except Exception as exc:
            handle.set_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            duration_ms = round((time.monotonic() - start) * 1000, 1)
            status = "error" if handle.error else "ok"
            logger.info(
                "span_end name=%s id=%s status=%s duration_ms=%s output=%r usage=%r error=%s",
                name, span_id, status, duration_ms, handle.output, handle.usage, handle.error,
            )

    def flush(self) -> None:
        pass  # nothing buffered — every span is logged synchronously as it happens
