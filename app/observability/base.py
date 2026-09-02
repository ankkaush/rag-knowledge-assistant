"""
Tracer / SpanHandle interface.

CONCEPT — trace and span:
A TRACE is one end-to-end operation (one /query request, one document
ingestion). A SPAN is one step inside it (retrieve, generate, embed_batch)
with a start/end time, inputs/outputs, and status. In this codebase there is
no separate "Trace" type — the root span of an operation (e.g. "query",
"ingest_document") IS the trace; everything opened while inside its `with`
block becomes a nested child span automatically, via the same mechanism
OpenTelemetry uses (context propagation) — Langfuse's SDK is itself built on
OpenTelemetry, which is the concrete link between "OTel" as a general concept
and "Langfuse" as the specific tool used here.

Boundary this represents: "start a named span, attach input/output/error/usage
to it." Two concrete implementations:
- LangfuseTracer: real backend, used when LANGFUSE_PUBLIC_KEY/SECRET_KEY are
  configured.
- LoggingTracer: default fallback, used otherwise — makes the trace/span
  concept directly visible via standard logging with no external dependency,
  and is what every test in this project uses (tracing must never make tests
  flaky or network-dependent).

This is a genuinely justified abstraction, not decoration: business logic
(ingestion, generation) calls `tracer.span(...)` without knowing or caring
which backend is behind it, which is exactly what let Langfuse be added in
Phase 4 without touching the Tracer *call sites'* reasoning — only the
factory in __init__.py decides which implementation to construct.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any


class SpanHandle(ABC):
    @abstractmethod
    def set_input(self, value: Any) -> None: ...

    @abstractmethod
    def set_output(self, value: Any) -> None: ...

    @abstractmethod
    def set_error(self, message: str) -> None: ...

    @abstractmethod
    def set_usage(
        self, *, input_tokens: int | None = None, output_tokens: int | None = None, model: str | None = None
    ) -> None: ...


class Tracer(ABC):
    @abstractmethod
    def span(
        self, name: str, *, as_type: str = "span", input: Any = None, metadata: dict | None = None
    ) -> AbstractContextManager[SpanHandle]:
        """
        Open a span. Any exception raised inside the `with` block is recorded
        via set_error and then RE-RAISED — a Tracer implementation must never
        swallow a business-logic exception, only observe it. See
        LangfuseTracer for the separate (and more important) guarantee that a
        TRACING-infrastructure failure (Langfuse unreachable) is what gets
        swallowed, never the pipeline's own errors.
        """
        ...
