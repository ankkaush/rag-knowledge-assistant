"""
Unit tests for LangfuseTracer's defensive behavior — never a live network
call, never allowed to break the caller. Constructing a Langfuse client with
dummy keys doesn't itself make a network call (confirmed against the SDK),
so these tests are fully offline; they verify the wrapping/fallback logic in
app/observability/langfuse_tracer.py, not real delivery to Langfuse Cloud
(that's what scripts/langfuse_demo.py is for, run manually with real keys).
"""
from __future__ import annotations

import logging

import pytest

from app.observability.langfuse_tracer import LangfuseTracer


@pytest.fixture
def tracer():
    return LangfuseTracer(public_key="pk-lf-test-dummy", secret_key="sk-lf-test-dummy", host="https://cloud.langfuse.com")


def test_flush_never_raises_even_if_underlying_client_fails(tracer, monkeypatch, caplog):
    def _boom():
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(tracer._client, "flush", _boom)

    with caplog.at_level(logging.WARNING, logger="rag.trace"):
        tracer.flush()  # must not raise

    assert any("langfuse_flush_failed" in r.message for r in caplog.records)


def test_span_start_failure_falls_back_to_noop_handle_and_pipeline_still_runs(tracer, monkeypatch, caplog):
    def _boom(**_kwargs):
        raise RuntimeError("langfuse unreachable")

    monkeypatch.setattr(tracer._client, "start_as_current_observation", _boom)

    ran = False
    with caplog.at_level(logging.WARNING, logger="rag.trace"):
        with tracer.span("query") as span:
            ran = True
            span.set_output({"answer": "still works"})  # no-op handle, must not raise

    assert ran
    assert any("langfuse_span_start_failed" in r.message for r in caplog.records)


def test_pipeline_exception_still_propagates_even_when_langfuse_is_down(tracer, monkeypatch):
    monkeypatch.setattr(
        tracer._client, "start_as_current_observation", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("down"))
    )

    with pytest.raises(ValueError):
        with tracer.span("query"):
            raise ValueError("real business logic error")
