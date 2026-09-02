import logging

import pytest

from app.observability.logging_tracer import LoggingTracer


def test_span_records_output(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.INFO, logger="rag.trace"):
        with tracer.span("do_thing", input={"x": 1}) as span:
            span.set_output({"y": 2})

    messages = [r.message for r in caplog.records]
    assert any("span_start name=do_thing" in m for m in messages)
    assert any("span_end name=do_thing" in m and "status=ok" in m and "{'y': 2}" in m for m in messages)


def test_span_records_error_and_reraises(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.INFO, logger="rag.trace"):
        with pytest.raises(ValueError):
            with tracer.span("do_thing") as span:
                raise ValueError("boom")

    messages = [r.message for r in caplog.records]
    assert any("status=error" in m and "boom" in m for m in messages)


def test_nested_spans_both_recorded(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.INFO, logger="rag.trace"):
        with tracer.span("outer") as outer:
            with tracer.span("inner") as inner:
                inner.set_output("inner done")
            outer.set_output("outer done")

    messages = [r.message for r in caplog.records]
    assert any("span_start name=outer" in m for m in messages)
    assert any("span_start name=inner" in m for m in messages)
    assert any("span_end name=inner" in m and "status=ok" in m for m in messages)
    assert any("span_end name=outer" in m and "status=ok" in m for m in messages)


def test_set_usage_appears_in_end_log(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.INFO, logger="rag.trace"):
        with tracer.span("generate", as_type="generation") as span:
            span.set_usage(input_tokens=10, output_tokens=5, model="gpt-4o-mini")

    messages = [r.message for r in caplog.records]
    assert any("gpt-4o-mini" in m and "status=ok" in m for m in messages)
