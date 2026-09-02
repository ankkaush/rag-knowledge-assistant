import pytest

from app.core.errors import UpstreamUnavailableError
from app.observability.logging_tracer import LoggingTracer
from app.providers.llm.base import GenerationResult, Message
from app.providers.llm.fake_provider import FakeLLMProvider
from app.providers.llm.router import LLMRouter


class _FailingProvider:
    model_name = "failing"

    def generate(self, messages, max_tokens, temperature):
        raise UpstreamUnavailableError(user_message="down", internal_detail="simulated failure")


MESSAGES = [Message(role="user", content="hello [1]")]


def test_non_sensitive_routes_to_api():
    router = LLMRouter(api_provider=FakeLLMProvider(response_text="from-api"), local_provider=FakeLLMProvider(response_text="from-local"))
    result, route = router.generate(MESSAGES, max_tokens=100, temperature=0.2, is_sensitive=False, tracer=LoggingTracer())
    assert route == "api"
    assert result.content == "from-api"


def test_sensitive_routes_to_local():
    router = LLMRouter(api_provider=FakeLLMProvider(response_text="from-api"), local_provider=FakeLLMProvider(response_text="from-local"))
    result, route = router.generate(MESSAGES, max_tokens=100, temperature=0.2, is_sensitive=True, tracer=LoggingTracer())
    assert route == "local"
    assert result.content == "from-local"


def test_sensitive_with_no_local_provider_hard_fails():
    router = LLMRouter(api_provider=FakeLLMProvider(), local_provider=None)
    with pytest.raises(UpstreamUnavailableError):
        router.generate(MESSAGES, max_tokens=100, temperature=0.2, is_sensitive=True, tracer=LoggingTracer())


def test_sensitive_with_local_down_hard_fails_never_falls_back_to_api():
    router = LLMRouter(api_provider=FakeLLMProvider(response_text="from-api"), local_provider=_FailingProvider())
    with pytest.raises(UpstreamUnavailableError):
        router.generate(MESSAGES, max_tokens=100, temperature=0.2, is_sensitive=True, tracer=LoggingTracer())


def test_non_sensitive_with_api_down_falls_back_to_local_by_default():
    router = LLMRouter(api_provider=_FailingProvider(), local_provider=FakeLLMProvider(response_text="from-local"))
    result, route = router.generate(MESSAGES, max_tokens=100, temperature=0.2, is_sensitive=False, tracer=LoggingTracer())
    assert route == "local-fallback"
    assert result.content == "from-local"


def test_non_sensitive_with_api_down_and_fallback_disabled_raises():
    router = LLMRouter(
        api_provider=_FailingProvider(), local_provider=FakeLLMProvider(),
        allow_api_to_local_fallback=False,
    )
    with pytest.raises(UpstreamUnavailableError):
        router.generate(MESSAGES, max_tokens=100, temperature=0.2, is_sensitive=False, tracer=LoggingTracer())


def test_non_sensitive_with_api_down_and_no_local_provider_raises():
    router = LLMRouter(api_provider=_FailingProvider(), local_provider=None)
    with pytest.raises(UpstreamUnavailableError):
        router.generate(MESSAGES, max_tokens=100, temperature=0.2, is_sensitive=False, tracer=LoggingTracer())


def test_route_decision_span_is_traced(caplog):
    import logging

    router = LLMRouter(api_provider=FakeLLMProvider(), local_provider=FakeLLMProvider())
    with caplog.at_level(logging.INFO, logger="rag.trace"):
        router.generate(MESSAGES, max_tokens=100, temperature=0.2, is_sensitive=True, tracer=LoggingTracer())

    messages = [r.message for r in caplog.records]
    assert any("span_start name=route_decision" in m and "'sensitive': True" in m for m in messages)
    assert any("span_end name=route_decision" in m and "'chosen_route': 'local'" in m for m in messages)
