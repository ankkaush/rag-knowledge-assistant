"""
Exercises the REAL OllamaChatProvider against a real local Ollama server —
skipped automatically if Ollama isn't reachable. This is the one place in the
suite that makes a real local-inference call.
"""
from __future__ import annotations

import httpx
import pytest

from app.core.config import settings
from app.providers.llm.base import Message
from app.providers.llm.ollama_provider import OllamaChatProvider


def _ollama_available() -> bool:
    try:
        resp = httpx.get(settings.ollama_base_url.replace("/v1", "/api/version"), timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(), reason=f"Ollama not reachable at {settings.ollama_base_url} — run `ollama serve`"
)


@requires_ollama
def test_ollama_provider_generates_a_real_local_response():
    provider = OllamaChatProvider(
        base_url=settings.ollama_base_url, model=settings.ollama_model,
        timeout_seconds=60, max_retries=1,
    )
    result = provider.generate(
        [Message(role="user", content="Reply with exactly the single word: pong")],
        max_tokens=20, temperature=0.0,
    )
    assert result.content.strip()
    assert result.model == settings.ollama_model
    assert result.input_tokens is not None
    assert result.output_tokens is not None


@requires_ollama
def test_ollama_provider_reports_missing_model_as_permanent_error():
    from app.core.errors import ValidationError

    provider = OllamaChatProvider(
        base_url=settings.ollama_base_url, model="definitely-not-a-real-model:latest",
        timeout_seconds=10, max_retries=1,
    )
    with pytest.raises(ValidationError):
        provider.generate([Message(role="user", content="hi")], max_tokens=10, temperature=0.0)
