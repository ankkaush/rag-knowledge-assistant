from app.core.config import settings
from app.providers.llm.base import LLMProvider
from app.providers.llm.ollama_provider import OllamaChatProvider
from app.providers.llm.openai_provider import OpenAIChatProvider
from app.providers.llm.router import LLMRouter

__all__ = ["LLMProvider", "LLMRouter", "get_llm_provider", "get_ollama_provider", "get_llm_router"]


def get_llm_provider() -> LLMProvider:
    """Factory: mirrors app/providers/embeddings/__init__.py."""
    if settings.llm_provider == "openai":
        return OpenAIChatProvider(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.llm_provider == "ollama":
        return get_ollama_provider()
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_ollama_provider() -> OllamaChatProvider:
    return OllamaChatProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        max_retries=settings.ollama_max_retries,
    )


def get_llm_router() -> LLMRouter | None:
    """
    None (disabled) is a normal, fully-supported state, matching the
    reranker/RAGAS pattern elsewhere in this project — the caller falls back
    to a single LLMProvider (see app/generation/service.py) exactly as in
    Phases 2-5.
    """
    if not settings.hybrid_routing_enabled:
        return None
    return LLMRouter(
        api_provider=get_llm_provider(),
        local_provider=get_ollama_provider(),
        allow_api_to_local_fallback=settings.hybrid_allow_api_to_local_fallback,
    )
