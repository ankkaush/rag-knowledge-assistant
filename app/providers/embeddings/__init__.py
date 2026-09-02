from app.core.config import settings
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

__all__ = ["EmbeddingProvider", "get_embedding_provider"]


def get_embedding_provider() -> EmbeddingProvider:
    """Factory: reads settings and constructs the configured provider.

    Only one branch exists today (openai); the if/else exists so a second
    provider (e.g. a local HF model, added when hybrid inference lands in
    Phase 6) has an obvious place to plug in without touching call sites.
    """
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
