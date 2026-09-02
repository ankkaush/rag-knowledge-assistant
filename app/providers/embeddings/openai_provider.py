from __future__ import annotations

import openai

from app.core.errors import UpstreamUnavailableError, ValidationError
from app.core.retry import openai_retry
from app.providers.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
        max_retries: int,
    ):
        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries
        # timeout is passed to the SDK client itself, so it bounds every call
        # made through this client — not something each call site has to remember.
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        @openai_retry(self._max_retries)
        def _call() -> list[list[float]]:
            response = self._client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in response.data]

        try:
            return _call()
        except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError) as exc:
            # Retries (if any were transient) are already exhausted by the time we get here.
            raise UpstreamUnavailableError(
                user_message="Embedding service is currently unavailable. Please try again shortly.",
                internal_detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        except openai.APIStatusError as exc:
            # Non-5xx status errors (e.g. 400 invalid request, 401 auth) are permanent — no retry happened.
            raise ValidationError(
                user_message="The embedding request was rejected (invalid input or configuration).",
                internal_detail=f"OpenAI APIStatusError {exc.status_code}: {exc}",
            ) from exc
