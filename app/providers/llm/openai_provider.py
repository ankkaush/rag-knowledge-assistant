from __future__ import annotations

import openai

from app.core.errors import UpstreamUnavailableError, ValidationError
from app.core.retry import openai_retry
from app.providers.llm.base import GenerationResult, LLMProvider, Message


class OpenAIChatProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, timeout_seconds: float, max_retries: int):
        self._model = model
        self._max_retries = max_retries
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: list[Message], max_tokens: int, temperature: float) -> GenerationResult:
        @openai_retry(self._max_retries)
        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        try:
            response = _call()
        except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError) as exc:
            raise UpstreamUnavailableError(
                user_message="The language model is currently unavailable. Please try again shortly.",
                internal_detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        except openai.APIStatusError as exc:
            raise ValidationError(
                user_message="The generation request was rejected (invalid input or configuration).",
                internal_detail=f"OpenAI APIStatusError {exc.status_code}: {exc}",
            ) from exc

        choice = response.choices[0]
        usage = response.usage
        return GenerationResult(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
