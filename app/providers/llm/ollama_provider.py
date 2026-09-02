"""
OllamaChatProvider: local inference via Ollama.

Ollama runs models entirely on this machine (CPU/GPU, no network egress) and
exposes an OpenAI-compatible endpoint at `/v1/chat/completions` — so this
provider reuses the `openai` SDK client, just pointed at Ollama's local
base_url with a throwaway API key (Ollama doesn't check it). This is
deliberate: it's the same client shape as OpenAIChatProvider, which is what
makes both providers substitutable behind the same LLMProvider interface
without any Ollama-specific request/response translation code.

WHAT "LOCAL" ACTUALLY BUYS HERE, CONCRETELY (not just in the abstract):
- Privacy: the prompt (which may contain sensitive document content, per
  app/providers/llm/router.py) never leaves this machine — no third-party
  API call is made at all.
- Cost: no per-token charge; the only cost is local compute/electricity.
- Trade-off paid for both: a small local model (this project defaults to
  `llama3.2:1b`, ~1.3GB) is meaningfully lower-quality than a frontier hosted
  model, and inference is CPU/GPU-bound by this machine rather than elastic
  cloud capacity. This is the real trade-off named in the project plan's
  hybrid-inference section — not a hypothetical.

ERROR CLASSIFICATION — one genuinely different case from OpenAI:
Ollama returns 404 when the requested model hasn't been pulled locally
(`ollama pull <model>`) rather than "invalid request" in the OpenAI sense.
This is still a PERMANENT error (retrying without pulling the model changes
nothing), but the internal detail below says so explicitly rather than
reusing OpenAI's generic 400/401 language, since the fix is different
("pull the model") from an OpenAI 400 ("fix your request").
"""
from __future__ import annotations

import openai

from app.core.errors import UpstreamUnavailableError, ValidationError
from app.core.retry import openai_retry
from app.providers.llm.base import GenerationResult, LLMProvider, Message


class OllamaChatProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: float, max_retries: int):
        self._model = model
        self._max_retries = max_retries
        # Ollama ignores the API key entirely; the SDK just requires a non-empty string.
        self._client = openai.OpenAI(base_url=base_url, api_key="ollama", timeout=timeout_seconds)

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
            # Most commonly: the Ollama server isn't running at all (`ollama serve`).
            raise UpstreamUnavailableError(
                user_message="The local language model is currently unavailable. Please try again shortly.",
                internal_detail=f"{type(exc).__name__}: {exc} (is `ollama serve` running at the configured base_url?)",
            ) from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 404:
                raise ValidationError(
                    user_message="The requested local model is not available.",
                    internal_detail=f"Ollama 404 for model={self._model!r} — has it been pulled? Run: ollama pull {self._model}",
                ) from exc
            raise ValidationError(
                user_message="The generation request was rejected (invalid input or configuration).",
                internal_detail=f"Ollama APIStatusError {exc.status_code}: {exc}",
            ) from exc

        choice = response.choices[0]
        usage = response.usage
        return GenerationResult(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
