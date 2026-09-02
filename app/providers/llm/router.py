"""
LLMRouter: decides which LLMProvider serves a generation call.

THE POLICY (agreed in the architecture review before Phase 1 began, §13):
- Non-sensitive request -> API provider (best available quality/latency).
- Sensitive request -> local provider (nothing leaves the machine).
- NO automatic sensitive-request fallback to the API provider, ever. If the
  local provider is unavailable for a sensitive request, this raises rather
  than silently downgrading privacy — a hard failure is the only acceptable
  outcome, because the entire point of the sensitive flag is that this
  request's content must not reach a third-party API. Silently rerouting
  would defeat the feature while looking like it worked.
- A non-sensitive request MAY fall back to the local provider if the API
  provider is unavailable (`hybrid_allow_api_to_local_fallback`, default
  True) — there's no privacy reason not to, and it's a genuine resilience
  win from having a local model available at all. This is a deliberate
  choice, not the only valid one: failing outright instead would also be
  defensible. Falling back to local (rather than just failing) is what's
  implemented here.

WHY THIS IS A SEPARATE OBJECT FROM LLMProvider, NOT A THIRD IMPLEMENTATION
OF IT: `is_sensitive` is not a property of a single generation call in the
LLMProvider sense (messages/max_tokens/temperature) — it's retrieval-layer
context about WHERE the content came from. Folding routing into LLMProvider
would mean every provider implementation needs to know about sensitivity,
metadata flags, and fallback policy, none of which is its concern. Keeping
Router separate means OpenAIChatProvider and OllamaChatProvider stay exactly
as simple as they'd be with no routing at all.

TRACING: every routing decision opens its own "route_decision" span (see
app/generation/service.py for where this sits relative to "retrieve" and
"generate") recording which path was chosen and, on fallback, why.
"""
from __future__ import annotations

from app.core.errors import UpstreamUnavailableError
from app.observability.base import Tracer
from app.providers.llm.base import GenerationResult, LLMProvider, Message


class LLMRouter:
    def __init__(
        self,
        api_provider: LLMProvider,
        local_provider: LLMProvider | None,
        allow_api_to_local_fallback: bool = True,
    ):
        self._api_provider = api_provider
        self._local_provider = local_provider
        self._allow_fallback = allow_api_to_local_fallback

    def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        *,
        is_sensitive: bool,
        tracer: Tracer,
    ) -> tuple[GenerationResult, str]:
        """Returns (result, route) where route is one of: "api", "local", "local-fallback"."""
        with tracer.span("route_decision", input={"sensitive": is_sensitive}) as span:
            chosen = "local" if is_sensitive else "api"
            span.set_output({"chosen_route": chosen, "local_provider_configured": self._local_provider is not None})

        if chosen == "local":
            if self._local_provider is None:
                raise UpstreamUnavailableError(
                    user_message="This request requires local, private inference, which is not currently configured.",
                    internal_detail="LLMRouter: sensitive request routed to local, but no local_provider is configured",
                )
            # No except here: an UpstreamUnavailableError from the local
            # provider propagates straight up. This is the hard-fail rule —
            # there is no fallback path for a sensitive request, on purpose.
            result = self._local_provider.generate(messages, max_tokens=max_tokens, temperature=temperature)
            return result, "local"

        # chosen == "api"
        try:
            result = self._api_provider.generate(messages, max_tokens=max_tokens, temperature=temperature)
            return result, "api"
        except UpstreamUnavailableError:
            if not self._allow_fallback or self._local_provider is None:
                raise
            result = self._local_provider.generate(messages, max_tokens=max_tokens, temperature=temperature)
            return result, "local-fallback"
