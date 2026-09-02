"""
LLMProvider interface.

Boundary: "given a system prompt and a conversation, produce a completion" —
plus the token accounting the provider's response reports, since that's
needed for cost/latency observability starting in Phase 4 and is naturally
available at this exact call site (retrofitting it later would mean touching
every call site again).

Why this earns an interface, concretely, not speculatively: Phase 6 (hybrid
inference) requires calling either an API model or a local Ollama model
through the SAME call site in the query service, decided by a routing rule —
that's not possible unless generation is already expressed against an
abstract interface today. This is the direct enabler named in the approved
plan for the hybrid-routing boundary.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class GenerationResult:
    content: str
    model: str
    input_tokens: int | None
    output_tokens: int | None


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def generate(self, messages: list[Message], max_tokens: int, temperature: float) -> GenerationResult:
        """
        Raises UpstreamUnavailableError if the provider is unreachable after
        retries, or ValidationError for a permanent/input error (e.g. prompt
        exceeds the model's context window).
        """
        ...
