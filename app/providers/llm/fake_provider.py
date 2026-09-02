"""
Deterministic fake LLM for tests — no network/API key required.

Default behavior: scans the user message for context markers of the form
"[N]" (the citation-block format produced by app/generation/context_builder.py)
and returns a canned answer that cites the HIGHEST such index found — enough
to exercise citation-extraction and citation-validation logic against a real,
predictable response, without depending on real model output (which would
make tests non-deterministic and network-dependent).

A fixed `response_text` can be passed to test specific scenarios, e.g. an
answer that cites an out-of-range index (to test invalid-citation handling)
or contains no citations at all.
"""
from __future__ import annotations

import re

from app.providers.llm.base import GenerationResult, LLMProvider, Message

_CONTEXT_MARKER = re.compile(r"\[(\d+)\]")


class FakeLLMProvider(LLMProvider):
    def __init__(self, response_text: str | None = None):
        self._fixed_response = response_text

    @property
    def model_name(self) -> str:
        return "fake-llm-v1"

    def generate(self, messages: list[Message], max_tokens: int, temperature: float) -> GenerationResult:
        if self._fixed_response is not None:
            content = self._fixed_response
        else:
            user_text = "\n".join(m.content for m in messages if m.role == "user")
            indices = [int(n) for n in _CONTEXT_MARKER.findall(user_text)]
            highest = max(indices) if indices else None
            content = (
                f"Based on the provided context, here is a grounded answer. [{highest}]"
                if highest
                else "I don't have enough information in the provided documents to answer this question."
            )

        return GenerationResult(
            content=content,
            model=self.model_name,
            input_tokens=sum(len(m.content) for m in messages) // 4,
            output_tokens=len(content) // 4,
        )
