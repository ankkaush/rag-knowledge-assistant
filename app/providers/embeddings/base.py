"""
EmbeddingProvider interface.

Boundary: "given text, return a fixed-length vector of floats." Why this
earns an interface (not just a bare function call to OpenAI): the same
pipeline code (chunking → embed → store) should not need to change if the
embedding backend changes from OpenAI to a local Hugging Face model — only
the concrete implementation swapped in changes. This also makes tests
possible without hitting a real API: a `FakeEmbeddingProvider` implements the
same interface deterministically.

`dimensions` is exposed because the vector column width in Postgres
(`vector(1536)`) must match whatever the concrete provider actually returns —
if you swap embedding models with a different output dimensionality, that's a
schema-affecting change, not a drop-in swap. Worth understanding: this is why
`documents.embedding_model` is stored per-document (see db/migrations/001_init.sql) —
mixing vectors from two differently-dimensioned or differently-trained models
in one similarity search is meaningless, even if the schema happened to allow it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts, preserving order (result[i] corresponds to texts[i]).
        Raises UpstreamUnavailableError if the provider is unreachable after retries,
        or ValidationError if a permanent/input error occurs (e.g. text too long).
        """
        ...
