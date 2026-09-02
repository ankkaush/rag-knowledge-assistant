"""
Deterministic fake embedding provider for tests.

Purpose: unit and integration tests for chunking/ingestion/retrieval logic
should not depend on network access or an OpenAI API key. This provider maps
text to a vector deterministically (via a hash of the text) so the SAME input
always produces the SAME vector, and semantically-similar test strings can be
constructed to produce closer vectors than dissimilar ones — enough to
exercise real similarity search behavior without a real model.

This is explicitly a test double, not a "cheap embedding option" for real use.
"""
from __future__ import annotations

import hashlib
import math

from app.providers.embeddings.base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 1536):
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return "fake-embedding-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        # Bag-of-words hashing: each word deterministically increments a fixed
        # set of vector positions. This means texts sharing words produce
        # vectors with nonzero cosine similarity, while unrelated texts don't —
        # good enough to exercise "does the most similar chunk come back first."
        vec = [0.0] * self._dimensions
        for word in text.lower().split():
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            idx = h % self._dimensions
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
