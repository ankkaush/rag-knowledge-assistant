"""
CrossEncoderReranker: local, open-weight reranker via Hugging Face
`sentence-transformers`.

WHY LOCAL RATHER THAN AN API RERANK SERVICE (e.g. Cohere Rerank):
No additional API key, no per-call cost, no extra network round-trip, and it
runs fast enough on CPU for a small candidate set (the whole point of the
two-stage pattern in reranker.py — reranking only ever runs over ~10-50
candidates, not the full corpus). This is also a deliberate, low-risk place
to introduce the Hugging Face ecosystem (model hub, `transformers`-family
local inference) ahead of Phase 6's hybrid/local-inference topic — reranking
carries none of Phase 6's privacy/routing concerns, since it never generates
text, only scores existing chunks.

MODEL: `cross-encoder/ms-marco-MiniLM-L-6-v2` — a small (~80MB), widely used
cross-encoder fine-tuned on the MS MARCO passage-ranking dataset. It outputs
an unbounded relevance logit per (query, passage) pair, not a probability —
we only use it for RELATIVE ordering (sort descending), never as an absolute
confidence score.

This is imported lazily inside __init__, not at module load time: importing
`sentence_transformers` pulls in `torch`, a heavy dependency (see the
`rerank` extra in pyproject.toml) that most of this codebase — including the
eval harness with reranking OFF — never needs to pay for.
"""
from __future__ import annotations

import dataclasses

from app.retrieval.reranker import Reranker
from app.retrieval.types import RetrievedChunk

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.content) for c in candidates]
        scores = self._model.predict(pairs)

        scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [
            dataclasses.replace(chunk, rerank_score=float(score))
            for chunk, score in scored[:top_k]
        ]
