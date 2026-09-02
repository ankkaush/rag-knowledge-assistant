"""
Deterministic fake reranker for tests — no torch/model download required.

Scores each candidate by word-overlap with the query (case-insensitive token
intersection count) — crude, but genuinely rearranges order based on
(query, content), which is what tests need to verify Retriever actually
calls the reranker and uses its ordering, without depending on a real model.
"""
from __future__ import annotations

import dataclasses

from app.retrieval.reranker import Reranker
from app.retrieval.types import RetrievedChunk


class FakeReranker(Reranker):
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        query_words = set(query.lower().split())

        def score(chunk: RetrievedChunk) -> float:
            content_words = set(chunk.content.lower().split())
            return float(len(query_words & content_words))

        scored = sorted(candidates, key=score, reverse=True)
        return [dataclasses.replace(c, rerank_score=score(c)) for c in scored[:top_k]]
