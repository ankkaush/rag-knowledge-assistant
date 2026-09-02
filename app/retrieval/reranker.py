"""
Reranker interface.

Boundary: "given a query and a set of candidate chunks, return the top-K
reordered by actual relevance to the query." Why this is a genuinely
different operation from vector similarity search, not a redundant second
pass of the same thing:

- Vector search (Retriever/VectorStore) compares the QUERY's embedding to
  each CHUNK's embedding independently — it never looks at the query and a
  candidate chunk together. This is what makes it fast enough to search over
  many chunks (approximate nearest-neighbor at scale), but it's a coarser
  signal: two texts can embed close together as a very rough proxy for "on
  the same topic" without one actually answering the other.
- A cross-encoder reranker takes the (query, chunk) PAIR as joint input to
  one model and outputs a single relevance score for that pair. This is
  strictly more accurate (it can directly judge "does this chunk answer this
  question") but far more expensive per comparison — you cannot run a
  cross-encoder over an entire corpus, only over a small candidate set.

This is why reranking is a two-stage pattern, not a replacement for vector
search: retrieve a wider candidate set cheaply (vector search), then rerank
that small set expensively but accurately (cross-encoder), then keep only
the top-K after reranking. See Retriever.retrieve in app/retrieval/retriever.py
for where the candidate-set widening happens.

WHETHER THIS ACTUALLY HELPS IS AN EMPIRICAL QUESTION, not a given — that's
exactly why Phase 5 is sequenced after Phase 3's evaluation harness exists:
`evaluation/run_eval.py --reranker compare` measures retrieval-hit-rate and
precision@k with reranking on vs. off against the SAME questions, rather than
assuming a cross-encoder must be better because it's fancier.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.retrieval.types import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Returns at most top_k chunks from `candidates`, reordered by relevance to `query`."""
        ...
