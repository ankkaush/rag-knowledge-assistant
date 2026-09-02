"""
Retriever: composes EmbeddingProvider + VectorStore (+ optional Reranker)
into "question -> ranked chunks".

This is the object named in the approved plan as the unit you'd swap when
comparing retrieval strategies during evaluation (Phase 3) — e.g. a later
hybrid (lexical+vector) strategy would implement the same call signature. In
Phase 1, `/search` embedded the query and called the vector store inline,
duplicating that logic; this class is that logic pulled into one place now
that a second caller (the Phase 2 query pipeline) needs it too.

RERANKING (Phase 5) — TWO-STAGE RETRIEVAL:
When a `reranker` is configured and active for a call, `retrieve()` widens
the vector-search candidate set (`top_k * rerank_candidate_multiplier`,
capped) BEFORE reranking, then narrows back down to `top_k` after reranking.
This is the standard two-stage pattern explained in app/retrieval/reranker.py:
vector search is cheap but coarse, so it's used to cast a wide net; the
cross-encoder is accurate but too expensive to run over the whole corpus, so
it only ever sees the narrowed candidate set.

`use_reranker` is a per-call override (default True when a reranker is
configured) specifically so the SAME Retriever/config can be compared with
reranking on vs. off — this is what
`evaluation/run_eval.py --reranker compare` uses to produce the before/after
comparison the approved plan's Phase 5 Definition of Done requires.
"""
from __future__ import annotations

from uuid import UUID

from app.providers.embeddings.base import EmbeddingProvider
from app.retrieval.base import VectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.types import RetrievedChunk

_MAX_CANDIDATES = 50


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        default_top_k: int = 5,
        reranker: Reranker | None = None,
        rerank_candidate_multiplier: int = 4,
    ):
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._default_top_k = default_top_k
        self._reranker = reranker
        self._rerank_candidate_multiplier = rerank_candidate_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: UUID | None = None,
        doc_type: str | None = None,
        use_reranker: bool = True,
    ) -> list[RetrievedChunk]:
        k = top_k or self._default_top_k
        reranker_active = use_reranker and self._reranker is not None
        candidate_k = min(k * self._rerank_candidate_multiplier, _MAX_CANDIDATES) if reranker_active else k

        query_embedding = self._embedding_provider.embed_batch([query])[0]
        candidates = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            document_id=document_id,
            doc_type=doc_type,
        )

        if reranker_active:
            return self._reranker.rerank(query, candidates, top_k=k)
        return candidates[:k]
