"""
Raw similarity-search endpoint.

This is intentionally NOT the full RAG query endpoint (no LLM call, no context
construction, no citations synthesis — that's /query, in app/api/query.py).
Its purpose is to prove, end-to-end and inspectably, that "a question embeds
to a vector that finds the right chunks" — the retrieval half of the pipeline
— independent of and cheaper than a full generation call. Useful for
debugging retrieval quality (and, since Phase 5, reranking) in isolation from
generation quality.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import RetrievedChunkResponse, SearchRequest, SearchResponse
from app.core.config import settings
from app.core.db import SessionLocal
from app.observability import get_tracer
from app.providers.embeddings import get_embedding_provider
from app.retrieval import get_reranker
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.retriever import Retriever

router = APIRouter(prefix="/search", tags=["retrieval"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    session = SessionLocal()
    try:
        retriever = Retriever(
            vector_store=PgVectorStore(session),
            embedding_provider=get_embedding_provider(),
            default_top_k=settings.default_top_k,
            reranker=get_reranker(),
            rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
        )
        tracer = get_tracer()
        with tracer.span(
            "retrieve", as_type="retriever",
            input={"query": request.query, "top_k": request.top_k},
            metadata={"use_reranker": request.rerank},
        ) as span:
            results = retriever.retrieve(
                request.query, top_k=request.top_k, document_id=request.document_id,
                doc_type=request.doc_type, use_reranker=request.rerank,
            )
            span.set_output({"retrieved_count": len(results), "source_uris": [r.source_uri for r in results]})
    finally:
        session.close()

    return SearchResponse(
        results=[
            RetrievedChunkResponse(
                chunk_id=r.id,
                document_id=r.document_id,
                document_title=r.document_title,
                source_uri=r.source_uri,
                chunk_index=r.chunk_index,
                page_number=r.page_number,
                content=r.content,
                distance=r.distance,
                rerank_score=r.rerank_score,
            )
            for r in results
        ]
    )
