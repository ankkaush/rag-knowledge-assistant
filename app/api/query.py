"""
The full RAG query endpoint: question -> retrieved chunks -> grounded, cited answer.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import CitationResponse, QueryRequest, QueryResponse, RetrievedChunkResponse
from app.core.config import settings
from app.core.db import SessionLocal
from app.generation.service import answer_query
from app.observability import get_tracer
from app.providers.embeddings import get_embedding_provider
from app.providers.llm import get_llm_provider, get_llm_router
from app.retrieval import get_reranker
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.retriever import Retriever

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    session = SessionLocal()
    try:
        retriever = Retriever(
            vector_store=PgVectorStore(session),
            embedding_provider=get_embedding_provider(),
            default_top_k=settings.default_top_k,
            reranker=get_reranker(),
            rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
        )
        router = get_llm_router()
        result = answer_query(
            retriever=retriever,
            llm_provider=None if router else get_llm_provider(),
            router=router,
            tracer=get_tracer(),
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            doc_type=request.doc_type,
            use_reranker=request.rerank,
            context_max_tokens=settings.context_max_tokens,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    finally:
        session.close()

    return QueryResponse(
        answer=result.answer,
        model=result.model,
        route=result.route,
        citations=[
            CitationResponse(
                index=c.index,
                chunk_id=c.chunk_id,
                document_title=c.document_title,
                source_uri=c.source_uri,
                page_number=c.page_number,
            )
            for c in result.citations
        ],
        retrieved_chunks=[
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
            for r in result.retrieved_chunks
        ],
    )
