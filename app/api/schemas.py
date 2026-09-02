from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: UUID
    status: str
    chunk_count: int


class RetrievedChunkResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_title: str
    source_uri: str
    chunk_index: int
    page_number: int | None
    content: str
    distance: float
    rerank_score: float | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: UUID | None = None
    doc_type: str | None = None
    rerank: bool = True


class SearchResponse(BaseModel):
    results: list[RetrievedChunkResponse]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: UUID | None = None
    doc_type: str | None = None
    rerank: bool = True


class CitationResponse(BaseModel):
    index: int
    chunk_id: UUID
    document_title: str
    source_uri: str
    page_number: int | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[RetrievedChunkResponse]
    model: str | None
    route: str | None = None  # "api" | "local" | "local-fallback" | None (hybrid routing disabled)


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    trace_id: str | None = None
