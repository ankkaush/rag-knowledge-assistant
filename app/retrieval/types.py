from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class RetrievedChunk:
    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    page_number: int | None
    section_title: str | None
    distance: float
    document_title: str
    source_uri: str
    # Reused from documents.metadata->>'sensitive' (see db/migrations/001_init.sql
    # and app/ingestion/service.py). Not routing logic (that's Phase 6) — Phase 4
    # uses this as the single source of truth for trace redaction: a query that
    # retrieves from ANY sensitive document gets its prompt/context redacted in
    # Langfuse. See app/generation/service.py.
    is_sensitive: bool
    # Set only when a Reranker has reordered this chunk (see
    # app/retrieval/reranker.py). None means "not reranked" — distinguishable
    # from a real score of 0.0, which matters for reporting/debugging whether
    # reranking was even active for a given result.
    rerank_score: float | None = None
