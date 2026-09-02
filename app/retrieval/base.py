"""
VectorStore interface.

Boundary: "persist chunks+embeddings; given a query vector, return nearest
chunks." Kept intentionally minimal — two methods, matching exactly what the
ingestion service and (Phase 2) retriever actually need. The concrete
implementation (`PgVectorStore`, in pgvector_store.py) is where the real SQL
lives; this interface exists so that code depending on "a vector store" names
the dependency abstractly, without meaning the pgvector SQL itself is hidden
or unimportant to understand — quite the opposite, read pgvector_store.py.

Realistic future swap this enables: a managed vector DB (Qdrant/Pinecone) if
this project ever outgrew a single Postgres instance. Not needed yet — this
interface is justified today mainly by making the ingestion/retrieval service
code testable against a fake/in-memory store, and by naming the boundary
explicitly for Phase 2's Retriever to depend on.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from app.ingestion.chunking import Chunk
from app.retrieval.types import RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def replace_document_chunks(
        self,
        document_id: UUID,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int: ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_id: UUID | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]: ...
