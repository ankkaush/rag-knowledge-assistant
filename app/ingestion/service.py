"""
Ingestion service: orchestrates load -> clean -> chunk -> embed -> store.

IDEMPOTENCY DESIGN:
`content_hash` (sha256 of the raw uploaded bytes) is a UNIQUE column on
`documents` (see db/migrations/001_init.sql). Re-uploading byte-identical
content therefore hits the same document row rather than creating a
duplicate. Re-ingesting a document whose row already exists deletes and
replaces its chunks (`VectorStore.replace_document_chunks`) inside the same
transaction as updating `documents.status` — so a retry or duplicate upload
converges to the same end state instead of accumulating duplicate chunks.

Note what this does NOT do: it does not detect *near*-duplicate content
(same document re-exported to a slightly different PDF, whitespace changes).
Byte-identical idempotency is the deliberately narrow, correct-by-construction
guarantee for Phase 1; content-similarity dedup is a different, harder problem
out of scope here.

TRANSACTION BOUNDARIES (three, deliberately separate):
1. Create-or-reset the document row to 'processing' — committed immediately,
   so the row exists and is visible even if everything after it fails.
2. Load/clean/chunk/embed — no DB writes; if any step raises, nothing in the
   database has changed yet except step 1's 'processing' status.
3. Replace chunks + mark 'ready' — one transaction. If it fails partway
   (e.g. mid-insert), it rolls back completely: chunks from any *previous*
   successful ingestion of this document are left untouched rather than
   half-overwritten. The document is then marked 'failed' in its own small
   transaction so the failure is recorded and visible.
This means a failed re-ingestion never corrupts an existing 'ready' document —
worst case, it stays on its last-good chunk set and status flips to 'failed'.

SENSITIVITY FLAG (Phase 4 adds it; Phase 6 will act on it):
`sensitive` is stored in the existing `documents.metadata` JSONB column
(no schema migration needed). Phase 4 uses it ONLY to decide what gets
redacted from Langfuse traces (see app/generation/service.py) — no routing
behavior exists yet. Introducing the flag now, with only its observability
half wired up, is the boundary the approved plan drew deliberately: the same
flag will drive Phase 6's local-vs-API routing decision later without a
second flag or a schema change.

TRACING: each step below is its own span, nested under one "ingest_document"
root span (nesting is automatic — see app/observability/base.py). If a step
raises, the exception propagates through both the span (which records it as
an error span) and the existing except block below (which marks the document
row 'failed') — the two concerns don't interfere with each other.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import AppError, IngestionError
from app.ingestion.chunking import Chunk, chunk_text
from app.ingestion.cleaning import clean_text, is_meaningful
from app.ingestion.loaders import get_loader
from app.observability.base import Tracer
from app.providers.embeddings.base import EmbeddingProvider
from app.retrieval.base import VectorStore


@dataclass(frozen=True)
class IngestResult:
    document_id: UUID
    status: str
    chunk_count: int


def ingest_document(
    session: Session,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    tracer: Tracer,
    *,
    title: str,
    source_uri: str,
    doc_type: str,
    raw_bytes: bytes,
    sensitive: bool = False,
) -> IngestResult:
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    document_id = _upsert_document_pending(
        session, title=title, source_uri=source_uri, doc_type=doc_type,
        content_hash=content_hash, sensitive=sensitive,
    )

    with tracer.span(
        "ingest_document",
        input={"title": title, "source_uri": source_uri, "doc_type": doc_type, "sensitive": sensitive},
    ) as root:
        try:
            with tracer.span("extract", input={"doc_type": doc_type}) as span:
                loader = get_loader(doc_type)
                pages = loader.load(raw_bytes)
                span.set_output({"page_count": len(pages)})

            with tracer.span("clean_and_chunk") as span:
                all_chunks: list[Chunk] = []
                next_index = 0
                for page in pages:
                    cleaned = clean_text(page.text)
                    if not is_meaningful(cleaned):
                        continue
                    page_chunks = chunk_text(
                        cleaned,
                        chunk_size_chars=settings.chunk_size_chars,
                        chunk_overlap_chars=settings.chunk_overlap_chars,
                        page_number=page.page_number,
                    )
                    for c in page_chunks:
                        all_chunks.append(
                            Chunk(
                                chunk_index=next_index,
                                content=c.content,
                                char_start=c.char_start,
                                char_end=c.char_end,
                                token_count=c.token_count,
                                page_number=c.page_number,
                                section_title=c.section_title,
                            )
                        )
                        next_index += 1

                if not all_chunks:
                    raise IngestionError(
                        user_message="No extractable text content found in this document.",
                        internal_detail=f"0 meaningful chunks after cleaning for document_id={document_id}",
                    )
                span.set_output({"chunk_count": len(all_chunks)})

            with tracer.span(
                "embed_batch", as_type="embedding",
                input={"chunk_count": len(all_chunks), "model": embedding_provider.model_name},
            ) as span:
                embeddings = embedding_provider.embed_batch([c.content for c in all_chunks])
                span.set_output({"embedded_count": len(embeddings)})

            with tracer.span("store", input={"chunk_count": len(all_chunks)}) as span:
                vector_store.replace_document_chunks(document_id, all_chunks, embeddings)
                _mark_document_ready(
                    session, document_id, chunk_count=len(all_chunks), embedding_model=embedding_provider.model_name,
                )
                session.commit()
                span.set_output({"stored": True})

            root.set_output({"document_id": str(document_id), "status": "ready", "chunk_count": len(all_chunks)})
            return IngestResult(document_id=document_id, status="ready", chunk_count=len(all_chunks))

        except Exception as exc:
            session.rollback()
            detail = exc.internal_detail if isinstance(exc, AppError) else f"{type(exc).__name__}: {exc}"
            _mark_document_failed(session, document_id, error_detail=detail)
            session.commit()
            raise


def _upsert_document_pending(
    session: Session, *, title: str, source_uri: str, doc_type: str, content_hash: str, sensitive: bool
) -> UUID:
    """
    Insert a new document row, or if this exact content was already ingested
    (content_hash collision), reuse its id and reset it to 'processing' so the
    re-ingestion below replaces its chunks. This is the idempotency entry point.
    """
    row = session.execute(
        text("SELECT id FROM documents WHERE content_hash = :content_hash"),
        {"content_hash": content_hash},
    ).first()

    if row is not None:
        document_id = row.id
        session.execute(
            text(
                """
                UPDATE documents
                SET status = 'processing', metadata = :metadata, updated_at = now()
                WHERE id = :id
                """
            ),
            {"id": str(document_id), "metadata": json.dumps({"sensitive": sensitive})},
        )
        session.commit()
        return document_id

    row = session.execute(
        text(
            """
            INSERT INTO documents (title, source_uri, doc_type, status, content_hash,
                                    chunk_size_chars, chunk_overlap_chars, metadata)
            VALUES (:title, :source_uri, :doc_type, 'processing', :content_hash,
                    :chunk_size_chars, :chunk_overlap_chars, :metadata)
            RETURNING id
            """
        ),
        {
            "title": title,
            "source_uri": source_uri,
            "doc_type": doc_type,
            "content_hash": content_hash,
            "chunk_size_chars": settings.chunk_size_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "metadata": json.dumps({"sensitive": sensitive}),
        },
    ).first()
    session.commit()
    return row.id


def _mark_document_ready(session: Session, document_id: UUID, *, chunk_count: int, embedding_model: str) -> None:
    session.execute(
        text(
            """
            UPDATE documents
            SET status = 'ready', embedding_model = :embedding_model,
                error_detail = NULL, updated_at = now()
            WHERE id = :id
            """
        ),
        {"id": str(document_id), "embedding_model": embedding_model},
    )


def _mark_document_failed(session: Session, document_id: UUID, *, error_detail: str) -> None:
    session.execute(
        text(
            "UPDATE documents SET status = 'failed', error_detail = :detail, updated_at = now() WHERE id = :id"
        ),
        {"id": str(document_id), "detail": error_detail[:2000]},
    )
