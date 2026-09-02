from sqlalchemy import text

from app.core.errors import IngestionError, ValidationError
from app.ingestion.service import ingest_document
from tests.integration.conftest import requires_db


@requires_db
def test_ingest_txt_document_end_to_end(db_session, vector_store, fake_embeddings, tracer):
    raw = b"The quick brown fox jumps over the lazy dog. " * 40  # long enough for multiple chunks
    result = ingest_document(
        session=db_session,
        vector_store=vector_store,
        embedding_provider=fake_embeddings,
        tracer=tracer,
        title="Fox Doc",
        source_uri="fox.txt",
        doc_type="txt",
        raw_bytes=raw,
    )

    assert result.status == "ready"
    assert result.chunk_count > 0

    row = db_session.execute(
        text("SELECT status, embedding_model FROM documents WHERE id = :id"), {"id": str(result.document_id)}
    ).first()
    assert row.status == "ready"
    assert row.embedding_model == fake_embeddings.model_name

    chunk_count = db_session.execute(
        text("SELECT count(*) FROM chunks WHERE document_id = :id"), {"id": str(result.document_id)}
    ).scalar()
    assert chunk_count == result.chunk_count


@requires_db
def test_reingesting_identical_bytes_is_idempotent(db_session, vector_store, fake_embeddings, tracer):
    raw = b"Identical content for idempotency testing. " * 20

    first = ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Doc", source_uri="doc.txt", doc_type="txt", raw_bytes=raw,
    )
    second = ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Doc", source_uri="doc.txt", doc_type="txt", raw_bytes=raw,
    )

    assert first.document_id == second.document_id  # same content_hash -> same row, not a duplicate

    doc_count = db_session.execute(text("SELECT count(*) FROM documents")).scalar()
    assert doc_count == 1


@requires_db
def test_ingesting_empty_meaningful_content_fails_cleanly(db_session, vector_store, fake_embeddings, tracer):
    # Whitespace-only content clears extraction but has nothing chunk-worthy.
    raw = b"   \n\n   "
    try:
        ingest_document(
            session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
            title="Empty", source_uri="empty.txt", doc_type="txt", raw_bytes=raw,
        )
        assert False, "expected IngestionError"
    except IngestionError:
        pass

    row = db_session.execute(
        text("SELECT status, error_detail FROM documents WHERE source_uri = 'empty.txt'")
    ).first()
    assert row.status == "failed"
    assert row.error_detail is not None


@requires_db
def test_unsupported_doc_type_raises_validation_error(db_session, vector_store, fake_embeddings, tracer):
    try:
        ingest_document(
            session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
            title="Bad", source_uri="bad.docx", doc_type="docx", raw_bytes=b"whatever",
        )
        assert False, "expected ValidationError"
    except ValidationError:
        pass


@requires_db
def test_failed_reingestion_does_not_destroy_previous_ready_chunks(db_session, vector_store, fake_embeddings, tracer):
    good_raw = b"This document has real content that will chunk successfully. " * 10
    good = ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Doc", source_uri="doc.txt", doc_type="txt", raw_bytes=good_raw,
    )
    assert good.status == "ready"
    good_chunk_count = db_session.execute(
        text("SELECT count(*) FROM chunks WHERE document_id = :id"), {"id": str(good.document_id)}
    ).scalar()
    assert good_chunk_count > 0

    # A second call with the SAME source_uri but different (empty) content
    # simulates a bad re-ingestion. Since content_hash differs, this actually
    # creates a NEW document row rather than colliding with the good one —
    # which is exactly the point: this test documents that a failed ingestion
    # of *different* content never touches a *different*, already-ready document's chunks.
    try:
        ingest_document(
            session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
            title="Doc", source_uri="doc.txt", doc_type="txt", raw_bytes=b"   ",
        )
    except IngestionError:
        pass

    # original document's chunks are untouched
    still_there = db_session.execute(
        text("SELECT count(*) FROM chunks WHERE document_id = :id"), {"id": str(good.document_id)}
    ).scalar()
    assert still_there == good_chunk_count


@requires_db
def test_sensitive_flag_is_stored_and_reflected_in_retrieved_chunks(db_session, vector_store, fake_embeddings, tracer):
    raw = b"Confidential internal salary bands and compensation details. " * 10
    result = ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Comp Bands", source_uri="comp.txt", doc_type="txt", raw_bytes=raw, sensitive=True,
    )
    assert result.status == "ready"

    row = db_session.execute(
        text("SELECT metadata FROM documents WHERE id = :id"), {"id": str(result.document_id)}
    ).first()
    assert row.metadata["sensitive"] is True

    hits = vector_store.similarity_search(
        fake_embeddings.embed_batch(["salary bands"])[0], top_k=5, document_id=result.document_id
    )
    assert len(hits) > 0
    assert all(c.is_sensitive for c in hits)
