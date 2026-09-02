import uuid

from sqlalchemy import text

from app.ingestion.chunking import Chunk
from tests.integration.conftest import requires_db


def _make_document(session, title="Doc", source_uri="doc.txt", content_hash=None) -> uuid.UUID:
    row = session.execute(
        text(
            """
            INSERT INTO documents (title, source_uri, doc_type, status, content_hash)
            VALUES (:title, :source_uri, 'txt', 'processing', :content_hash)
            RETURNING id
            """
        ),
        {"title": title, "source_uri": source_uri, "content_hash": content_hash or str(uuid.uuid4())},
    ).first()
    session.commit()
    return row.id


@requires_db
def test_similarity_search_returns_most_similar_chunk_first(db_session, vector_store, fake_embeddings):
    doc_id = _make_document(db_session)

    texts = ["the cat sat on the mat", "quantum physics and relativity", "stock market closed higher today"]
    chunks = [
        Chunk(chunk_index=i, content=t, char_start=0, char_end=len(t), token_count=len(t) // 4)
        for i, t in enumerate(texts)
    ]
    embeddings = fake_embeddings.embed_batch(texts)
    vector_store.replace_document_chunks(doc_id, chunks, embeddings)
    db_session.commit()

    query_embedding = fake_embeddings.embed_batch(["a cat sitting on a mat"])[0]
    results = vector_store.similarity_search(query_embedding, top_k=3)

    assert len(results) == 3
    assert results[0].content == "the cat sat on the mat"
    # results are ordered by increasing distance (most similar first)
    assert results[0].distance <= results[1].distance <= results[2].distance


@requires_db
def test_similarity_search_filters_by_document_id(db_session, vector_store, fake_embeddings):
    doc_a = _make_document(db_session, title="A", content_hash="hash-a")
    doc_b = _make_document(db_session, title="B", content_hash="hash-b")

    texts_a = ["apple banana cherry"]
    texts_b = ["apple banana cherry"]  # identical content, different document
    vector_store.replace_document_chunks(
        doc_a,
        [Chunk(chunk_index=0, content=texts_a[0], char_start=0, char_end=len(texts_a[0]), token_count=5)],
        fake_embeddings.embed_batch(texts_a),
    )
    vector_store.replace_document_chunks(
        doc_b,
        [Chunk(chunk_index=0, content=texts_b[0], char_start=0, char_end=len(texts_b[0]), token_count=5)],
        fake_embeddings.embed_batch(texts_b),
    )
    db_session.commit()

    query_embedding = fake_embeddings.embed_batch(["apple banana cherry"])[0]
    results = vector_store.similarity_search(query_embedding, top_k=10, document_id=doc_a)

    assert len(results) == 1
    assert results[0].document_id == doc_a


@requires_db
def test_replace_document_chunks_is_idempotent(db_session, vector_store, fake_embeddings):
    doc_id = _make_document(db_session)

    first_texts = ["chunk one", "chunk two", "chunk three"]
    first_chunks = [
        Chunk(chunk_index=i, content=t, char_start=0, char_end=len(t), token_count=3) for i, t in enumerate(first_texts)
    ]
    vector_store.replace_document_chunks(doc_id, first_chunks, fake_embeddings.embed_batch(first_texts))
    db_session.commit()

    count = db_session.execute(
        text("SELECT count(*) FROM chunks WHERE document_id = :id"), {"id": str(doc_id)}
    ).scalar()
    assert count == 3

    # Re-ingest with a different, smaller chunk set for the SAME document_id.
    second_texts = ["only one chunk now"]
    second_chunks = [Chunk(chunk_index=0, content=second_texts[0], char_start=0, char_end=len(second_texts[0]), token_count=4)]
    vector_store.replace_document_chunks(doc_id, second_chunks, fake_embeddings.embed_batch(second_texts))
    db_session.commit()

    count = db_session.execute(
        text("SELECT count(*) FROM chunks WHERE document_id = :id"), {"id": str(doc_id)}
    ).scalar()
    assert count == 1  # old chunks were replaced, not accumulated
