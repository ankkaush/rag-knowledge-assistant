from app.ingestion.service import ingest_document
from app.retrieval.fake_reranker import FakeReranker
from app.retrieval.retriever import Retriever
from tests.integration.conftest import requires_db


@requires_db
def test_retriever_without_reranker_returns_vector_order(db_session, vector_store, fake_embeddings, tracer):
    ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Doc", source_uri="doc.txt", doc_type="txt",
        raw_bytes=b"paris eiffel tower facts here in this document about paris. " * 10,
    )
    retriever = Retriever(vector_store=vector_store, embedding_provider=fake_embeddings, default_top_k=3)
    results = retriever.retrieve("eiffel tower paris", top_k=3)
    assert all(r.rerank_score is None for r in results)


@requires_db
def test_retriever_with_reranker_widens_candidates_then_narrows(db_session, vector_store, fake_embeddings, tracer):
    # Ingest several distinct short documents so there's a real candidate pool to narrow.
    for i in range(6):
        ingest_document(
            session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
            title=f"Doc {i}", source_uri=f"doc{i}.txt", doc_type="txt",
            raw_bytes=f"this is filler content number {i} about topic {i}. ".encode() * 10,
        )

    reranker = FakeReranker()
    retriever = Retriever(
        vector_store=vector_store, embedding_provider=fake_embeddings,
        default_top_k=2, reranker=reranker, rerank_candidate_multiplier=3,
    )
    results = retriever.retrieve("topic 3", top_k=2)

    assert len(results) == 2
    assert all(r.rerank_score is not None for r in results)


@requires_db
def test_use_reranker_false_bypasses_reranker_even_when_configured(db_session, vector_store, fake_embeddings, tracer):
    ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Doc", source_uri="doc.txt", doc_type="txt",
        raw_bytes=b"paris eiffel tower facts here. " * 10,
    )
    retriever = Retriever(
        vector_store=vector_store, embedding_provider=fake_embeddings,
        default_top_k=3, reranker=FakeReranker(),
    )
    results = retriever.retrieve("eiffel tower", top_k=3, use_reranker=False)
    assert all(r.rerank_score is None for r in results)


@requires_db
def test_doc_type_filter_excludes_other_types(db_session, vector_store, fake_embeddings, tracer):
    # NOTE: content must differ between the two documents — content_hash-based
    # idempotency (app/ingestion/service.py) keys off raw bytes only, so
    # byte-identical content across two "different" uploads would collide
    # onto the same document row regardless of the declared doc_type.
    ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Text Doc", source_uri="a.txt", doc_type="txt",
        raw_bytes=b"shared keyword content about apples in the text file. " * 10,
    )
    ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Markdown Doc", source_uri="b.md", doc_type="md",
        raw_bytes=b"shared keyword content about apples in the markdown file. " * 10,
    )

    retriever = Retriever(vector_store=vector_store, embedding_provider=fake_embeddings, default_top_k=10)
    results = retriever.retrieve("apples", top_k=10, doc_type="md")

    assert len(results) > 0
    assert all(r.source_uri == "b.md" for r in results)
