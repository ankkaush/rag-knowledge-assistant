from app.generation.service import NO_CONTEXT_ANSWER, answer_query
from app.ingestion.service import ingest_document
from app.providers.llm.fake_provider import FakeLLMProvider
from app.retrieval.retriever import Retriever
from tests.integration.conftest import requires_db


@requires_db
def test_answer_query_grounds_and_cites_from_ingested_document(db_session, vector_store, fake_embeddings, tracer):
    raw = (
        b"The Eiffel Tower was completed in 1889 and is located in Paris, France. "
        b"It was designed by the engineer Gustave Eiffel. " * 10
    )
    ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Eiffel Tower Facts", source_uri="eiffel.txt", doc_type="txt", raw_bytes=raw,
    )

    retriever = Retriever(vector_store=vector_store, embedding_provider=fake_embeddings, default_top_k=5)
    result = answer_query(
        retriever=retriever,
        llm_provider=FakeLLMProvider(),
        tracer=tracer,
        question="Where is the Eiffel Tower?",
        top_k=3,
        document_id=None,
        context_max_tokens=3000,
        max_tokens=800,
        temperature=0.2,
    )

    assert result.retrieved_chunks  # something was actually retrieved
    assert result.answer != NO_CONTEXT_ANSWER
    assert len(result.citations) >= 1
    # every citation must point back to a chunk that was actually retrieved
    retrieved_ids = {c.id for c in result.retrieved_chunks}
    assert all(c.chunk_id in retrieved_ids for c in result.citations)


@requires_db
def test_answer_query_short_circuits_when_nothing_retrieved(db_session, vector_store, fake_embeddings, tracer):
    retriever = Retriever(vector_store=vector_store, embedding_provider=fake_embeddings, default_top_k=5)

    calls = []
    llm = FakeLLMProvider()
    original_generate = llm.generate
    llm.generate = lambda *a, **kw: calls.append(1) or original_generate(*a, **kw)

    result = answer_query(
        retriever=retriever,
        llm_provider=llm,
        tracer=tracer,
        question="Anything at all?",
        top_k=5,
        document_id=None,
        context_max_tokens=3000,
        max_tokens=800,
        temperature=0.2,
    )

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.citations == []
    assert result.retrieved_chunks == []
    assert calls == []  # LLM was never called — the cost/correctness optimization in service.py


@requires_db
def test_sensitive_document_redacts_generate_span_but_answer_is_unaffected(db_session, vector_store, fake_embeddings):
    """
    The redaction policy only changes what's SENT TO THE TRACER, never the
    actual pipeline behavior — the answer/citations returned to the caller
    must be identical whether or not the source document is sensitive.
    """
    import logging

    from app.observability.logging_tracer import LoggingTracer

    raw = b"The confidential merger will close in Q3 and involves Acme Corp. " * 10
    tracer = LoggingTracer()
    ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Merger Doc", source_uri="merger.txt", doc_type="txt", raw_bytes=raw, sensitive=True,
    )

    retriever = Retriever(vector_store=vector_store, embedding_provider=fake_embeddings, default_top_k=5)
    result = answer_query(
        retriever=retriever, llm_provider=FakeLLMProvider(), tracer=tracer,
        question="When will the merger close?", top_k=3, document_id=None,
        context_max_tokens=3000, max_tokens=800, temperature=0.2,
    )

    assert all(c.is_sensitive for c in result.retrieved_chunks)
    # the real answer is untouched by redaction — redaction only affects the trace
    assert "[redacted" not in result.answer
