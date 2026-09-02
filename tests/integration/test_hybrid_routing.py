"""
End-to-end hybrid routing through the real query pipeline (answer_query),
using a real local Ollama model for the "local" side and a fake for "api"
(no OPENAI_API_KEY in this environment — see test_ollama_provider.py for
Ollama-only coverage, and test_llm_router.py for the routing policy itself
in isolation). This is the one place that proves the FULL path: a sensitive
document -> retrieval flags it -> router picks local -> a real local model
actually answers it.
"""
from __future__ import annotations

from app.core.config import settings
from app.generation.service import answer_query
from app.ingestion.service import ingest_document
from app.providers.llm.fake_provider import FakeLLMProvider
from app.providers.llm.ollama_provider import OllamaChatProvider
from app.providers.llm.router import LLMRouter
from app.retrieval.retriever import Retriever
from tests.integration.conftest import requires_db
from tests.integration.test_ollama_provider import requires_ollama


@requires_db
@requires_ollama
def test_sensitive_document_query_routes_to_and_answers_via_local_model(db_session, vector_store, fake_embeddings, tracer):
    raw = b"CONFIDENTIAL: the acquisition target is codenamed Project Falcon. " * 15
    ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Confidential Memo", source_uri="memo.txt", doc_type="txt", raw_bytes=raw, sensitive=True,
    )

    retriever = Retriever(vector_store=vector_store, embedding_provider=fake_embeddings, default_top_k=3)
    router = LLMRouter(
        api_provider=FakeLLMProvider(response_text="THIS SHOULD NEVER BE USED FOR A SENSITIVE REQUEST"),
        local_provider=OllamaChatProvider(
            base_url=settings.ollama_base_url, model=settings.ollama_model, timeout_seconds=60, max_retries=1,
        ),
    )

    result = answer_query(
        retriever=retriever, router=router, tracer=tracer,
        question="What is the codename for the acquisition target?",
        top_k=3, document_id=None, context_max_tokens=3000, max_tokens=100, temperature=0.0,
    )

    assert result.route == "local"
    assert result.model == settings.ollama_model
    assert "NEVER BE USED" not in result.answer  # proves the fake API provider was never called


@requires_db
@requires_ollama
def test_non_sensitive_document_query_routes_to_api(db_session, vector_store, fake_embeddings, tracer):
    raw = b"Public fact: the company was founded in 2010 and is headquartered in Austin. " * 10
    ingest_document(
        session=db_session, vector_store=vector_store, embedding_provider=fake_embeddings, tracer=tracer,
        title="Public Facts", source_uri="public.txt", doc_type="txt", raw_bytes=raw, sensitive=False,
    )

    retriever = Retriever(vector_store=vector_store, embedding_provider=fake_embeddings, default_top_k=3)
    router = LLMRouter(
        api_provider=FakeLLMProvider(response_text="answer from api [1]"),
        local_provider=OllamaChatProvider(
            base_url=settings.ollama_base_url, model=settings.ollama_model, timeout_seconds=60, max_retries=1,
        ),
    )

    result = answer_query(
        retriever=retriever, router=router, tracer=tracer,
        question="When was the company founded?",
        top_k=3, document_id=None, context_max_tokens=3000, max_tokens=100, temperature=0.0,
    )

    assert result.route == "api"
    assert result.answer == "answer from api [1]"
