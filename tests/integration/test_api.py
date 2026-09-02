"""
API-level integration tests: exercise the actual HTTP boundary (FastAPI
TestClient), including request validation and the AppError -> JSON error
mapping — not just the service layer directly.

The embedding provider is monkeypatched to FakeEmbeddingProvider at the route
module level (rather than a real OpenAI call) so these tests don't require a
live API key or network access; they verify the API wiring/contract, not the
OpenAI SDK integration itself (see app/providers/embeddings/openai_provider.py
for that logic, exercised at the unit-test level via error classification).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.ingestion as ingestion_module
import app.api.query as query_module
import app.api.retrieval as retrieval_module
from app.core.config import settings
from app.main import app
from app.observability.logging_tracer import LoggingTracer
from app.providers.embeddings.fake_provider import FakeEmbeddingProvider
from app.providers.llm.fake_provider import FakeLLMProvider
from app.providers.llm.ollama_provider import OllamaChatProvider
from app.providers.llm.router import LLMRouter
from tests.integration.conftest import requires_db
from tests.integration.test_ollama_provider import requires_ollama

client = TestClient(app)


@pytest.fixture(autouse=True)
def use_fake_providers(monkeypatch):
    fake_embeddings = FakeEmbeddingProvider(dimensions=1536)
    monkeypatch.setattr(ingestion_module, "get_embedding_provider", lambda: fake_embeddings)
    monkeypatch.setattr(retrieval_module, "get_embedding_provider", lambda: fake_embeddings)
    monkeypatch.setattr(query_module, "get_embedding_provider", lambda: fake_embeddings)
    monkeypatch.setattr(query_module, "get_llm_provider", lambda: FakeLLMProvider())
    # These routes call get_tracer() directly (not dependency-injected) — the
    # real one now resolves to LangfuseTracer whenever real credentials are
    # configured in .env, which would send every test request as a real trace
    # to whatever Langfuse project is configured. Force LoggingTracer here so
    # this test file's behavior never depends on what's in a developer's .env.
    monkeypatch.setattr(ingestion_module, "get_tracer", lambda: LoggingTracer())
    monkeypatch.setattr(retrieval_module, "get_tracer", lambda: LoggingTracer())
    monkeypatch.setattr(query_module, "get_tracer", lambda: LoggingTracer())


@requires_db
def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@requires_db
def test_upload_and_search_round_trip(db_session):
    content = ("The Eiffel Tower is located in Paris, France. " * 30).encode("utf-8")
    resp = client.post(
        "/documents",
        files={"file": ("paris.txt", content, "text/plain")},
        data={"title": "Paris Facts"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] > 0

    search_resp = client.post("/search", json={"query": "Where is the Eiffel Tower?", "top_k": 3})
    assert search_resp.status_code == 200
    results = search_resp.json()["results"]
    assert len(results) > 0
    assert "Eiffel" in results[0]["content"]


@requires_db
def test_upload_rejects_unsupported_extension():
    resp = client.post("/documents", files={"file": ("bad.docx", b"whatever", "application/octet-stream")})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error_code"] == "bad_request"
    assert "trace_id" in body


@requires_db
def test_upload_rejects_empty_file():
    resp = client.post("/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "bad_request"


@requires_db
def test_search_validates_top_k_bounds():
    resp = client.post("/search", json={"query": "test", "top_k": 500})
    assert resp.status_code == 422  # FastAPI/Pydantic validation error, not our AppError path


@requires_db
def test_query_returns_grounded_answer_with_citations(db_session):
    content = ("The Eiffel Tower is located in Paris, France. " * 30).encode("utf-8")
    upload = client.post("/documents", files={"file": ("paris.txt", content, "text/plain")})
    assert upload.status_code == 201

    resp = client.post("/query", json={"question": "Where is the Eiffel Tower?", "top_k": 3})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"]
    assert body["model"] == "fake-llm-v1"
    assert len(body["retrieved_chunks"]) > 0
    assert len(body["citations"]) >= 1
    # every citation must resolve to a real document/source, not a placeholder
    for c in body["citations"]:
        assert c["document_title"]
        assert c["source_uri"]


@requires_db
def test_query_against_empty_corpus_returns_no_context_answer(db_session):
    resp = client.post("/query", json={"question": "Anything?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "I don't have enough information in the provided documents to answer this question."
    assert body["citations"] == []
    assert body["retrieved_chunks"] == []
    assert body["model"] is None


@requires_db
def test_query_validates_question_length():
    resp = client.post("/query", json={"question": "", "top_k": 3})
    assert resp.status_code == 422


@requires_db
@requires_ollama
def test_query_hybrid_routing_end_to_end_through_http(db_session, monkeypatch):
    """Full HTTP-boundary proof of Phase 6: a sensitive document routes to
    and is answered by the real local Ollama model, never the API provider."""
    router = LLMRouter(
        api_provider=FakeLLMProvider(response_text="SHOULD NEVER BE CALLED FOR A SENSITIVE REQUEST"),
        local_provider=OllamaChatProvider(
            base_url=settings.ollama_base_url, model=settings.ollama_model, timeout_seconds=60, max_retries=1,
        ),
    )
    monkeypatch.setattr(query_module, "get_llm_router", lambda: router)

    content = b"CONFIDENTIAL: internal codename for the new product is Nighthawk. " * 15
    upload = client.post(
        "/documents", files={"file": ("secret.txt", content, "text/plain")}, data={"sensitive": "true"},
    )
    assert upload.status_code == 201

    resp = client.post("/query", json={"question": "What is the internal codename?", "top_k": 3})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["route"] == "local"
    assert body["model"] == settings.ollama_model
    assert "SHOULD NEVER BE CALLED" not in body["answer"]


@requires_db
def test_error_response_never_leaks_internal_detail(db_session):
    # A whitespace-only .txt is valid UTF-8 but has no meaningful content ->
    # IngestionError. Confirm the response is generic and contains no
    # exception class names, file paths, or stack trace fragments.
    resp = client.post("/documents", files={"file": ("blank.txt", b"   \n\n  ", "text/plain")})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "ingestion_failed"
    assert "Traceback" not in body["message"]
    assert "app/ingestion" not in body["message"]
