"""
API-level tests for Phase 7 (auth + upload limits) through the real app,
including the exception-handler -> JSON mapping for the new AuthenticationError.
Rate limiting is covered directly against the middleware in
tests/unit/test_rate_limit.py — not repeated here at the full-app level.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.ingestion as ingestion_module
import app.api.retrieval as retrieval_module
from app.core import config
from app.main import app
from app.observability.logging_tracer import LoggingTracer
from app.providers.embeddings.fake_provider import FakeEmbeddingProvider
from tests.integration.conftest import requires_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def use_fake_embeddings(monkeypatch):
    fake_embeddings = FakeEmbeddingProvider(dimensions=1536)
    monkeypatch.setattr(ingestion_module, "get_embedding_provider", lambda: fake_embeddings)
    monkeypatch.setattr(retrieval_module, "get_embedding_provider", lambda: fake_embeddings)
    # See tests/integration/test_api.py for why this matters: these routes
    # call get_tracer() directly, which resolves to a real LangfuseTracer
    # whenever real credentials are configured in .env.
    monkeypatch.setattr(ingestion_module, "get_tracer", lambda: LoggingTracer())
    monkeypatch.setattr(retrieval_module, "get_tracer", lambda: LoggingTracer())


@requires_db
def test_auth_disabled_by_default_no_key_needed():
    resp = client.post("/search", json={"query": "anything", "top_k": 3})
    assert resp.status_code == 200


@requires_db
def test_auth_enabled_rejects_request_without_key(monkeypatch):
    monkeypatch.setattr(config.settings, "api_auth_enabled", True)
    monkeypatch.setattr(config.settings, "api_keys", "test-key-123")
    resp = client.post("/search", json={"query": "anything", "top_k": 3})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "unauthorized"


@requires_db
def test_auth_enabled_accepts_request_with_valid_key(monkeypatch):
    monkeypatch.setattr(config.settings, "api_auth_enabled", True)
    monkeypatch.setattr(config.settings, "api_keys", "test-key-123")
    resp = client.post(
        "/search", json={"query": "anything", "top_k": 3}, headers={"X-API-Key": "test-key-123"},
    )
    assert resp.status_code == 200


@requires_db
def test_auth_enabled_never_blocks_health_check(monkeypatch):
    monkeypatch.setattr(config.settings, "api_auth_enabled", True)
    monkeypatch.setattr(config.settings, "api_keys", "test-key-123")
    resp = client.get("/health")
    assert resp.status_code == 200


@requires_db
def test_upload_limit_is_configurable(monkeypatch):
    monkeypatch.setattr(config.settings, "max_upload_mb", 0)  # anything nonzero should now be rejected
    resp = client.post("/documents", files={"file": ("small.txt", b"just a few bytes", "text/plain")})
    assert resp.status_code == 400
    assert "MB upload limit" in resp.json()["message"]
