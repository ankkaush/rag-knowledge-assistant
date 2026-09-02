"""
Integration test fixtures.

These tests hit the real Dockerized Postgres+pgvector (see docker/docker-compose.yml)
and use the FakeEmbeddingProvider (no network/API key needed) so ingestion and
retrieval logic — including the actual pgvector SQL — is exercised for real,
without depending on an external embedding API in CI or local dev.

Each test gets a clean `chunks`/`documents` state via truncation in a fixture,
rather than a fresh database per test — faster, and fine at this scale since
tests don't run concurrently against the same DB.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from app.core.db import SessionLocal, get_engine
from app.observability.logging_tracer import LoggingTracer
from app.providers.embeddings.fake_provider import FakeEmbeddingProvider
from app.retrieval.pgvector_store import PgVectorStore


def _db_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_available(), reason="Postgres not reachable — run `docker compose -f docker/docker-compose.yml up -d`"
)


@pytest.fixture()
def db_session():
    session = SessionLocal()
    session.execute(text("TRUNCATE documents CASCADE"))
    session.commit()
    yield session
    session.close()


@pytest.fixture()
def vector_store(db_session):
    return PgVectorStore(db_session)


@pytest.fixture()
def tracer():
    # Every test uses LoggingTracer, never LangfuseTracer — tracing must not
    # make tests flaky or dependent on an external service. See
    # app/observability/logging_tracer.py.
    return LoggingTracer()


@pytest.fixture()
def fake_embeddings():
    # Small dimension count for test speed; note this intentionally does NOT
    # match the schema's vector(1536) column width. pgvector requires the
    # stored vector's length to match the column, so tests use 1536 too —
    # see test files for where this matters.
    return FakeEmbeddingProvider(dimensions=1536)
