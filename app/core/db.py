"""
Database engine/session setup.

We use SQLAlchemy Core (not the ORM) for the `chunks`/`documents` tables in
Phase 1, and write retrieval queries as explicit SQL. This is a deliberate
choice for a learning project: an ORM's query builder would abstract away
exactly the pgvector operators (`<=>` cosine distance) we most need to
understand. SQLAlchemy is used here only for connection pooling, parameterized
query execution, and transactions — not as a query-generation layer.

`connect_timeout` (a psycopg/libpq option) bounds how long we wait to
*establish* a connection if Postgres is unreachable or overloaded — without it,
a downed database can hang a request indefinitely instead of failing fast.
`pool_pre_ping` checks a pooled connection is still alive before handing it to
a caller, so a connection that died from an idle timeout doesn't surface as a
confusing mid-query error.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

_engine: Engine = create_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    pool_pre_ping=True,
    connect_args={"connect_timeout": settings.db_connect_timeout_seconds},
)

SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_engine() -> Engine:
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """One transaction per call: commits on success, rolls back on any exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
