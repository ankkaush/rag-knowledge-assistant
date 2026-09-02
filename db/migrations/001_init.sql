-- Phase 1 schema.
--
-- Design decisions worth understanding:
--
-- 1. `documents` and `chunks` are separate tables. A document is the unit of
--    ingestion/idempotency; a chunk is the unit of retrieval. Keeping them
--    separate (rather than one denormalized table) is what lets us re-ingest
--    a document (delete+replace its chunks) without touching document identity,
--    and lets multiple chunks share one document's metadata by foreign key
--    instead of duplicating it into every chunk row.
--
-- 2. `embedding vector(1536)` is a pgvector column. pgvector is a *Postgres
--    extension*, not a separate database — the vector lives in the same row,
--    same table, same transaction as the chunk text and metadata. That's the
--    core thing to understand: "vector database" here just means "a regular
--    relational table with one column that supports nearest-neighbor search."
--
-- 3. We store `embedding_model` on `documents` (not assumed globally) so that
--    if the embedding model is ever changed, old and new documents can be
--    distinguished — mixing vectors from two different embedding models in
--    one similarity search silently produces meaningless results, since the
--    vector spaces aren't comparable.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    source_uri      TEXT NOT NULL,
    doc_type        TEXT NOT NULL,               -- 'pdf' | 'txt' | 'md'
    status          TEXT NOT NULL DEFAULT 'pending', -- pending|processing|ready|failed
    content_hash    TEXT NOT NULL,                -- sha256 of raw bytes; idempotency key
    embedding_model TEXT,
    chunk_size_chars    INTEGER,
    chunk_overlap_chars INTEGER,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_detail    TEXT,                         -- internal-only diagnostic, never returned to API callers
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (content_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    page_number     INTEGER,
    section_title   TEXT,
    char_start      INTEGER NOT NULL,
    char_end        INTEGER NOT NULL,
    token_count     INTEGER,
    embedding       vector(1536) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- No approximate-nearest-neighbor index (IVFFlat/HNSW) in Phase 1 — deliberately.
--
-- IVFFlat/HNSW trade correctness for speed: they cluster vectors and only
-- search a subset of clusters per query. That trade only pays off once you
-- have enough rows for the clustering to be meaningful. At small row counts
-- (a handful of documents, as in dev/testing), an IVFFlat index with the
-- default `lists`/`probes` settings can genuinely return EMPTY or WRONG
-- results — most clusters end up with ~0 rows, and the single cluster probed
-- by default may not contain any of the few rows that exist. This was caught
-- directly during Phase 1 integration testing: adding IVFFlat with lists=100
-- against a 3-row test table made `similarity_search` return nothing, even
-- though a plain sequential scan found the correct nearest neighbor.
--
-- Without an index, pgvector falls back to an exact sequential scan for
-- `ORDER BY embedding <=> :query LIMIT k` — slower at large scale, but always
-- correct, and fast enough at the row counts this project will see through
-- Phase 1-3. Add IVFFlat or HNSW deliberately later (with `lists` sized to
-- actual row count, and only after ANALYZE has run) if/when retrieval latency
-- actually becomes a measured problem — not preemptively.

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
