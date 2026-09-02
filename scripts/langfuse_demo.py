"""
Runs a couple of representative queries through the REAL query pipeline
(app.generation.service.answer_query — the exact code path /query uses) with
a real LangfuseTracer, so the resulting traces can be screenshotted from
Langfuse Cloud for the project README/portfolio.

This script does not add any new architecture: it wires together the same
get_embedding_provider() / get_llm_provider() / get_llm_router() / get_reranker()
/ get_tracer() factories the API itself uses (see app/api/query.py), just from
a script instead of an HTTP request, so the two demo queries below produce
traces shaped exactly like real production traffic would.

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to be set (in .env) —
this script deliberately refuses to run with LoggingTracer, since the whole
point is to prove a trace reaches Langfuse Cloud, not to demonstrate the
fallback.

NO OPENAI KEY REQUIRED: this project has no local/free embedding provider
(see README's AI/ML components section), so without a real OPENAI_API_KEY,
embeddings fall back to the deterministic FakeEmbeddingProvider — retrieval
still works correctly against this small demo corpus (verified in Phase 1/5
testing), it just isn't a real embedding model. Generation, however, is real:
set LLM_PROVIDER=ollama in .env to use a real local model. The embedding
model name is never shown in a *query* trace (only in the separate ingestion
trace's embed_batch span — see app/generation/service.py), so this doesn't
compromise what the query trace demonstrates: a real model actually answering,
with real token counts.

Usage:
    python -m scripts.langfuse_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.core.config import settings
from app.core.db import SessionLocal
from app.generation.service import answer_query
from app.ingestion.service import ingest_document
from app.observability import get_tracer
from app.observability.langfuse_tracer import LangfuseTracer
from app.providers.embeddings import get_embedding_provider
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.embeddings.fake_provider import FakeEmbeddingProvider
from app.providers.llm import get_llm_provider, get_llm_router
from app.retrieval import get_reranker
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.retriever import Retriever

CORPUS_DIR = Path(__file__).parent.parent / "evaluation" / "dataset" / "corpus"

# This demo ingests only 2 documents (2 chunks each = 4 chunks total). Using
# settings.default_top_k (5) would retrieve every chunk from BOTH documents
# for every question, regardless of relevance — including the sensitive
# document's chunks, which would redact every trace, not just the one that's
# supposed to demonstrate redaction. A smaller top_k, explicit to this demo
# script (not a change to the app's actual default), keeps each query's
# retrieval scoped to its actually-relevant document.
_DEMO_TOP_K = 2

_PLACEHOLDER_OPENAI_KEY = "sk-..."  # the literal placeholder shipped in .env.example


def _resolve_embedding_provider() -> EmbeddingProvider:
    if settings.openai_api_key and settings.openai_api_key != _PLACEHOLDER_OPENAI_KEY:
        return get_embedding_provider()
    print(
        "No real OPENAI_API_KEY configured -> using FakeEmbeddingProvider for "
        "embeddings. Generation still uses whatever LLM_PROVIDER is configured "
        "(e.g. a real local Ollama model). See this script's module docstring."
    )
    return FakeEmbeddingProvider(dimensions=settings.embedding_dimensions)


def main() -> None:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        print(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set in .env.\n"
            "This script requires them — it exists specifically to prove a real\n"
            "Langfuse Cloud trace, not to demonstrate the LoggingTracer fallback.\n"
            "See README.md's Observability section for how to get them.",
            file=sys.stderr,
        )
        sys.exit(1)

    tracer = get_tracer()
    if not isinstance(tracer, LangfuseTracer):
        print("get_tracer() did not return a LangfuseTracer despite keys being set — aborting.", file=sys.stderr)
        sys.exit(1)

    embedding_provider = _resolve_embedding_provider()
    reranker = get_reranker()
    router = get_llm_router()
    llm_provider = None if router else get_llm_provider()

    session = SessionLocal()
    try:
        vector_store = PgVectorStore(session)

        print("Ingesting demo documents...")
        eiffel_result = ingest_document(
            session=session, vector_store=vector_store, embedding_provider=embedding_provider, tracer=tracer,
            title="Eiffel Tower", source_uri="eiffel_tower.txt", doc_type="txt",
            raw_bytes=(CORPUS_DIR / "eiffel_tower.txt").read_bytes(), sensitive=False,
        )
        policy_result = ingest_document(
            session=session, vector_store=vector_store, embedding_provider=embedding_provider, tracer=tracer,
            title="Remote Work Policy", source_uri="remote_work_policy.txt", doc_type="txt",
            raw_bytes=(CORPUS_DIR / "remote_work_policy.txt").read_bytes(), sensitive=True,
        )
        print(f"  eiffel_tower.txt -> {eiffel_result.status}, {eiffel_result.chunk_count} chunks (not sensitive)")
        print(f"  remote_work_policy.txt -> {policy_result.status}, {policy_result.chunk_count} chunks (flagged sensitive)")

        retriever = Retriever(
            vector_store=vector_store, embedding_provider=embedding_provider,
            default_top_k=settings.default_top_k, reranker=reranker,
            rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
        )

        print("\nRunning representative query 1/2 (non-sensitive, full trace visible)...")
        result1 = answer_query(
            retriever=retriever, tracer=tracer, llm_provider=llm_provider, router=router,
            question="How tall is the Eiffel Tower and when was it completed?",
            top_k=_DEMO_TOP_K, document_id=None,
            context_max_tokens=settings.context_max_tokens,
            max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature,
        )
        print(f"  answer: {result1.answer[:200]}")
        print(f"  model: {result1.model} | route: {result1.route or 'n/a (hybrid routing disabled)'}")

        print("\nRunning representative query 2/2 (sensitive document -> redacted trace)...")
        result2 = answer_query(
            retriever=retriever, tracer=tracer, llm_provider=llm_provider, router=router,
            question="How many paid time off days do employees get?",
            top_k=_DEMO_TOP_K, document_id=None,
            context_max_tokens=settings.context_max_tokens,
            max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature,
        )
        print(f"  answer: {result2.answer[:200]}")
        print(f"  model: {result2.model} | route: {result2.route or 'n/a (hybrid routing disabled)'}")
    finally:
        session.close()

    print("\nFlushing pending spans to Langfuse Cloud...")
    tracer.flush()

    print(
        f"\nDone. Open {settings.langfuse_host} -> your project -> Traces.\n"
        "The two most recent traces are named \"query\" — the second one\n"
        "(remote work policy question) should show a redacted prompt/answer"
        + (" and route=local." if router else ".")
    )


if __name__ == "__main__":
    main()
