"""
Query service: retrieve -> build context -> generate -> resolve citations.

KEY ARCHITECTURAL DECISION — no-context short-circuit:
If the retriever returns zero chunks (empty corpus, or a document_id filter
that matches nothing), we do NOT call the LLM at all. We return the fixed
"not enough information" answer directly. Two reasons:
1. Correctness: an LLM given no context has nothing to ground an answer in —
   calling it anyway is just inviting it to answer from outside knowledge
   despite the system prompt, i.e. exactly the failure mode grounding exists
   to prevent.
2. Cost/latency: a generation call that's guaranteed to say "I don't know"
   still costs tokens and time. Skipping it is a free, correct optimization.

TRACING AND THE PRIVACY/REDACTION POLICY:
Every call is wrapped in a "query" root span with "retrieve" and "generate"
child spans (nesting is automatic — see app/observability/base.py). The
retrieve span never logs chunk CONTENT (only counts and source URIs) even for
non-sensitive queries — content only matters for debugging the generation
step, not the retrieval step.

The generate span is where the actual policy lives: if ANY retrieved chunk
comes from a document flagged `sensitive` (app/ingestion/service.py), the
full prompt and answer text are redacted from the trace — only counts,
model, and token usage are recorded. This reuses the exact same `sensitive`
flag that drives Phase 6's local-vs-API routing decision (see
app/providers/llm/router.py) — Phase 4 implemented its observability half,
Phase 6 implements its routing half, off the same flag.

HYBRID ROUTING (Phase 6):
When a `router` (LLMRouter) is supplied, `is_sensitive` also decides WHICH
provider actually serves the request — not just what gets redacted from the
trace. When no router is supplied, `llm_provider` is called directly and
`is_sensitive` only affects tracing, exactly as in Phases 2-5. This keeps
hybrid routing strictly additive: existing callers that pass `llm_provider`
alone are unaffected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from app.generation.citations import Citation, resolve_citations
from app.generation.context_builder import build_context
from app.generation.prompts import SYSTEM_PROMPT, build_user_message
from app.observability.base import Tracer
from app.providers.llm.base import LLMProvider, Message
from app.providers.llm.router import LLMRouter
from app.retrieval.retriever import Retriever
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger("rag.query")

NO_CONTEXT_ANSWER = "I don't have enough information in the provided documents to answer this question."
_REDACTED = "[redacted: retrieved context includes a document flagged sensitive]"


@dataclass(frozen=True)
class QueryResult:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    model: str | None
    route: str | None = None  # "api" | "local" | "local-fallback" | None (no router configured)


def answer_query(
    retriever: Retriever,
    tracer: Tracer,
    *,
    llm_provider: LLMProvider | None = None,
    router: LLMRouter | None = None,
    question: str,
    top_k: int,
    document_id: UUID | None,
    context_max_tokens: int,
    max_tokens: int,
    temperature: float,
    doc_type: str | None = None,
    use_reranker: bool = True,
) -> QueryResult:
    if (llm_provider is None) == (router is None):
        raise ValueError("answer_query requires exactly one of llm_provider or router")
    with tracer.span("query", input={"question": question, "top_k": top_k}) as root:
        with tracer.span(
            "retrieve", as_type="retriever",
            input={"question": question, "top_k": top_k},
            metadata={"use_reranker": use_reranker},
        ) as span:
            chunks = retriever.retrieve(
                question, top_k=top_k, document_id=document_id, doc_type=doc_type, use_reranker=use_reranker,
            )
            span.set_output(
                {
                    "retrieved_count": len(chunks),
                    "source_uris": [c.source_uri for c in chunks],
                    "reranked": any(c.rerank_score is not None for c in chunks),
                }
            )

        if not chunks:
            root.set_output({"answer": NO_CONTEXT_ANSWER, "citation_count": 0, "retrieved_count": 0})
            return QueryResult(answer=NO_CONTEXT_ANSWER, citations=[], retrieved_chunks=[], model=None)

        is_sensitive = any(c.is_sensitive for c in chunks)

        built = build_context(chunks, max_tokens=context_max_tokens)
        if built.dropped_chunk_count:
            logger.info(
                "context_truncated dropped=%d kept=%d question_len=%d",
                built.dropped_chunk_count, len(built.blocks), len(question),
            )

        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=build_user_message(question, built.text)),
        ]

        route: str | None = None
        with tracer.span(
            "generate",
            as_type="generation",
            input=_REDACTED if is_sensitive else {"messages": [{"role": m.role, "content": m.content} for m in messages]},
            metadata={"sensitive": is_sensitive, "context_block_count": len(built.blocks)},
        ) as span:
            if router is not None:
                result, route = router.generate(
                    messages, max_tokens=max_tokens, temperature=temperature, is_sensitive=is_sensitive, tracer=tracer,
                )
            else:
                result = llm_provider.generate(messages, max_tokens=max_tokens, temperature=temperature)
            content_out = _REDACTED if is_sensitive else result.content
            span.set_output({"route": route, "content": content_out} if route is not None else content_out)
            span.set_usage(input_tokens=result.input_tokens, output_tokens=result.output_tokens, model=result.model)

        citations = resolve_citations(result.content, built.blocks)

        root.set_output(
            {
                "citation_count": len(citations),
                "retrieved_count": len(chunks),
                "dropped_chunk_count": built.dropped_chunk_count,
                "sensitive": is_sensitive,
                "route": route,
                "answer": _REDACTED if is_sensitive else result.content,
            }
        )

        return QueryResult(
            answer=result.content,
            citations=citations,
            retrieved_chunks=chunks,
            model=result.model,
            route=route,
        )
