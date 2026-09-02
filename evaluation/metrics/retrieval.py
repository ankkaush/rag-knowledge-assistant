"""
Deterministic retrieval metrics — no LLM judge involved.

These are the "cheap, trustworthy first line of evidence" from the approved
evaluation strategy: they need no judge model, cost nothing per run, and are
not subject to LLM-as-judge variance. RAGAS's LLM-judged metrics (faithfulness,
answer relevance) are a separate, complementary layer — see ragas_metrics.py.

Because reference_chunk_ids aren't practical to pin in a version-controlled
dataset (see questions.yaml's header comment), correctness here is measured
at DOCUMENT granularity: did chunks from the expected source document appear
in the retrieved set, and what fraction of the retrieved set came from it.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class RetrievalScore:
    hit: bool | None          # None when expected_source_uri is null (not applicable)
    precision_at_k: float | None
    retrieved_source_uris: list[str]


def score_retrieval(retrieved_chunks: list[RetrievedChunk], expected_source_uri: str | None) -> RetrievalScore:
    source_uris = [c.source_uri for c in retrieved_chunks]

    if expected_source_uri is None:
        return RetrievalScore(hit=None, precision_at_k=None, retrieved_source_uris=source_uris)

    if not retrieved_chunks:
        return RetrievalScore(hit=False, precision_at_k=0.0, retrieved_source_uris=source_uris)

    matches = sum(1 for uri in source_uris if uri == expected_source_uri)
    return RetrievalScore(
        hit=matches > 0,
        precision_at_k=matches / len(retrieved_chunks),
        retrieved_source_uris=source_uris,
    )


def score_refusal(answer: str, no_context_answer: str) -> bool:
    """
    For 'unanswerable' questions: did the system correctly decline rather
    than answering from outside knowledge? Exact-match against the fixed
    refusal string is deliberately strict — a near-miss (e.g. the model
    hedges instead of refusing outright) should show up as a failure here,
    not be scored as a pass by a fuzzy match.
    """
    return answer.strip() == no_context_answer


def score_keyword_coverage(answer: str, expected_keywords: list[str]) -> float | None:
    """
    Crude, cheap sanity signal: fraction of expected keywords/phrases that
    appear (case-insensitively) in the answer. NOT a substitute for RAGAS's
    faithfulness/relevance metrics — a wrong sentence can contain the right
    keyword, and a correct paraphrase can miss it. Useful as a fast first
    signal and for a human skimming the report, not as the final word on
    answer quality.
    """
    if not expected_keywords:
        return None
    lowered = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lowered)
    return hits / len(expected_keywords)
