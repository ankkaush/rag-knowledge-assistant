"""
Evaluation harness: ingest the fixed corpus, run every question through the
REAL query pipeline (app.generation.service.answer_query — the exact same
code path /query uses), score retrieval + refusal + citations
deterministically, optionally score faithfulness/relevance/context-precision
via RAGAS, and write a timestamped JSON report.

WHY THIS REUSES THE REAL PIPELINE RATHER THAN RE-IMPLEMENTING IT:
An evaluation harness that calls its own separate retrieval/generation logic
would tell you whether THAT logic is good — not whether the actual system
users hit is good. Every question below goes through
app.retrieval.retriever.Retriever and app.generation.service.answer_query,
unmodified.

REPRODUCIBILITY: the corpus is a fixed, version-controlled set of files
(evaluation/dataset/corpus/) ingested idempotently (same content_hash logic
as Phase 1) — re-running this script twice ingests nothing new the second
time and should produce statistically similar results (LLM-judge scores can
vary slightly run to run; deterministic retrieval scores will not).

Usage:
    python -m evaluation.run_eval                          # real providers (needs OPENAI_API_KEY)
    python -m evaluation.run_eval --skip-ragas              # real providers, deterministic metrics only
    python -m evaluation.run_eval --provider fake           # fully offline smoke-test of the harness itself
    python -m evaluation.run_eval --reranker on             # enable reranking (needs the `rerank` extra with --provider real)
    python -m evaluation.run_eval --reranker compare         # run twice (off vs on) and print/save a before/after diff —
                                                               # this is the Phase 5 Definition-of-Done check.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.core.config import settings
from app.core.db import SessionLocal
from app.generation.service import NO_CONTEXT_ANSWER, answer_query
from app.ingestion.service import ingest_document
from app.observability import get_tracer
from app.observability.base import Tracer
from app.providers.embeddings import get_embedding_provider
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.embeddings.fake_provider import FakeEmbeddingProvider
from app.providers.llm import get_llm_provider
from app.providers.llm.base import LLMProvider
from app.providers.llm.fake_provider import FakeLLMProvider
from app.retrieval import get_reranker
from app.retrieval.fake_reranker import FakeReranker
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from evaluation.metrics.ragas_metrics import run_ragas_evaluation
from evaluation.metrics.retrieval import score_keyword_coverage, score_refusal, score_retrieval

logger = logging.getLogger("rag.eval")

DATASET_DIR = Path(__file__).parent / "dataset"
CORPUS_DIR = DATASET_DIR / "corpus"
QUESTIONS_PATH = DATASET_DIR / "questions.yaml"
REPORTS_DIR = Path(__file__).parent / "reports"

_EXT_TO_DOC_TYPE = {".txt": "txt", ".md": "md", ".pdf": "pdf"}


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def ingest_corpus(session, vector_store, embedding_provider: EmbeddingProvider, tracer: Tracer) -> None:
    for file_path in sorted(CORPUS_DIR.iterdir()):
        doc_type = _EXT_TO_DOC_TYPE.get(file_path.suffix)
        if doc_type is None:
            continue
        result = ingest_document(
            session=session,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            tracer=tracer,
            title=file_path.stem.replace("_", " ").title(),
            source_uri=file_path.name,
            doc_type=doc_type,
            raw_bytes=file_path.read_bytes(),
        )
        logger.info("ingested source_uri=%s status=%s chunks=%d", file_path.name, result.status, result.chunk_count)


def run_evaluation(
    *,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    top_k: int,
    run_ragas: bool,
    tracer: Tracer | None = None,
    reranker: Reranker | None = None,
) -> dict:
    tracer = tracer or get_tracer()
    session = SessionLocal()
    try:
        vector_store = PgVectorStore(session)
        ingest_corpus(session, vector_store, embedding_provider, tracer)
        retriever = Retriever(
            vector_store=vector_store, embedding_provider=embedding_provider,
            default_top_k=top_k, reranker=reranker,
        )

        questions = load_questions()
        items: list[dict] = []

        for q in questions:
            start = time.monotonic()
            result = answer_query(
                retriever=retriever,
                llm_provider=llm_provider,
                tracer=tracer,
                question=q["question"],
                top_k=top_k,
                document_id=None,
                context_max_tokens=settings.context_max_tokens,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
            latency_seconds = time.monotonic() - start

            retrieval_score = score_retrieval(result.retrieved_chunks, q["expected_source_uri"])
            item = {
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "expected_source_uri": q["expected_source_uri"],
                "answer": result.answer,
                "model": result.model,
                "citation_count": len(result.citations),
                "retrieved_chunk_count": len(result.retrieved_chunks),
                "latency_seconds": round(latency_seconds, 3),
                "retrieval_hit": retrieval_score.hit,
                "retrieval_precision_at_k": retrieval_score.precision_at_k,
                "refusal_correct": (
                    score_refusal(result.answer, NO_CONTEXT_ANSWER) if q["category"] == "unanswerable" else None
                ),
                "keyword_coverage": score_keyword_coverage(result.answer, q.get("expected_keywords") or []),
                "user_input": q["question"],
                "response": result.answer,
                "retrieved_contexts": [c.content for c in result.retrieved_chunks],
            }
            items.append(item)

        ragas_by_id: dict[str, dict] = {}
        if run_ragas:
            if not settings.openai_api_key:
                logger.warning("ragas_skipped reason=no_openai_api_key")
            else:
                ragas_results = run_ragas_evaluation(
                    items,
                    openai_api_key=settings.openai_api_key,
                    judge_model=settings.llm_model,
                    embedding_model=settings.embedding_model,
                )
                ragas_by_id = {r.question_id: r for r in ragas_results}

        for item in items:
            r = ragas_by_id.get(item["id"])
            item["faithfulness"] = r.faithfulness if r else None
            item["answer_relevancy"] = r.answer_relevancy if r else None
            item["context_precision"] = r.context_precision if r else None
            # drop the raw fields only needed to build the ragas dataset
            del item["user_input"], item["response"], item["retrieved_contexts"]

        return _build_report(items, embedding_provider, llm_provider, top_k, reranker)
    finally:
        session.close()


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _build_report(items: list[dict], embedding_provider, llm_provider, top_k: int, reranker: Reranker | None) -> dict:
    factual = [i for i in items if i["category"] == "factual"]
    unanswerable = [i for i in items if i["category"] == "unanswerable"]

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "embedding_model": embedding_provider.model_name,
            "llm_model": llm_provider.model_name,
            "top_k": top_k,
            "chunk_size_chars": settings.chunk_size_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "context_max_tokens": settings.context_max_tokens,
            "reranker_enabled": reranker is not None,
        },
        "summary": {
            "question_count": len(items),
            "factual_question_count": len(factual),
            "unanswerable_question_count": len(unanswerable),
            "retrieval_hit_rate": _mean([1.0 if i["retrieval_hit"] else 0.0 for i in factual]),
            "retrieval_precision_at_k_mean": _mean([i["retrieval_precision_at_k"] for i in factual]),
            "refusal_accuracy": _mean([1.0 if i["refusal_correct"] else 0.0 for i in unanswerable]),
            "keyword_coverage_mean": _mean([i["keyword_coverage"] for i in factual]),
            "faithfulness_mean": _mean([i["faithfulness"] for i in items]),
            "answer_relevancy_mean": _mean([i["answer_relevancy"] for i in items]),
            "context_precision_mean": _mean([i["context_precision"] for i in items]),
            "latency_seconds_mean": _mean([i["latency_seconds"] for i in items]),
        },
        "items": items,
    }


def _print_summary(report: dict) -> None:
    s = report["summary"]
    print(f"\n=== Evaluation run: {report['run_at']} ===")
    print(f"Config: {report['config']}")
    print(f"Questions: {s['question_count']} ({s['factual_question_count']} factual, {s['unanswerable_question_count']} unanswerable)")
    print(f"  Retrieval hit rate (factual):        {s['retrieval_hit_rate']}")
    print(f"  Retrieval precision@k (factual):     {s['retrieval_precision_at_k_mean']}")
    print(f"  Refusal accuracy (unanswerable):     {s['refusal_accuracy']}")
    print(f"  Keyword coverage (factual, sanity):  {s['keyword_coverage_mean']}")
    print(f"  Faithfulness (RAGAS):                {s['faithfulness_mean']}")
    print(f"  Answer relevancy (RAGAS):             {s['answer_relevancy_mean']}")
    print(f"  Context precision (RAGAS):            {s['context_precision_mean']}")
    print(f"  Mean latency (s):                     {s['latency_seconds_mean']}")


def _build_reranker(provider: str) -> Reranker:
    """
    --provider fake -> FakeReranker (deterministic, offline, no torch needed —
    exercises the retrieval/rerank WIRING, not real ranking quality).
    --provider real -> the configured real reranker (needs the `rerank` extra).
    """
    if provider == "fake":
        return FakeReranker()
    reranker = get_reranker()
    if reranker is None:
        # get_reranker() returns None when settings.reranker_enabled is False;
        # here the CLI flag is the explicit override, so construct it directly
        # rather than silently running without reranking.
        from app.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    return reranker


def _print_comparison(report_off: dict, report_on: dict) -> None:
    s_off, s_on = report_off["summary"], report_on["summary"]
    print("\n=== Reranking comparison: OFF vs ON ===")
    rows = [
        ("Retrieval hit rate (factual)", "retrieval_hit_rate"),
        ("Retrieval precision@k (factual)", "retrieval_precision_at_k_mean"),
        ("Refusal accuracy (unanswerable)", "refusal_accuracy"),
        ("Keyword coverage (factual, sanity)", "keyword_coverage_mean"),
        ("Faithfulness (RAGAS)", "faithfulness_mean"),
        ("Answer relevancy (RAGAS)", "answer_relevancy_mean"),
        ("Context precision (RAGAS)", "context_precision_mean"),
        ("Mean latency (s)", "latency_seconds_mean"),
    ]
    for label, key in rows:
        off_v, on_v = s_off[key], s_on[key]
        delta = f"{on_v - off_v:+.4f}" if (off_v is not None and on_v is not None) else "n/a"
        print(f"  {label:38s} off={off_v!s:>8} on={on_v!s:>8} delta={delta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation harness.")
    parser.add_argument("--provider", choices=["real", "fake"], default="real")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    parser.add_argument(
        "--reranker", choices=["off", "on", "compare"], default="off",
        help="'compare' runs the full eval twice (off vs on) and prints/saves a before/after diff.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level)

    if args.provider == "fake":
        embedding_provider: EmbeddingProvider = FakeEmbeddingProvider(dimensions=settings.embedding_dimensions)
        llm_provider: LLMProvider = FakeLLMProvider()
        run_ragas = False  # RAGAS scoring a fake echo LLM is meaningless
    else:
        embedding_provider = get_embedding_provider()
        llm_provider = get_llm_provider()
        run_ragas = not args.skip_ragas

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.reranker == "compare":
        report_off = run_evaluation(
            embedding_provider=embedding_provider, llm_provider=llm_provider,
            top_k=args.top_k, run_ragas=run_ragas, reranker=None,
        )
        report_on = run_evaluation(
            embedding_provider=embedding_provider, llm_provider=llm_provider,
            top_k=args.top_k, run_ragas=run_ragas, reranker=_build_reranker(args.provider),
        )
        out_path = REPORTS_DIR / f"eval_{timestamp}_compare.json"
        out_path.write_text(json.dumps({"off": report_off, "on": report_on}, indent=2, default=str))

        print("--- Reranking OFF ---")
        _print_summary(report_off)
        print("\n--- Reranking ON ---")
        _print_summary(report_on)
        _print_comparison(report_off, report_on)
    else:
        reranker = _build_reranker(args.provider) if args.reranker == "on" else None
        report = run_evaluation(
            embedding_provider=embedding_provider, llm_provider=llm_provider,
            top_k=args.top_k, run_ragas=run_ragas, reranker=reranker,
        )
        out_path = REPORTS_DIR / f"eval_{timestamp}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        _print_summary(report)

    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
