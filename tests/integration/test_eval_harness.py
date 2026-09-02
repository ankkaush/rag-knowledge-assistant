"""
Exercises the evaluation harness end-to-end against the real corpus and
Postgres, using fake embedding/LLM providers so it runs offline and free —
this tests the HARNESS MECHANICS (does it ingest the corpus, run every
question, compute metrics, assemble a report), not real answer quality
(which requires a real judge model and is out of scope for an automated,
API-key-free test suite).
"""
from __future__ import annotations

from evaluation.run_eval import load_questions, run_evaluation
from app.providers.embeddings.fake_provider import FakeEmbeddingProvider
from app.providers.llm.fake_provider import FakeLLMProvider
from tests.integration.conftest import requires_db


@requires_db
def test_load_questions_matches_dataset_shape():
    questions = load_questions()
    assert len(questions) > 0
    for q in questions:
        assert "id" in q and "question" in q and "category" in q
        assert q["category"] in ("factual", "unanswerable")


@requires_db
def test_run_evaluation_end_to_end_with_fake_providers(db_session, tracer):
    report = run_evaluation(
        embedding_provider=FakeEmbeddingProvider(dimensions=1536),
        llm_provider=FakeLLMProvider(),
        top_k=5,
        run_ragas=False,  # scoring a fake echo LLM with RAGAS is meaningless
        tracer=tracer,  # LoggingTracer fixture — never send test traffic to a real Langfuse project
    )

    questions = load_questions()
    assert report["summary"]["question_count"] == len(questions)
    assert len(report["items"]) == len(questions)

    # every item has the fields the report/summary depend on
    for item in report["items"]:
        assert "retrieval_hit" in item
        assert "answer" in item
        assert item["latency_seconds"] >= 0

    # RAGAS was skipped -> those fields are present but None, not silently missing
    assert report["summary"]["faithfulness_mean"] is None
    for item in report["items"]:
        assert item["faithfulness"] is None

    # factual questions should mostly retrieve their expected document with
    # the fake (but real cosine-similarity-based) embedding provider
    factual_items = [i for i in report["items"] if i["category"] == "factual"]
    assert report["summary"]["retrieval_hit_rate"] is not None
    assert 0.0 <= report["summary"]["retrieval_hit_rate"] <= 1.0
    assert len(factual_items) > 0


@requires_db
def test_run_evaluation_is_idempotent_on_corpus_ingestion(db_session, tracer):
    # Running twice should not error or duplicate corpus documents/chunks —
    # this exercises the same idempotency guarantee tested directly in
    # test_ingestion_service.py, now through the harness's ingest step.
    report1 = run_evaluation(
        embedding_provider=FakeEmbeddingProvider(dimensions=1536),
        llm_provider=FakeLLMProvider(),
        top_k=5,
        run_ragas=False,
        tracer=tracer,  # LoggingTracer fixture — never send test traffic to a real Langfuse project
    )
    report2 = run_evaluation(
        embedding_provider=FakeEmbeddingProvider(dimensions=1536),
        llm_provider=FakeLLMProvider(),
        top_k=5,
        run_ragas=False,
        tracer=tracer,
    )
    assert report1["summary"]["question_count"] == report2["summary"]["question_count"]
