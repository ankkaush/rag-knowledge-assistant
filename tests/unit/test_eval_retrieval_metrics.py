import uuid

from evaluation.metrics.retrieval import score_keyword_coverage, score_refusal, score_retrieval
from app.retrieval.types import RetrievedChunk


def _chunk(source_uri: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        content="content",
        page_number=None,
        section_title=None,
        distance=0.1,
        document_title="Doc",
        source_uri=source_uri,
        is_sensitive=False,
    )


def test_score_retrieval_hit_and_precision():
    chunks = [_chunk("a.txt"), _chunk("a.txt"), _chunk("b.txt")]
    score = score_retrieval(chunks, expected_source_uri="a.txt")
    assert score.hit is True
    assert score.precision_at_k == 2 / 3


def test_score_retrieval_miss():
    chunks = [_chunk("b.txt"), _chunk("c.txt")]
    score = score_retrieval(chunks, expected_source_uri="a.txt")
    assert score.hit is False
    assert score.precision_at_k == 0.0


def test_score_retrieval_empty_results_is_a_miss_not_none():
    score = score_retrieval([], expected_source_uri="a.txt")
    assert score.hit is False
    assert score.precision_at_k == 0.0


def test_score_retrieval_not_applicable_for_unanswerable_questions():
    score = score_retrieval([_chunk("a.txt")], expected_source_uri=None)
    assert score.hit is None
    assert score.precision_at_k is None


def test_score_refusal_exact_match_required():
    no_context = "I don't have enough information in the provided documents to answer this question."
    assert score_refusal(no_context, no_context) is True
    assert score_refusal(no_context + " But maybe...", no_context) is False
    assert score_refusal("The capital of Japan is Tokyo.", no_context) is False


def test_score_keyword_coverage():
    assert score_keyword_coverage("Python was created by Guido van Rossum in 1991.", ["1991", "guido van rossum"]) == 1.0
    assert score_keyword_coverage("Python is a language.", ["1991", "guido van rossum"]) == 0.0
    assert score_keyword_coverage("Released in 1991.", ["1991", "guido van rossum"]) == 0.5


def test_score_keyword_coverage_none_when_no_keywords_expected():
    assert score_keyword_coverage("anything", []) is None
