import uuid

from app.retrieval.fake_reranker import FakeReranker
from app.retrieval.types import RetrievedChunk


def _chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.uuid4(), document_id=uuid.uuid4(), chunk_index=0, content=content,
        page_number=None, section_title=None, distance=0.5,
        document_title="Doc", source_uri="doc.txt", is_sensitive=False,
    )


def test_rerank_reorders_by_word_overlap():
    candidates = [
        _chunk("completely unrelated text about weather"),
        _chunk("the eiffel tower is in paris"),
        _chunk("paris is also known for the louvre museum"),
    ]
    reranker = FakeReranker()
    result = reranker.rerank("eiffel tower paris", candidates, top_k=3)

    assert result[0].content == "the eiffel tower is in paris"
    assert all(c.rerank_score is not None for c in result)


def test_rerank_respects_top_k():
    candidates = [_chunk(f"chunk {i} paris") for i in range(10)]
    reranker = FakeReranker()
    result = reranker.rerank("paris", candidates, top_k=3)
    assert len(result) == 3


def test_rerank_empty_candidates():
    reranker = FakeReranker()
    assert reranker.rerank("anything", [], top_k=5) == []


def test_original_chunk_fields_preserved_after_rerank():
    original = _chunk("paris facts")
    result = FakeReranker().rerank("paris", [original], top_k=1)
    assert result[0].id == original.id
    assert result[0].content == original.content
    assert result[0].distance == original.distance
