import uuid

from app.generation.context_builder import build_context
from app.retrieval.types import RetrievedChunk


def _chunk(content: str, distance: float, page: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        content=content,
        page_number=page,
        section_title=None,
        distance=distance,
        document_title="Doc",
        source_uri="doc.txt",
        is_sensitive=False,
    )


def test_build_context_numbers_blocks_from_one():
    chunks = [_chunk("first chunk text", 0.1), _chunk("second chunk text", 0.2)]
    built = build_context(chunks, max_tokens=10_000)
    assert [b.index for b in built.blocks] == [1, 2]
    assert "[1]" in built.text
    assert "[2]" in built.text


def test_build_context_includes_page_number_when_present():
    built = build_context([_chunk("text", 0.1, page=7)], max_tokens=10_000)
    assert "page 7" in built.text


def test_build_context_omits_page_when_absent():
    built = build_context([_chunk("text", 0.1, page=None)], max_tokens=10_000)
    assert "page" not in built.text


def test_build_context_truncates_least_similar_chunks_first():
    # Each chunk ~400 chars -> ~100 tokens. Budget for 2 chunks only.
    chunks = [_chunk("a" * 400, 0.1), _chunk("b" * 400, 0.2), _chunk("c" * 400, 0.3)]
    built = build_context(chunks, max_tokens=220)

    assert len(built.blocks) == 2
    assert built.dropped_chunk_count == 1
    kept_contents = {b.chunk.content[0] for b in built.blocks}
    assert kept_contents == {"a", "b"}  # the most-similar two were kept
    assert "c" * 400 not in built.text  # least-similar (highest distance) was dropped


def test_build_context_always_keeps_at_least_one_chunk_even_over_budget():
    huge = _chunk("x" * 100_000, 0.1)
    built = build_context([huge], max_tokens=10)
    assert len(built.blocks) == 1
    assert built.dropped_chunk_count == 0


def test_build_context_empty_input():
    built = build_context([], max_tokens=1000)
    assert built.blocks == []
    assert built.text == ""
    assert built.dropped_chunk_count == 0
