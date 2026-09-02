import uuid

from app.generation.citations import resolve_citations
from app.generation.context_builder import ContextBlock
from app.retrieval.types import RetrievedChunk


def _block(index: int) -> ContextBlock:
    chunk = RetrievedChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        content="content",
        page_number=3,
        section_title=None,
        distance=0.1,
        document_title=f"Doc {index}",
        source_uri=f"doc{index}.txt",
        is_sensitive=False,
    )
    return ContextBlock(index=index, chunk=chunk)


def test_resolves_single_valid_citation():
    blocks = [_block(1), _block(2)]
    citations = resolve_citations("Paris is the capital of France [1].", blocks)
    assert len(citations) == 1
    assert citations[0].index == 1
    assert citations[0].document_title == "Doc 1"


def test_resolves_multiple_citations_in_order_of_first_appearance():
    blocks = [_block(1), _block(2), _block(3)]
    citations = resolve_citations("Fact A [2]. Fact B [1]. Fact C [2].", blocks)
    assert [c.index for c in citations] == [2, 1]  # dedup, preserves first-seen order


def test_drops_out_of_range_citation_index():
    blocks = [_block(1)]
    # Model hallucinated [5] which was never shown to it.
    citations = resolve_citations("Some fact [1] and another [5].", blocks)
    assert [c.index for c in citations] == [1]


def test_no_citations_present():
    blocks = [_block(1)]
    citations = resolve_citations("I don't have enough information.", blocks)
    assert citations == []


def test_no_context_blocks_means_no_citations_resolve():
    citations = resolve_citations("Some answer [1].", [])
    assert citations == []
