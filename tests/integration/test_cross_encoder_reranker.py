"""
Exercises the REAL local cross-encoder (not the fake) — skipped automatically
if the optional `rerank` extra (sentence-transformers/torch) isn't installed.
This is the one place in the test suite that downloads/loads an actual model;
kept separate and clearly marked since it's slower than everything else.
"""
from __future__ import annotations

import uuid

import pytest

from app.retrieval.types import RetrievedChunk

try:
    from app.retrieval.cross_encoder_reranker import CrossEncoderReranker

    _RERANK_AVAILABLE = True
except ImportError:
    _RERANK_AVAILABLE = False

requires_rerank_extra = pytest.mark.skipif(
    not _RERANK_AVAILABLE, reason="sentence-transformers not installed — pip install -e '.[rerank]'"
)


def _chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.uuid4(), document_id=uuid.uuid4(), chunk_index=0, content=content,
        page_number=None, section_title=None, distance=0.5,
        document_title="Doc", source_uri="doc.txt", is_sensitive=False,
    )


@requires_rerank_extra
def test_cross_encoder_ranks_relevant_passage_above_irrelevant_one():
    reranker = CrossEncoderReranker()
    candidates = [
        _chunk("Photosynthesis converts sunlight into chemical energy in plants."),
        _chunk("The Eiffel Tower stands 330 meters tall and is located in Paris."),
    ]
    result = reranker.rerank("How tall is the Eiffel Tower?", candidates, top_k=2)

    assert result[0].content.startswith("The Eiffel Tower")
    assert result[0].rerank_score > result[1].rerank_score


@requires_rerank_extra
def test_cross_encoder_respects_top_k():
    reranker = CrossEncoderReranker()
    candidates = [_chunk(f"Filler passage number {i} about nothing in particular.") for i in range(5)]
    result = reranker.rerank("irrelevant query", candidates, top_k=2)
    assert len(result) == 2
