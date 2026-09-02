"""
Citation extraction and validation.

This is the deterministic guardrail behind the prompt's grounding instruction
(prompts.py): the LLM is *asked* to cite honestly, but nothing stops it from
citing a bracket number it invented. We do not trust the prompt alone —
every citation the model emits is checked against the actual context blocks
that were sent, by index. An index outside that range is dropped, not passed
through, and logged, since it's evidence the model deviated from the
grounding instruction (useful signal, formalized as an observability metric
in Phase 4).

This directly implements the "invalid citation references" failure mode
identified in the architecture review before Phase 1 began.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from uuid import UUID

from app.generation.context_builder import ContextBlock

logger = logging.getLogger("rag.citations")

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: UUID
    document_title: str
    source_uri: str
    page_number: int | None


def resolve_citations(answer_text: str, context_blocks: list[ContextBlock]) -> list[Citation]:
    blocks_by_index = {b.index: b for b in context_blocks}

    cited_indices: list[int] = []
    seen = set()
    for match in _CITATION_PATTERN.finditer(answer_text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            cited_indices.append(idx)

    citations: list[Citation] = []
    for idx in cited_indices:
        block = blocks_by_index.get(idx)
        if block is None:
            logger.warning(
                "invalid_citation_index cited_index=%s valid_range=1..%d",
                idx, len(context_blocks),
            )
            continue
        c = block.chunk
        citations.append(
            Citation(
                index=idx,
                chunk_id=c.id,
                document_title=c.document_title,
                source_uri=c.source_uri,
                page_number=c.page_number,
            )
        )
    return citations
