"""
Context construction: turns ranked RetrievedChunks into the numbered,
citation-taggable text block that goes into the prompt.

KEY CONCEPT — context budget vs. model context window:
`context_max_tokens` (default 3000, see app/core/config.py) bounds only the
retrieved-context portion of the prompt — it is deliberately smaller than the
model's actual context window (e.g. 128k for gpt-4o-mini). Why a separate,
smaller budget:
- The model's full context window is a hard ceiling, not a target — filling
  it with retrieved text leaves no room for a growing conversation, wastes
  money on tokens that dilute relevance, and (per the chunking rationale in
  Phase 1) more low-relevance context tends to produce vaguer, not better,
  answers.
- Chunks arrive from the Retriever already ranked by distance (most similar
  first). Truncation therefore always drops the LEAST similar chunks first —
  the ones least likely to matter — never the most relevant ones.

TRUNCATION IS LOGGED, NOT SILENT:
When chunks are dropped, the caller (app/generation/service.py) is told how
many were dropped, specifically so this is visible behavior, not something
that quietly degrades answer quality with no trace of why. Full observability
(a span/log entry) is wired up formally in Phase 4; today it's a plain return
value the service logs.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class ContextBlock:
    index: int  # 1-based — matches the "[N]" citation marker shown to the LLM
    chunk: RetrievedChunk


@dataclass(frozen=True)
class BuiltContext:
    text: str
    blocks: list[ContextBlock]
    dropped_chunk_count: int


def build_context(chunks: list[RetrievedChunk], max_tokens: int) -> BuiltContext:
    blocks: list[ContextBlock] = []
    used_tokens = 0

    for chunk in chunks:
        # Reuses the same len(text)//4 approximation as Phase 1 chunking
        # (app/ingestion/chunking.py) — consistent, not exact, and documented
        # there as a known simplification.
        chunk_tokens = max(1, len(chunk.content) // 4)
        if blocks and used_tokens + chunk_tokens > max_tokens:
            break  # stop at the first chunk that would exceed budget; keep at least one
        blocks.append(ContextBlock(index=len(blocks) + 1, chunk=chunk))
        used_tokens += chunk_tokens

    lines = []
    for block in blocks:
        c = block.chunk
        location = f", page {c.page_number}" if c.page_number is not None else ""
        lines.append(f'[{block.index}] (Source: "{c.document_title}"{location}):\n{c.content}')

    return BuiltContext(
        text="\n\n".join(lines),
        blocks=blocks,
        dropped_chunk_count=len(chunks) - len(blocks),
    )
