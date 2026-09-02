"""
Chunking: splitting cleaned document text into retrieval-sized pieces.

WHY CHUNK AT ALL:
Embedding an entire document as one vector would average away everything
specific in it — a 20-page document's embedding would be a vague blur, unable
to distinguish "what does section 3 say about X" from "what does section 7 say
about Y". Chunking trades this for many smaller, more specific vectors: each
chunk's embedding represents a narrow enough slice of text that similarity
search against it can genuinely discriminate.

STRATEGY CHOSEN: fixed-size character window with overlap, snapped to the
nearest whitespace so words aren't split mid-token.

  chunk_size_chars=1200, chunk_overlap_chars=200  (see .env.example)

Why these numbers, and the trade-off they represent:
- Too small (e.g. 200 chars): each chunk is nearly a single sentence. Retrieval
  becomes very precise about matching exact phrasing, but a chunk alone often
  lacks enough surrounding context for the LLM to answer from it correctly —
  and you multiply the number of chunks (and embedding calls) per document.
- Too large (e.g. 4000 chars): each chunk covers so much ground that its
  embedding becomes an average over multiple unrelated ideas, diluting
  semantic precision — a query about one specific fact in a large chunk may
  not score that chunk highly even though the fact is present. You also waste
  context-window budget carrying irrelevant text alongside the relevant part.
- ~1200 chars (~250-300 words) is a common practical middle ground: usually
  large enough to contain a complete idea/paragraph, small enough to stay
  topically focused.

WHY OVERLAP (200 chars, ~15-20% of chunk size):
A fact that happens to sit right at a chunk boundary would otherwise be split
across two chunks, with neither one containing the complete idea. Overlap
means the boundary region appears in both neighboring chunks, so at least one
of them contains it intact. The cost is redundant storage/embedding compute —
acceptable at this scale; not free at very large scale.

WHAT THIS STRATEGY DELIBERATELY DOES NOT DO (Phase 1 scope):
- No semantic/structure-aware chunking (e.g. splitting on markdown headers,
  or using an LLM to find "natural" boundaries). That's a real technique but
  adds complexity not justified until evaluation (Phase 3) shows fixed-size
  chunking is actually a bottleneck.
- No token-exact sizing via a real tokenizer (e.g. tiktoken). We approximate
  token_count as len(text) / 4, a well-known rough heuristic for English text.
  This is stored for visibility/debugging, not used to enforce hard limits
  here — good enough for Phase 1, worth revisiting if context-budget bugs show up.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    token_count: int
    page_number: int | None = None
    section_title: str | None = None


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _snap_to_whitespace(text: str, pos: int, search_window: int = 80) -> int:
    """
    Adjust a candidate cut point `pos` to the nearest preceding whitespace,
    within `search_window` characters, so we don't split a word in half.
    Falls back to the raw position if no whitespace is found nearby.
    """
    if pos >= len(text):
        return len(text)
    window_start = max(0, pos - search_window)
    slice_ = text[window_start:pos]
    last_space = slice_.rfind(" ")
    if last_space == -1:
        return pos
    return window_start + last_space + 1


def _snap_forward_to_whitespace(text: str, pos: int, search_window: int = 80) -> int:
    """
    Like `_snap_to_whitespace` but looks forward: used for a chunk's START
    position, so overlap never begins mid-word. Without this, the *end* of a
    chunk snaps cleanly to a word boundary, but the next chunk's start
    (computed from `end - overlap`) is a raw character offset that can land
    inside the word straddling that boundary, e.g. starting mid-way through
    "word18" and emitting the stray token "18".
    """
    if pos <= 0 or pos >= len(text):
        return pos
    if text[pos - 1] == " ":
        return pos  # already on a word boundary
    window_end = min(len(text), pos + search_window)
    next_space = text.find(" ", pos, window_end)
    if next_space == -1:
        return pos  # no boundary nearby; fall back rather than skipping too much text
    return next_space + 1


def chunk_text(
    text: str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    page_number: int | None = None,
    section_title: str | None = None,
) -> list[Chunk]:
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    text_len = len(text)
    step = chunk_size_chars - chunk_overlap_chars

    while start < text_len:
        raw_end = min(start + chunk_size_chars, text_len)
        end = _snap_to_whitespace(text, raw_end) if raw_end < text_len else text_len
        if end <= start:
            end = raw_end  # guard against pathological no-whitespace text

        content = text[start:end].strip()
        if content:
            chunks.append(
                Chunk(
                    chunk_index=index,
                    content=content,
                    char_start=start,
                    char_end=end,
                    token_count=_approx_token_count(content),
                    page_number=page_number,
                    section_title=section_title,
                )
            )
            index += 1

        if end >= text_len:
            break
        next_start = max(start + step, end - chunk_overlap_chars) if step > 0 else end
        start = _snap_forward_to_whitespace(text, next_start)

    return chunks
