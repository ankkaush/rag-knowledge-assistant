"""
Text cleaning/normalization applied after extraction, before chunking.

Kept deliberately small and deterministic for Phase 1. Each step exists to fix
a specific, observable problem in extracted text — not as generic "cleanup":

- Unicode normalization (NFKC): PDF extraction frequently produces visually
  identical but differently-encoded characters (e.g. ligatures like "ﬁ" vs
  "fi", or non-breaking vs regular spaces). Without normalizing, two chunks
  containing "the same" word can embed slightly differently, and exact-text
  operations (dedup, testing) become unreliable.
- Whitespace collapsing: PDF extraction often inserts stray newlines mid-sentence
  (from line wrapping in the original layout) and runs of blank lines between
  paragraphs. Collapsing these makes chunk boundaries reflect actual content
  breaks rather than PDF line-wrap artifacts.
- Blank-page / near-empty-page detection: scanned/cover pages sometimes extract
  to a handful of stray characters. These are dropped so they don't become
  useless chunks that pad top-K retrieval results.
"""
from __future__ import annotations

import re
import unicodedata

_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MID_LINE_BREAK = re.compile(r"(?<![.\n:;!?])\n(?!\n)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

MIN_MEANINGFUL_CHARS = 20


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    # Join lines that were broken mid-sentence by PDF layout, but keep
    # deliberate paragraph breaks (blank line) and breaks after punctuation.
    text = _MID_LINE_BREAK.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def is_meaningful(text: str) -> bool:
    """True if a cleaned page/section has enough content to be worth chunking."""
    return len(text.strip()) >= MIN_MEANINGFUL_CHARS
