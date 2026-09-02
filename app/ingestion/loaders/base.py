"""
DocumentLoader interface.

The boundary this represents: "given raw bytes of some file type, produce
plain text (optionally split by page)". Different file formats need genuinely
different extraction logic (a PDF has a binary structure and pagination; a
.txt file is already plain text) — this is a case where an interface is
earning its keep, not decoration, because there are real, different
implementations behind it from day one.

`LoadedPage` keeps page-level granularity where the format has it (PDF), so
`page_number` can be attached to chunks and survive to citations later. For
formats with no native pagination (txt/md), a loader returns one page with
`page_number=None`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedPage:
    text: str
    page_number: int | None


class DocumentLoader(ABC):
    @abstractmethod
    def supports(self, doc_type: str) -> bool: ...

    @abstractmethod
    def load(self, raw_bytes: bytes) -> list[LoadedPage]:
        """Extract text from raw file bytes. Raises IngestionError on failure."""
        ...
