from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.errors import IngestionError
from app.ingestion.loaders.base import DocumentLoader, LoadedPage


class PdfLoader(DocumentLoader):
    """
    Extracts text page-by-page via pypdf.

    Important limitation to understand: pypdf does *layout-unaware* text
    extraction. It does not run OCR, so a scanned/image-only PDF (no embedded
    text layer) will extract to empty or near-empty pages. Phase 1 treats that
    as an ingestion failure per-page (dropped by the cleaning step's
    `is_meaningful` check) rather than silently producing empty chunks.
    OCR is a real feature but explicitly out of scope for Phase 1.
    """

    def supports(self, doc_type: str) -> bool:
        return doc_type == "pdf"

    def load(self, raw_bytes: bytes) -> list[LoadedPage]:
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
        except PdfReadError as exc:
            raise IngestionError(
                user_message="File could not be read as a PDF.",
                internal_detail=f"PdfReadError: {exc}",
            ) from exc

        if reader.is_encrypted:
            raise IngestionError(
                user_message="Encrypted PDFs are not supported.",
                internal_detail="PdfReader.is_encrypted=True",
            )

        pages: list[LoadedPage] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pypdf can raise a variety of parse errors per-page
                raise IngestionError(
                    user_message=f"Failed to extract text from page {i}.",
                    internal_detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            pages.append(LoadedPage(text=text, page_number=i))
        return pages
