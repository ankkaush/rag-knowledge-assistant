from __future__ import annotations

from app.core.errors import IngestionError
from app.ingestion.loaders.base import DocumentLoader, LoadedPage


class PlainTextLoader(DocumentLoader):
    """Handles .txt and .md — both are already plain text; no structural extraction needed."""

    def supports(self, doc_type: str) -> bool:
        return doc_type in ("txt", "md")

    def load(self, raw_bytes: bytes) -> list[LoadedPage]:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IngestionError(
                user_message="File could not be decoded as UTF-8 text.",
                internal_detail=f"UnicodeDecodeError: {exc}",
            ) from exc
        return [LoadedPage(text=text, page_number=None)]
