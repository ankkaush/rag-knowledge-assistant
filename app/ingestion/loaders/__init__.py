from app.core.errors import ValidationError
from app.ingestion.loaders.base import DocumentLoader, LoadedPage
from app.ingestion.loaders.pdf_loader import PdfLoader
from app.ingestion.loaders.text_loaders import PlainTextLoader

_LOADERS: list[DocumentLoader] = [PlainTextLoader(), PdfLoader()]


def get_loader(doc_type: str) -> DocumentLoader:
    for loader in _LOADERS:
        if loader.supports(doc_type):
            return loader
    raise ValidationError(
        user_message=f"Unsupported document type: '{doc_type}'. Supported: txt, md, pdf.",
    )


__all__ = ["DocumentLoader", "LoadedPage", "get_loader"]
