import io

import pytest
from pypdf import PdfWriter

from app.core.errors import IngestionError, ValidationError
from app.ingestion.loaders import get_loader


def test_get_loader_dispatches_by_type():
    assert get_loader("txt").supports("txt")
    assert get_loader("md").supports("md")
    assert get_loader("pdf").supports("pdf")


def test_get_loader_rejects_unsupported_type():
    with pytest.raises(ValidationError):
        get_loader("docx")


def test_plain_text_loader_decodes_utf8():
    loader = get_loader("txt")
    pages = loader.load("hello world".encode("utf-8"))
    assert len(pages) == 1
    assert pages[0].text == "hello world"
    assert pages[0].page_number is None


def test_plain_text_loader_rejects_invalid_utf8():
    loader = get_loader("txt")
    with pytest.raises(IngestionError):
        loader.load(b"\xff\xfe\x00invalid")


def test_pdf_loader_extracts_text_per_page():
    # Build a minimal real PDF in-memory rather than relying on a fixture file,
    # so the test is self-contained and doesn't depend on repo assets.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    loader = get_loader("pdf")
    pages = loader.load(buf.getvalue())

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_pdf_loader_rejects_garbage_bytes():
    loader = get_loader("pdf")
    with pytest.raises(IngestionError):
        loader.load(b"this is not a pdf")
