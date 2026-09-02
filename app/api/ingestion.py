from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from app.api.schemas import IngestResponse
from app.core.config import settings
from app.core.db import SessionLocal
from app.core.errors import ValidationError
from app.ingestion.service import ingest_document
from app.observability import get_tracer
from app.providers.embeddings import get_embedding_provider
from app.retrieval.pgvector_store import PgVectorStore

router = APIRouter(prefix="/documents", tags=["ingestion"])

_SUPPORTED_TYPES = {"pdf": "pdf", "txt": "txt", "md": "md"}


@router.post("", response_model=IngestResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    sensitive: bool = Form(default=False),
) -> IngestResponse:
    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if ext not in _SUPPORTED_TYPES:
        raise ValidationError(
            user_message=f"Unsupported file extension '.{ext}'. Supported: {', '.join(_SUPPORTED_TYPES)}."
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise ValidationError(user_message="Uploaded file is empty.")
    max_upload_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw_bytes) > max_upload_bytes:
        raise ValidationError(user_message=f"File exceeds the {settings.max_upload_mb}MB upload limit.")

    # NOTE: ingest_document manages its own transaction boundaries internally
    # (see app/ingestion/service.py docstring) — it commits/rolls back the
    # session itself at well-defined points, so we open a plain session here
    # rather than the auto-commit-on-exit `session_scope()` helper used
    # elsewhere, to avoid two different commit policies fighting over the
    # same session.
    session = SessionLocal()
    try:
        result = ingest_document(
            session=session,
            vector_store=PgVectorStore(session),
            embedding_provider=get_embedding_provider(),
            tracer=get_tracer(),
            title=title or file.filename or "untitled",
            source_uri=file.filename or "unknown",
            doc_type=_SUPPORTED_TYPES[ext],
            raw_bytes=raw_bytes,
            sensitive=sensitive,
        )
    finally:
        session.close()

    return IngestResponse(document_id=result.document_id, status=result.status, chunk_count=result.chunk_count)
