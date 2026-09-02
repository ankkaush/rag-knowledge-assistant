from app.core.config import settings
from app.observability.base import SpanHandle, Tracer
from app.observability.logging_tracer import LoggingTracer

__all__ = ["Tracer", "SpanHandle", "get_tracer"]


def get_tracer() -> Tracer:
    """
    Factory: LangfuseTracer if keys are configured, LoggingTracer otherwise.
    The `langfuse` import is deferred into the branch that needs it so an
    environment without Langfuse configured never has to import that SDK.
    """
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        from app.observability.langfuse_tracer import LangfuseTracer

        return LangfuseTracer(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return LoggingTracer()
