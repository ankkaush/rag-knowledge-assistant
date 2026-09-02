from app.core.config import settings
from app.retrieval.reranker import Reranker

__all__ = ["Reranker", "get_reranker"]


def get_reranker() -> Reranker | None:
    """
    None (disabled) is a normal, fully-supported state — Retriever treats a
    missing reranker as "skip the rerank stage," not an error. The
    `sentence_transformers` import is deferred into the branch that needs it
    so an environment without the `rerank` extra installed never has to
    import torch.
    """
    if not settings.reranker_enabled:
        return None
    from app.retrieval.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker(model_name=settings.reranker_model)
