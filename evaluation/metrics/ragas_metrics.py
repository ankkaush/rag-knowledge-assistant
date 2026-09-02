"""
RAGAS integration — LLM-as-judge metrics: faithfulness, answer relevance,
context precision.

WHY THESE THREE (and not more):
- Faithfulness: does the answer's claims actually trace back to the provided
  context? This is the hallucination signal — arguably the single most
  important metric for a grounded-QA system, and not something the
  deterministic metrics in retrieval.py can measure at all (they check
  whether the right document was RETRIEVED, not whether the GENERATED answer
  actually used it faithfully).
- Answer relevance: does the answer actually address the question asked
  (independent of whether it's grounded)? Catches a technically-faithful but
  off-topic or evasive answer.
- Context precision: of the retrieved chunks, how many were actually useful
  for answering — a judge-based cross-check on the deterministic
  precision_at_k in retrieval.py, using semantic relevance instead of
  same-document-or-not.

Context RECALL and answer CORRECTNESS are deliberately NOT wired in here:
both require a human-authored reference_answer for every item, which this
dataset only has for a few questions (see questions.yaml) — running them
against items with no reference would silently score against an empty
string, which is worse than not reporting the metric at all.

IMPORTANT DEPENDENCY NOTE (worth understanding, not hiding):
Installing `ragas` (the `eval` extra) pulls in `langchain-core`,
`langchain-community`, and `langchain-openai` as RAGAS's OWN internal
implementation detail — RAGAS uses LangChain's LLM/embedding wrapper
interfaces to talk to the judge model. This is NOT this project adopting
LangChain as its own pipeline framework; app/ and this project's own
generation/retrieval code never import langchain. It's an unavoidable
transitive dependency of using RAGAS at all, worth knowing about if a
dependency audit ever asks "why is langchain in this environment."

IMPORTANT CONFIG NOTE: ragas builds its OWN OpenAI client for the judge
model. It does NOT automatically see OPENAI_API_KEY loaded via this
project's pydantic-settings unless we explicitly hand it a client — .env
values are read into `Settings`, not exported to the process environment.
That's why `client=openai.OpenAI(api_key=settings.openai_api_key)` is passed
explicitly below rather than relying on an ambient env var.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("rag.eval.ragas")


@dataclass(frozen=True)
class RagasItemResult:
    question_id: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None


def run_ragas_evaluation(
    items: list[dict],
    *,
    openai_api_key: str,
    judge_model: str,
    embedding_model: str,
) -> list[RagasItemResult]:
    """
    `items`: list of dicts with keys: id, user_input, response, retrieved_contexts (list[str]).
    Only items with at least one retrieved context are scored (faithfulness
    and context precision are undefined with zero context — this mirrors the
    query service's own no-context short-circuit).
    """
    import openai
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference

    scoreable = [it for it in items if it["retrieved_contexts"]]
    if not scoreable:
        logger.info("ragas_skipped reason=no_scoreable_items")
        return []

    client = openai.OpenAI(api_key=openai_api_key)
    judge_llm = llm_factory(judge_model, provider="openai", client=client)
    judge_embeddings = OpenAIEmbeddings(client=client, model=embedding_model)

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": it["user_input"],
                "response": it["response"],
                "retrieved_contexts": it["retrieved_contexts"],
            }
            for it in scoreable
        ]
    )

    faithfulness_metric = Faithfulness()
    relevancy_metric = AnswerRelevancy()
    precision_metric = LLMContextPrecisionWithoutReference()

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness_metric, relevancy_metric, precision_metric],
        llm=judge_llm,
        embeddings=judge_embeddings,
        show_progress=False,
    )

    # Column names are read off each metric's own `.name` rather than
    # hardcoded, so this doesn't silently break if ragas renames a column in
    # a future version — it would raise a clear KeyError instead.
    df = result.to_pandas()
    out: list[RagasItemResult] = []
    for (_, row), item in zip(df.iterrows(), scoreable):
        out.append(
            RagasItemResult(
                question_id=item["id"],
                faithfulness=_safe_float(row.get(faithfulness_metric.name)),
                answer_relevancy=_safe_float(row.get(relevancy_metric.name)),
                context_precision=_safe_float(row.get(precision_metric.name)),
            )
        )
    return out


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return f if f == f else None  # filter NaN
    except (TypeError, ValueError):
        return None
