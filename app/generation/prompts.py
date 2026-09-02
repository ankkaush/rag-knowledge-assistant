"""
The grounding prompt.

This is the actual mechanism that makes the system "RAG" rather than "chat
with extra text pasted in": the model is explicitly instructed to answer only
from the provided context and to refuse when the context doesn't contain the
answer. This is a soft guarantee (a good-enough system prompt, not a proof) —
it's backed up by the deterministic citation-index validation in
citations.py, which is the part that can't be talked around by the model.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a knowledge assistant that answers questions using ONLY the numbered \
context blocks provided below each question. Each block is labeled with a bracketed number, e.g. [1].

Rules:
- Answer only using information present in the provided context blocks. Do not use outside knowledge.
- When you state a fact drawn from a context block, cite it inline using its bracket number, e.g. "Paris is the capital of France [1]."
- If multiple blocks support a statement, cite all of them, e.g. "[1][3]".
- If the answer is not contained in the provided context, respond with exactly this sentence and nothing else: \
"I don't have enough information in the provided documents to answer this question."
- Do not fabricate a citation number that was not shown to you.
"""


def build_user_message(question: str, context_text: str) -> str:
    return f"Context:\n\n{context_text}\n\nQuestion: {question}"
