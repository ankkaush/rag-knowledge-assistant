# RAG Knowledge Assistant

A retrieval-augmented question-answering backend that answers questions from your own documents — with citations checked against the actual source text, measurable retrieval/answer quality, full request tracing, and a routing layer that keeps sensitive documents off third-party APIs entirely.

**This is a public repository, released under the [MIT License](LICENSE)** — free to use, modify, and adapt.

## Why this exists

"Upload a PDF, ask ChatGPT" demos are easy to build and hard to trust: there's no way to tell whether retrieval is actually finding the right passages, whether the model is answering from the document or from its own training data, or what a request actually costs. This project treats those as first-class requirements, not afterthoughts — every generated answer's citations are checked against the context that was actually sent to the model, retrieval and answer quality are measured against a hand-written evaluation set rather than assumed, every request is traceable end-to-end, and documents can be flagged sensitive so they're answered by a model running entirely on-device instead of a third-party API.

## Status

| Capability | Status |
|---|---|
| Ingestion (PDF/TXT/MD → clean → chunk → embed → store) | ✅ Complete — idempotent by content hash, verified via automated tests against a real Postgres/pgvector instance |
| Query pipeline (retrieve → context → generate → cite) | ✅ Complete — citations validated against the actual context sent, not trusted from the model |
| Retrieval — vector search + metadata filtering | ✅ Complete |
| Retrieval — optional local cross-encoder reranking | ✅ Complete — verified against a real Hugging Face model running locally |
| Evaluation — deterministic retrieval/refusal metrics | ✅ Complete, verified |
| Evaluation — RAGAS LLM-judge metrics (faithfulness, relevance, context precision) | ⚠️ Implemented and code-reviewed — not yet exercised against a live OpenAI judge (no API key configured in development) |
| Observability — request tracing | ✅ Complete — falls back to a zero-dependency local tracer when Langfuse isn't configured; verified end-to-end against a real Langfuse Cloud project, confirmed via the Langfuse API (real model, real token counts, and correct redaction on a sensitive-flagged document) |
| Hybrid inference — routing to a local model | ✅ Complete — verified end-to-end with a real local model (Ollama) installed and running |
| Hybrid inference — routing to the API model | ⚠️ Routing logic verified; the real OpenAI generation call itself hasn't been exercised (no API key configured in development) |
| Deployment hardening (auth, rate limiting, CORS, fail-fast config check) | ✅ Complete — verified directly, including a boot-refusal test for unsafe production settings |
| Live cloud deployment | ⏸️ Not deployed — deliberately out of scope for this stage, see [Deployment status](#deployment-status) |

**Tests: 116/116 passing.** Everything above marked ⚠️ is implemented, unit-tested, and reviewed against the real provider's documented behavior — what's missing is a live call using real paid credentials, not missing code. See [Documentation](#documentation) for the full detail behind every line in this table.

## How it works

```
Document upload (.pdf / .txt / .md)
   → load & extract text (page-aware for PDFs)
   → clean & normalize (unicode, whitespace, boilerplate)
   → chunk (fixed-size, overlap, whitespace-safe boundaries)
   → embed each chunk
   → store chunk + embedding + metadata in Postgres/pgvector, idempotently

Question → /query
   → embed the question
   → vector similarity search (optionally widened, then narrowed by a local reranker)
   → build a numbered, citation-tagged context block within a token budget
   → route to a local or API model, based on whether the retrieved content is flagged sensitive
   → generate a grounded answer
   → validate every citation the model emitted against the context actually sent
   → return the answer, its citations, and the retrieved chunks
```

Every step above is traced (see [Observability](#observability)), and the whole pipeline — not a separate approximation of it — is what the evaluation harness runs against real questions to produce the numbers in [Evaluation](#evaluation).

## Architecture

| Component | Location | Role |
|---|---|---|
| API | `app/main.py`, `app/api/` | FastAPI app — `/documents`, `/search`, `/query`, `/health`; auth, rate limiting, and CORS wired centrally in one place |
| Ingestion | `app/ingestion/` | Load → clean → chunk → embed → store; idempotent re-ingestion by content hash |
| Retrieval | `app/retrieval/` | pgvector similarity search, optional two-stage cross-encoder reranking, `document_id`/`doc_type` metadata filtering |
| Generation | `app/generation/` | Context construction with a token budget, grounded prompting, deterministic citation validation |
| Providers | `app/providers/` | Embedding and LLM backends behind narrow interfaces — OpenAI, Ollama, and deterministic fakes for tests |
| Observability | `app/observability/` | Nested trace spans for every ingestion/query; Langfuse or a zero-dependency logging tracer |
| Evaluation | `evaluation/` | A fixed demo corpus and hand-written question set run through the real pipeline; deterministic + optional RAGAS metrics |
| Core | `app/core/` | Settings, the error taxonomy (internal vs. user-facing), retry policy, API-key auth, rate limiting |
| Persistence | `db/migrations/` | Raw SQL schema — `documents`/`chunks` tables, pgvector column, no ORM migration framework |

## AI / ML components

| Role | Backend | Notes |
|---|---|---|
| Embeddings | OpenAI (`text-embedding-3-small` by default) | Swappable via `EmbeddingProvider` — see [Modular provider architecture](#modular-provider-architecture) |
| Generation (API) | OpenAI (`gpt-4o-mini` by default) | Used for non-sensitive requests when hybrid routing is enabled, or as the sole provider otherwise |
| Generation (local) | Ollama (`llama3.2:1b` by default) | Runs entirely on-device, no data leaves the machine; used for sensitive requests when hybrid routing is enabled |
| Reranking | A local Hugging Face cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Two-stage: vector search casts a wide net, the cross-encoder reorders it accurately; optional, no API key |
| Evaluation judge | OpenAI, via RAGAS | Only used for the LLM-judged metrics (faithfulness, relevance, context precision); optional |

Grounding is enforced two ways, not one: a system prompt instructs the model to answer only from the provided context, and — independent of whether the model cooperates — every citation marker in its output is checked against the context blocks that were actually sent, with anything out of range dropped and logged rather than passed through.

## Modular provider architecture

Every external AI/infrastructure dependency sits behind a narrow interface, not called directly from pipeline logic:

| Interface | Boundary | Concrete implementations |
|---|---|---|
| `EmbeddingProvider` | text → vector | OpenAI, deterministic fake (tests) |
| `LLMProvider` | messages → completion | OpenAI, Ollama, deterministic fake (tests) |
| `LLMRouter` | which `LLMProvider` handles this request | Sensitivity-based: local for flagged-sensitive documents, API otherwise, with an explicit no-silent-fallback rule for the sensitive path |
| `VectorStore` | store/search chunk embeddings | PostgreSQL + pgvector (SQL kept explicit and readable, not ORM-generated) |
| `Reranker` | reorder candidates by relevance | A local cross-encoder, deterministic fake (tests) |
| `Tracer` | record a span | Langfuse, or a zero-dependency logging tracer |

Swapping any one of these — a different embedding model, a different vector store, a different local model — means writing one new class against an existing interface, not touching pipeline code. This was a deliberate design constraint, not an incidental one: pipeline logic never imports a provider SDK directly.

## Evaluation

`evaluation/run_eval.py` ingests a fixed three-document demo corpus and runs 13 hand-written questions through the real `/query` pipeline — the same code path the API uses, not a separate approximation of it.

- **Deterministic metrics** (no judge model, no cost, no variance): retrieval hit-rate, retrieval precision@k, and refusal accuracy on questions the corpus can't answer.
- **LLM-judged metrics** (via RAGAS): faithfulness, answer relevance, and context precision. Implemented, code-reviewed, and integration-tested against the harness's mechanics — not yet run against a live judge model, since no OpenAI API key was configured during development.
- **Reranking comparison mode** (`--reranker compare`): runs the full evaluation twice, identical questions, reranking off vs. on, and prints a before/after delta on every metric — so "does reranking help" is answered with evidence for this corpus, not assumed from first principles.

```bash
python -m evaluation.run_eval --provider fake        # offline smoke test of the harness itself, no external calls
python -m evaluation.run_eval --skip-ragas            # real embeddings/LLM, deterministic metrics only
python -m evaluation.run_eval                          # real providers + RAGAS (needs OPENAI_API_KEY)
python -m evaluation.run_eval --reranker compare        # before/after reranking comparison
```

## Observability

Every ingestion and every query is wrapped in a trace, with a nested child span per pipeline step (`extract`, `clean_and_chunk`, `embed_batch`, `store` for ingestion; `retrieve`, `route_decision`, `generate` for queries). If `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, spans go to Langfuse; otherwise they're logged locally with zero external dependency — "no Langfuse configured" is a normal, fully-supported state, not a degraded one.

A document can be flagged `sensitive` at upload time. If any retrieved chunk comes from a sensitive document, that query's prompt and answer are redacted from the trace (only counts, model, and token usage are recorded) — the same flag also drives which model actually answers the request (see [AI / ML components](#ai--ml-components) above). A Langfuse outage never breaks a request: every SDK call is individually wrapped, and tracing failures fall back to a no-op rather than propagating. Spans are flushed on application shutdown so a clean stop doesn't rely solely on the SDK's own background export interval.

To see real traces in Langfuse Cloud: set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` in `.env` (see `.env.example` for how to get them), then run `python -m scripts.langfuse_demo` — it ingests two demo documents and runs two representative queries (one normal, one against a sensitive-flagged document) through the real pipeline, so the resulting traces show retrieval, generation with model/token usage, and redaction, exactly as production traffic would produce them.

## Security

- API key authentication (`X-API-Key` header, constant-time comparison), rate limiting (in-memory sliding window, per API key or IP), and a CORS allow-list — all opt-in and off by default for local development, matching every other optional feature in this project.
- **A fail-fast startup check**: the application refuses to start at all with `APP_ENV=production` if authentication is disabled, no API keys are configured, CORS is wildcarded, or Langfuse isn't configured — verified directly (the process exits with a clear error, not just documented behavior).
- Secrets are read from environment variables only; `.env` is git-ignored and was never part of any commit in this repository's history.
- Every exception that can cross the API boundary is a categorized, safe error — no stack traces, exception text, or internal file paths ever reach an HTTP response.
- Parameterized SQL throughout; explicit timeouts and classified (transient vs. permanent) bounded retries on every external call.

Full checklist, including what's deliberately deferred and why, in [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Testing

**116/116 tests passing.** Unit tests need nothing external. Integration tests run against a real Postgres/pgvector instance (via Docker) using deterministic fake embedding/LLM providers by default, so the suite needs no API key to run; tests that exercise a real local model (Ollama) or a real Hugging Face cross-encoder skip automatically when that dependency isn't available, rather than failing.

```bash
pytest tests/ -v
```

## Configuration & credentials

```bash
cp .env.example .env
```

| Variable | Required? | Purpose |
|---|---|---|
| `DATABASE_URL` | Has a working local default | Postgres connection string, matches `docker-compose.yml` |
| `OPENAI_API_KEY` | Required for real embeddings/generation/RAGAS | [platform.openai.com](https://platform.openai.com/account/api-keys) |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Only for hybrid/local inference | Requires a local Ollama install — see below |
| `HYBRID_ROUTING_ENABLED` | Optional, off by default | Sensitivity-based local/API routing |
| `RERANKER_ENABLED` | Optional, off by default | Requires the `rerank` extra (`sentence-transformers`) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Optional | [cloud.langfuse.com](https://cloud.langfuse.com) — omit to use the local logging tracer |
| `API_AUTH_ENABLED` / `API_KEYS` | Optional locally, **required in production** | See [Security](#security) |
| `RATE_LIMIT_ENABLED`, `CORS_ALLOWED_ORIGINS`, `MAX_UPLOAD_MB` | Optional | Deployment hardening — see [`DEPLOYMENT.md`](DEPLOYMENT.md) |

Full list with defaults and reasoning in [`.env.example`](.env.example). **`.env` is never committed** — verified as part of this repository's pre-publish audit.

## Running it locally

Requires Python 3.12+ and Docker.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,eval,rerank]"   # eval=RAGAS, rerank=local cross-encoder; both optional
cp .env.example .env
docker compose -f docker/docker-compose.yml up -d
.venv/bin/uvicorn app.main:app --reload
```

For hybrid local inference, also install and start Ollama:

```bash
brew install ollama
ollama serve &
ollama pull llama3.2:1b
```

```bash
curl -X POST http://localhost:8000/documents -F "file=@some-document.pdf"
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```

## Deployment status

Deployment *hardening* is complete and verified — authentication, rate limiting, CORS policy, configurable upload limits, and a fail-fast production config check that refuses to boot with unsafe settings. **The application has not been deployed to any cloud platform.** There is no live URL, and this README makes no claim that one exists. Actual cloud deployment is a deliberate next step, out of scope for this stage — see [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full readiness checklist and what's intentionally deferred until a real deployment target is chosen.

## Limitations

- No hybrid lexical+vector search — pure vector retrieval only; nothing in evaluation so far has shown a case where it's insufficient for this corpus.
- No OCR — scanned/image-only PDFs extract to empty text and fail ingestion cleanly.
- Fixed-size chunking, not structure-aware (no markdown-header-based or semantic chunking).
- Citation validation checks that a cited index is in range, not that the cited passage actually supports the claim next to it — that's what RAGAS faithfulness scoring is for.
- No multi-turn conversation memory — each query is independent.
- The evaluation corpus is intentionally small (3 documents, 13 questions) — enough to exercise the harness and catch regressions, not a statistically powerful benchmark.
- RAGAS's context-recall and answer-correctness metrics aren't wired in (they need a reference answer on every item; this dataset only has one on a few questions).
- Rate limiting is in-memory and single-process — correct for one instance, would under-count abuse across multiple instances. No distributed store (e.g. Redis) has been added, since no multi-instance deployment exists yet to justify it.
- Upload size is checked after the file is fully read into memory, not streamed; a request-body size limit at the reverse-proxy layer is the standard production mitigation and belongs in infrastructure config, not this codebase.
- Uploads are validated by file extension, not by sniffing actual file content — the document loaders' own parse-failure handling is the correctness backstop for a mismatched file.
- No malware scanning on uploads, and no per-user quotas/multi-tenant access control — `API_KEYS` is a flat set of static keys appropriate for a single-operator deployment, not a multi-tenant auth system.
- The small local model used by default (`llama3.2:1b`, ~1.3GB, chosen for a fast local dev loop) produces noticeably weaker answers than a frontier hosted model — the real, hands-on version of the privacy/cost-vs-quality trade-off hybrid inference exists to make concrete.

## Documentation

- [`DEPLOYMENT.md`](DEPLOYMENT.md) — full deployment-readiness checklist: what's done, what's deferred, and why.
- [`.env.example`](.env.example) — every configuration variable, with defaults and reasoning.
- Engineering decisions are documented inline, next to the code they explain, throughout `app/` — see [Engineering decisions](#engineering-decisions) below for the highlights, or the module docstrings themselves for full reasoning.

---

## Engineering decisions

Detailed reasoning behind the choices above, kept close to the code rather than in a separate design document.

**Ingestion & retrieval**
- Chunking is a fixed-size character window (1200 chars, 200 overlap), snapped to whitespace at both ends so no chunk starts or ends mid-word. See `app/ingestion/chunking.py`.
- No approximate vector index (IVFFlat/HNSW): tried during development and found to silently return zero results against a small dataset — a real, documented pgvector behavior at low row counts, where the default probe count can miss the only matching cluster. Exact sequential scan is used until data volume actually justifies an index. See `db/migrations/001_init.sql`.
- Retrieval SQL is written out explicitly (`app/retrieval/pgvector_store.py`) rather than generated by an ORM query builder, so the actual `<=>` cosine-distance query is directly readable.
- Ingestion is idempotent by `content_hash` (sha256 of raw bytes) — re-uploading identical content replaces its chunks rather than duplicating them. See `app/ingestion/service.py`.

**Generation & citations**
- The context budget sent to the model is deliberately smaller than the model's actual context window, and truncation always drops the least-similar chunks first. See `app/generation/context_builder.py`.
- If retrieval returns zero chunks, the LLM is never called — a fixed "not enough information" answer is returned directly, both because there's nothing to ground an answer in and because it's a free cost/latency optimization. See `app/generation/service.py`.
- Every citation the model emits is checked against the context blocks actually sent; an out-of-range index is dropped and logged, never passed through silently. See `app/generation/citations.py`.

**Evaluation**
- The evaluation dataset pins to `expected_source_uri`, not `reference_chunk_ids` — chunk IDs are server-generated UUIDs that change on every re-ingestion and can't live in a version-controlled dataset. Document-level matching is coarser but robust to chunking-strategy changes. See `evaluation/dataset/questions.yaml`.
- Deterministic metrics and LLM-judge metrics are kept as separate layers on purpose: the deterministic ones need no judge model, cost nothing, and aren't subject to judge variance; the judged ones are the only way to check whether a *generated answer* actually used its context faithfully. See `evaluation/metrics/retrieval.py` vs. `evaluation/metrics/ragas_metrics.py`.
- Installing `ragas` pulls in LangChain as RAGAS's own internal implementation detail (its judge-model/embedding wrapper interfaces) — this project's own pipeline code never imports LangChain. Kept as an optional `eval` extra so the core app doesn't require it.

**Observability**
- A trace *is* a root span, not a separate type — there's no distinct "Trace" object. The root span of an operation is the trace, and anything opened inside its `with` block becomes a nested child span automatically. This works because Langfuse's SDK is itself built on OpenTelemetry, which pushes a span onto the current context on entry — the concrete link between "OpenTelemetry" as a general concept and "Langfuse" as the specific tool used here.
- A Langfuse failure never breaks a request: every Langfuse SDK call is individually wrapped in try/except and falls back to a no-op span with a logged warning on failure. This is deliberately asymmetric with how spans treat pipeline exceptions (recorded as an error span, then re-raised, never swallowed) — tracing-infrastructure failures and business-logic failures are different failure domains. See `app/observability/langfuse_tracer.py`.
- Redaction reuses the same `sensitive` flag introduced for exactly this purpose: if any retrieved chunk comes from a sensitive document, the `generate` span's prompt and answer are replaced with a fixed redaction placeholder before being sent to Langfuse — the actual answer returned to the caller is never affected, only what gets traced.

**Reranking**
- Two-stage retrieval, not a replacement for vector search: `Retriever` widens the vector-search candidate set before handing it to the reranker, then narrows back down after. A cross-encoder scores a (query, chunk) pair jointly — far more accurate than comparing independent embeddings, but too expensive to run at corpus scale, hence the second stage over a small candidate pool. See `app/retrieval/reranker.py`.
- Reranking uses a local, open-weight model rather than an API call: it never generates text, so it carries none of the hybrid-routing privacy concerns, needs no API key, costs nothing per call, and is fast enough on CPU for the small candidate sets it actually operates on. `sentence-transformers` and its ~1GB torch dependency are gated behind an optional `rerank` extra, exactly like `ragas`. See `app/retrieval/cross_encoder_reranker.py`.
- Verified the real cross-encoder actually reranks correctly, not just that it runs: a test asserts it scores a genuinely relevant passage well above an unrelated one for a real question.

**Hybrid inference**
- A sensitive request that can't reach the local model hard-fails — it never silently falls back to the API. There is no except-clause around the local-provider call for a sensitive request at all; an unavailability error propagates straight up. This is the one rule in the whole routing design that isn't a judgment call: silently rerouting a flagged-sensitive request to a third-party API would defeat the feature while looking like it worked. The mirror case (non-sensitive, API down) does fall back to local by default — a defensible but genuinely different choice, since there's no privacy reason not to. See `app/providers/llm/router.py`.
- Routing is additive, not a fork: the query service accepts either a single provider (unchanged, original behavior) or a router (mutually exclusive) — existing single-provider behavior needed zero changes when routing is disabled, which it is by default.
- Ollama reuses the `openai` SDK client pointed at a different `base_url`, since Ollama's `/v1/chat/completions` endpoint is OpenAI-compatible — the one real behavioral difference (a 404 for an unpulled model) is classified as a permanent error with a specific, actionable message. See `app/providers/llm/ollama_provider.py`.

**Deployment hardening**
- Production doesn't just run insecurely — it refuses to run at all. A config-validation check executes at import time and raises before the app can serve a single request if production settings look unsafe. Verified directly: confirmed both that an unsafe production config fails to boot and that a fully-configured one starts cleanly.
- Rate limiting is deliberately in-memory, not Redis-backed — correct for a single process, and explicitly documented as under-counting abuse across multiple instances. Added because it's real, free insurance against one instance being hammered into a large API bill; not extended to a distributed store because no multi-instance deployment exists yet to justify that infrastructure. See `app/core/rate_limit.py`.
- The auth/rate-limit/CORS layer is attached centrally at app-construction time rather than scattered per-route — "which routes require auth" is answered by reading one file. The health check endpoint deliberately has neither dependency.

## License

[MIT](LICENSE) — free to use, modify, and adapt.
