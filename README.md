# RAG Knowledge Assistant — Phases 1-7 (complete)

Track B, Project 1. All 7 phases of the approved plan are implemented.
- **Phase 1**: document ingestion (load → clean → chunk → embed → store) and raw vector similarity search.
- **Phase 2**: the full RAG query pipeline — retrieve → construct context → generate → resolve citations.
- **Phase 3**: evaluation harness — deterministic retrieval/refusal metrics plus optional RAGAS LLM-judge metrics.
- **Phase 4**: observability — every ingestion and query is traced (Langfuse if configured, otherwise a zero-dependency logging tracer), with a privacy/redaction policy for sensitive documents.
- **Phase 5**: retrieval improvements — optional local cross-encoder reranking (two-stage: wide vector search → narrow accurate rerank) and `doc_type` metadata filtering, with a harness mode to measure whether reranking actually helps.
- **Phase 6**: hybrid inference — sensitive requests route to a local Ollama model, non-sensitive requests route to the API model, reusing the exact `sensitive` flag Phase 4 introduced for trace redaction.
- **Phase 7**: deployment hardening — API key auth, rate limiting, configurable upload limits, CORS policy, and a fail-fast production config check. See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full checklist.

## Setup

Requires Python 3.12+ and Docker.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,eval,rerank]"   # `eval`=RAGAS/langchain, `rerank`=sentence-transformers/torch (~1GB); both optional
cp .env.example .env   # fill in OPENAI_API_KEY for real embeddings/generation/RAGAS
docker compose -f docker/docker-compose.yml up -d
```

For hybrid routing (Phase 6), also install and start Ollama:

```bash
brew install ollama
ollama serve &                # or: brew services start ollama
ollama pull llama3.2:1b       # ~1.3GB, the default OLLAMA_MODEL
```

The Postgres container runs on host port **5434** (not 5432 — avoided a
collision with another local project's Postgres). `db/migrations/001_init.sql`
runs automatically on first container start via Postgres's
`docker-entrypoint-initdb.d` mechanism.

Run the app:

```bash
.venv/bin/uvicorn app.main:app --reload
```

Run tests (unit tests need nothing external; integration/API tests need the
Postgres container running — they use fake, deterministic embedding and LLM
providers, not real OpenAI calls, so no API key is needed to run the suite;
Ollama-dependent tests skip automatically if Ollama isn't running):

```bash
.venv/bin/pytest tests/ -v
```

Run the evaluation harness:

```bash
.venv/bin/python -m evaluation.run_eval --provider fake        # offline smoke test of the harness itself
.venv/bin/python -m evaluation.run_eval --skip-ragas            # real embeddings/LLM, deterministic metrics only
.venv/bin/python -m evaluation.run_eval                          # real providers + RAGAS (needs OPENAI_API_KEY)
.venv/bin/python -m evaluation.run_eval --reranker compare       # before/after: reranking off vs on (needs `rerank` extra for --provider real)
```

## What's implemented

- `POST /documents` — upload a `.txt`/`.md`/`.pdf` file, ingest it (load,
  clean, chunk, embed, store). Idempotent: re-uploading identical bytes
  updates the same document row rather than duplicating it.
- `POST /search` — raw vector similarity search over stored chunks (no LLM
  involved — useful for debugging retrieval quality in isolation).
- `POST /query` — the full RAG pipeline: retrieve chunks, build a numbered
  context block, generate a grounded answer, resolve citations against the
  actual context shown to the model. Returns `answer`, `citations[]`, and
  `retrieved_chunks[]`.
- `GET /health`
- `evaluation/run_eval.py` — ingests a fixed 3-document corpus
  (`evaluation/dataset/corpus/`), runs 13 hand-written questions
  (`evaluation/dataset/questions.yaml`) through the real `/query` pipeline,
  scores them, and writes a timestamped JSON report to `evaluation/reports/`.
- Tracing on every ingestion and query: `ingest_document` (with `extract`,
  `clean_and_chunk`, `embed_batch`, `store` child spans) and `query` (with
  `retrieve` and `generate` child spans). Uses Langfuse if
  `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, otherwise falls back to
  a logging-based tracer with zero external dependency.
- `sensitive` flag on document upload (`POST /documents`, form field) —
  redacts prompt/context/answer content from traces for that document's
  chunks, while leaving the actual returned answer untouched. Same flag Phase
  6 will use for local-vs-API routing.
- Optional reranking (`/query` and `/search`, `rerank: bool` request field,
  default `true` — a no-op unless `RERANKER_ENABLED=true`): a local
  Hugging Face cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reorders
  a widened candidate set down to `top_k`. `doc_type` filtering
  (`RetrievedChunkResponse`/`SearchRequest`/`QueryRequest`) alongside the
  existing `document_id` filter.
- `evaluation/run_eval.py --reranker {off,on,compare}` — `compare` runs the
  full eval twice (identical questions, reranking off vs on) and prints/saves
  a before/after delta on every metric, which is how Phase 5's "does this
  actually help" question gets answered with evidence instead of assumed.
- Hybrid routing (`HYBRID_ROUTING_ENABLED=true`, off by default): `/query`
  routes a sensitive-flagged request to a local Ollama model and a
  non-sensitive request to the configured API model, reusing the same
  `sensitive` flag from Phase 4. `QueryResponse.route` reports which path was
  actually used (`"api"` / `"local"` / `"local-fallback"`).
- Deployment hardening (all off/permissive by default, see `DEPLOYMENT.md`):
  API key auth (`API_AUTH_ENABLED` + `X-API-Key` header), in-memory rate
  limiting (`RATE_LIMIT_ENABLED`), configurable upload size limit
  (`MAX_UPLOAD_MB`), CORS policy (`CORS_ALLOWED_ORIGINS`), and a fail-fast
  startup check that refuses to run with `APP_ENV=production` and unsafe
  settings.

## Key decisions (see inline docstrings for full reasoning)

- **Chunking**: fixed-size character window (1200 chars, 200 overlap),
  snapped to whitespace at both ends. See `app/ingestion/chunking.py`.
- **No approximate vector index (IVFFlat/HNSW) yet**: an IVFFlat index was
  tried during Phase 1 and caused `similarity_search` to silently return zero
  results against a small dataset — a real, documented pgvector behavior at
  low row counts. Exact sequential scan is used instead until data volume
  justifies an index. See `db/migrations/001_init.sql`.
- **SQL is explicit, not ORM-generated**: `app/retrieval/pgvector_store.py`
  writes the pgvector `<=>` cosine-distance query directly so the actual
  retrieval mechanics are visible.
- **Retry policy**: bounded retry with backoff only for transient OpenAI
  errors (timeout, connection error, 429, 5xx); permanent errors (401, 400,
  context-too-long) fail immediately. Shared across embeddings and generation.
  See `app/core/retry.py`.
- **Error handling**: every exception crossing the API boundary is either a
  categorized `AppError` (safe `user_message` returned, `internal_detail`
  logged only) or caught by a catch-all handler and converted to a generic
  `internal_error` — no stack traces or internals ever reach a response. See
  `app/core/errors.py` and `app/main.py`.
- **Idempotency**: `documents.content_hash` (sha256 of raw bytes) is unique;
  re-ingesting identical content replaces chunks rather than duplicating them.
  See `app/ingestion/service.py`.
- **Context budget is separate from the model's context window**: retrieved
  context is capped at `CONTEXT_MAX_TOKENS` (default 3000), much smaller than
  the model's actual window, and truncation always drops the least-similar
  (highest-distance) chunks first. See `app/generation/context_builder.py`.
- **No-context short-circuit**: if retrieval returns zero chunks, the LLM is
  never called — a fixed "not enough information" answer is returned
  directly. Cheaper and more correct than asking a model to answer from
  nothing. See `app/generation/service.py`.
- **Citations are validated, not trusted**: the LLM is instructed to cite
  using `[N]` markers, but every citation in its output is checked against
  the actual context blocks sent — an out-of-range index is dropped and
  logged, never passed through. See `app/generation/citations.py`.
- **Evaluation dataset pins to `expected_source_uri`, not `reference_chunk_ids`**:
  chunk IDs are server-generated UUIDs that change on every re-ingestion and
  can't live in a version-controlled dataset. Document-level matching is
  coarser but robust to chunking-strategy changes — exactly what Phase 3
  needs to evaluate safely. See `evaluation/dataset/questions.yaml`.
- **Deterministic metrics and LLM-judge (RAGAS) metrics are separate layers**:
  retrieval hit-rate/precision and refusal-accuracy need no judge model, cost
  nothing, and aren't subject to LLM-judge variance; faithfulness/relevance/
  context-precision need a judge and are the only way to check whether a
  *generated answer* actually used its context faithfully. See
  `evaluation/metrics/retrieval.py` vs `evaluation/metrics/ragas_metrics.py`.
- **Installing `ragas` pulls in LangChain as RAGAS's own internal
  implementation detail** (its judge-model/embedding wrapper interfaces) —
  this project's own pipeline code never imports LangChain. Kept as an
  optional `eval` extra specifically so the core app doesn't require it.
- **A trace IS a root span, not a separate type**: there's no distinct
  "Trace" object in `app/observability/` — the root span of an operation
  (e.g. "query") is the trace, and anything opened inside its `with` block
  becomes a nested child span automatically. This works because Langfuse's
  SDK (v4+) is itself built on OpenTelemetry: `start_as_current_observation`
  pushes an OTel span onto the current context, so nesting needs no explicit
  parent-passing. This is the concrete link between "OpenTelemetry" as a
  general concept and "Langfuse" as the specific tool used here.
- **A Langfuse failure never breaks a request**: every Langfuse SDK call in
  `LangfuseTracer` is individually wrapped in try/except — if starting or
  ending a span fails, it falls back to a no-op span and logs a warning; the
  pipeline continues normally. This is deliberately asymmetric with how
  spans treat PIPELINE exceptions (recorded as an error span, then
  re-raised, never swallowed) — tracing-infrastructure failures and
  business-logic failures are different failure domains. See
  `app/observability/langfuse_tracer.py`.
- **Redaction policy reuses the `sensitive` flag introduced now for exactly
  this purpose**: if ANY retrieved chunk comes from a document flagged
  sensitive, the `generate` span's prompt and answer are replaced with a
  fixed redaction placeholder before being sent to Langfuse — the actual
  answer returned to the API caller is never affected, only what gets
  traced. Verified directly: see
  `tests/integration/test_query_service.py::test_sensitive_document_redacts_generate_span_but_answer_is_unaffected`.

- **Two-stage retrieval, not a replacement for vector search**: reranking
  never runs over the whole corpus — `Retriever` widens the vector-search
  candidate set (`top_k * rerank_candidate_multiplier`, capped at 50) *before*
  handing it to the reranker, then narrows back to `top_k` after. A
  cross-encoder scores a (query, chunk) pair jointly and is far more accurate
  than comparing independent embeddings, but too expensive to run at corpus
  scale — this is why it's a second stage over a small candidate pool, not a
  first-stage search method. See `app/retrieval/reranker.py`.
- **Reranking is a local, open-weight model, not an API call** — deliberately:
  it never generates text (no privacy/routing concerns the way Phase 6's LLM
  choice has), needs no API key, costs nothing per call, and is fast enough
  on CPU for the small candidate sets reranking actually operates on. This
  doubles as the project's first real exposure to running a Hugging Face
  model locally, ahead of Phase 6. See `app/retrieval/cross_encoder_reranker.py`.
- **`sentence-transformers` (and its ~1GB torch dependency) is an optional
  `rerank` extra**, exactly like `ragas`/`langchain` is an optional `eval`
  extra — `RERANKER_ENABLED=false` (the default) and `reranker=None` are
  normal, fully-supported states, not degraded ones. `get_reranker()` only
  imports `sentence_transformers` inside the branch that needs it.
- **Verified the real cross-encoder actually reranks correctly**, not just
  that it runs: `tests/integration/test_cross_encoder_reranker.py` loads the
  real model and asserts it scores "The Eiffel Tower is 330m tall" above an
  unrelated passage for the query "How tall is the Eiffel Tower?" — real
  relevance signal, not just "doesn't crash."
- **A sensitive request that can't reach the local model hard-fails — it
  never silently falls back to the API.** `LLMRouter` has no except-clause
  around the local-provider call for a sensitive request at all; an
  `UpstreamUnavailableError` propagates straight up. This is the one rule in
  the whole hybrid-routing design that isn't a judgment call: silently
  rerouting a flagged-sensitive request to a third-party API would defeat the
  feature while looking like it worked. The mirror case (non-sensitive, API
  down) DOES fall back to local by default (`HYBRID_ALLOW_API_TO_LOCAL_FALLBACK=true`)
  — a defensible but genuinely different choice, since there's no privacy
  reason not to. See `app/providers/llm/router.py`.
- **Routing is additive, not a fork**: `answer_query` takes either
  `llm_provider` (single-provider, Phases 2-5 behavior, unchanged) or
  `router` (mutually exclusive) — existing callers that only ever knew about
  a single `LLMProvider` needed zero changes in behavior when routing is
  disabled (the default). See `app/generation/service.py`.
- **Ollama reuses the `openai` SDK client, pointed at a different `base_url`**
  — Ollama's `/v1/chat/completions` endpoint is OpenAI-compatible, so
  `OllamaChatProvider` is nearly identical to `OpenAIChatProvider`. The one
  real behavioral difference: Ollama returns 404 for a model that hasn't been
  `ollama pull`ed, which is classified as a permanent error with a specific,
  actionable message (not OpenAI's generic "bad request" text) — the fix is
  genuinely different ("pull the model," not "fix your request"). See
  `app/providers/llm/ollama_provider.py`.
- **The route actually taken is surfaced in the API response**
  (`QueryResponse.route`), not just buried in traces — useful for a caller
  or the eval harness to confirm routing behaved as expected without having
  to go inspect Langfuse/logs for every request.
- **Production doesn't just run insecurely — it refuses to run at all.**
  `validate_production_config()` executes at import time in `app/main.py`
  and raises `RuntimeError` if `APP_ENV=production` with auth disabled, no
  API keys configured, wildcard CORS, or no Langfuse configured. Verified
  directly, not just described: manually confirmed both that an unsafe
  production config fails to boot and that a fully-configured one starts
  cleanly (see `DEPLOYMENT.md`'s Verification section).
- **Rate limiting is deliberately in-memory, not Redis-backed** — correct
  for a single process, and explicitly documented as under-counting abuse
  across multiple instances if this app is ever horizontally scaled. Added
  now because it's real, free insurance against one instance being hammered
  into a large API bill; NOT extended to a distributed store because no
  multi-instance deployment exists yet to justify that infrastructure — the
  approved plan explicitly warns against reaching for Redis ahead of a real
  need. See `app/core/rate_limit.py` and `DEPLOYMENT.md`.
- **The auth/rate-limit/CORS layer is attached centrally in `app/main.py`**
  (`dependencies=[Depends(require_api_key)]` on `include_router`, middleware
  added at app-construction time) rather than scattered per-route — "which
  routes require auth" is answered by reading one file. `/health`
  deliberately has neither dependency, since health checks must be reachable
  without a key and shouldn't count against a client's rate limit.

## Known limitations (Phase 1-7 scope — all phases now implemented)

- No hybrid/lexical search — the one retrieval strategy explicitly deferred
  throughout (see planning doc §9); nothing in the corpus/eval work so far
  has shown a case where pure vector search is insufficient.
- No OCR — scanned/image-only PDFs extract to empty text and fail ingestion.
- Chunking is fixed-size, not structure-aware (no markdown-header-based or
  semantic chunking).
- Citation validation is index-range checking only — it cannot detect a
  citation that's in-range but doesn't actually support the claim next to it
  (that's what RAGAS faithfulness is for).
- No conversation memory — each `/query` call is independent; no multi-turn context.
- The evaluation corpus is small (3 documents, 13 questions) — enough to
  exercise the harness mechanics and catch gross regressions, not a
  statistically powerful benchmark. Expanding it doesn't require code
  changes, just more entries in `questions.yaml` and `corpus/`.
- RAGAS's context-recall and answer-correctness metrics are not wired in —
  both need a `reference_answer` on every item; this dataset only has one on
  a few questions, and scoring the rest against an empty reference would be
  worse than not reporting the metric.
- **RAGAS was not run against the live API in this session** — no
  `OPENAI_API_KEY` was configured, so `ragas_metrics.py` is implemented and
  unit/integration-reviewable, but only exercised via `--provider fake`
  (which deliberately skips RAGAS, since judging a canned echo response is
  meaningless). Run `python -m evaluation.run_eval` with a real key before
  trusting its numbers.
- **LangfuseTracer was not exercised against a live Langfuse instance** — no
  `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` were configured, so all testing (and the
  end-to-end demo run) used `LoggingTracer`. The Langfuse SDK call shapes
  (`start_as_current_observation`, `.update(...)` fields) were verified
  against the installed SDK's actual signatures, not guessed, but the full
  round-trip to a live Langfuse project/dashboard has not been observed.
  Configure real keys and check the Langfuse UI before relying on it.
- Redaction is document-granularity and applies to the ENTIRE `generate`
  span's prompt/answer if any retrieved chunk is from a sensitive document —
  it does not selectively redact only the sensitive chunk's text within an
  otherwise-visible prompt. Simpler and more conservative; a mixed
  sensitive/non-sensitive context redacts the whole thing.
- **The `--reranker compare` run in this session (fake providers, 3-document
  corpus) showed a ZERO delta on every metric** — not a bug, an honest
  result: with `top_k=5` and a corpus this small, the widened candidate set
  already contains everything the plain vector search would have returned,
  so reordering it doesn't change what ends up in the top-K. This is exactly
  the kind of finding the comparison mode exists to surface — reranking's
  value is an empirical question that depends on corpus size and query
  difficulty, not a given. It has not yet been run with the real
  cross-encoder against real embeddings/a larger corpus, where a genuine
  before/after difference would be more likely to show up.
- Reranking has not been run through `--provider real` in this session — no
  `OPENAI_API_KEY` was configured. The real cross-encoder itself was verified
  directly (see above), just not through the full real-embeddings pipeline.
- **Hybrid routing was verified end-to-end with a REAL local model** —
  Ollama was installed and `llama3.2:1b` (~1.3GB) was pulled and run live in
  this session; `tests/integration/test_hybrid_routing.py` and
  `test_api.py::test_query_hybrid_routing_end_to_end_through_http` prove a
  sensitive-flagged document actually routes to and is answered by Ollama,
  not just that the routing logic looks right in isolation. The API-provider
  side of routing (non-sensitive → OpenAI) was only exercised with
  `FakeLLMProvider`, since no `OPENAI_API_KEY` was configured — the routing
  *decision* is proven for both directions; the real OpenAI call itself
  wasn't.
- `llama3.2:1b` is a genuinely small, fast model chosen for a responsive dev
  loop — its answer quality is noticeably weaker than a frontier hosted
  model (visible directly in the demo run: terse, sometimes just a bare
  citation marker). This is the real, hands-on version of the
  quality-vs-privacy/cost trade-off the project plan named abstractly —
  worth trying a larger local model (`ollama pull llama3.1:8b`) to feel the
  difference before treating `llama3.2:1b`'s output quality as representative.
- Ollama's server process was left running in the background for this
  session (`ollama serve`, or `brew services start ollama` to persist across
  reboots). Stop it with `brew services stop ollama` or by killing the
  process if not needed.
- **No streaming upload validation** — `await file.read()` loads the entire
  file into memory before the size check runs, so a large-enough request
  still costs memory before being rejected. The standard mitigation (a
  max-body-size limit at the reverse-proxy/ASGI-server layer) belongs in
  infrastructure config, not application code — deferred, see `DEPLOYMENT.md`.
- **No content-based upload validation** — files are gated by extension
  only (`.pdf`/`.txt`/`.md`), not by sniffing actual file content/MIME type.
  The loaders' own parse-failure handling (`app/ingestion/loaders/`) is the
  correctness backstop for a mismatched/malformed file; this is
  defense-in-depth that wasn't added, not a gap causing wrong behavior today.
- **No malware scanning on uploads** — would require external
  infrastructure (e.g. ClamAV or a cloud scanning API); out of scope for
  this project's threat model (a single operator uploading their own
  documents). Flagged as a real gap for a deployment accepting uploads from
  untrusted users.
- **`API_KEYS` is a flat set of static keys, not a user/account system** —
  appropriate for "don't let strangers spend my API budget," explicitly not
  multi-tenant access control, per-user quotas, or billing. A real
  multi-user product needs considerably more than this.
- **Rate limiting/auth were tested against the middleware and the app
  directly (TestClient), not against a real deployed/networked instance** —
  the logic is verified, but load-balancer interactions, real client IP
  extraction behind a proxy (`X-Forwarded-For` handling), and behavior under
  genuine concurrent load have not been observed.
