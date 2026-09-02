# Deployment Checklist

This is the Phase 7 checklist promised in the project plan (§14): every item
from the architecture review's "local dev vs. production" list, marked
**Done**, **Deferred (with reason)**, or **Not applicable**.

## Security — baseline (in place since Phase 1, not Phase 7)

These were never optional or deferred — they're correctness properties of
the code, not deployment configuration:

- [x] Secrets via environment variables, `.env` gitignored, never committed.
- [x] Parameterized SQL everywhere (`sqlalchemy.text()` with bound params — see `app/retrieval/pgvector_store.py`).
- [x] Input validation at the API boundary (Pydantic schemas, `app/api/schemas.py`).
- [x] Internal vs. user-facing error separation — no stack traces, exception
      text, file paths, or credentials ever reach an HTTP response
      (`app/core/errors.py`, `app/main.py`'s exception handlers).
- [x] Explicit timeouts on every external call (DB, OpenAI, Ollama, Langfuse).
- [x] Bounded retries, transient-vs-permanent classified per provider (`app/core/retry.py`).

## Deployment hardening — Phase 7

- [x] **Authentication.** `API_AUTH_ENABLED` + `API_KEYS` (`X-API-Key` header,
      constant-time comparison). Off by default (matches every other
      opt-in flag in this project); `validate_production_config()` refuses
      to boot with `APP_ENV=production` and auth disabled. See `app/core/auth.py`.
- [x] **Rate limiting.** In-memory sliding-window limiter, per API key (or
      IP if no key), `RATE_LIMIT_ENABLED` + `RATE_LIMIT_REQUESTS_PER_WINDOW`
      + `RATE_LIMIT_WINDOW_SECONDS`. See `app/core/rate_limit.py` and the
      **deferred** note below on multi-instance correctness.
- [x] **Upload limits.** `MAX_UPLOAD_MB` (default 20), enforced in
      `app/api/ingestion.py`. See the **deferred** note below on streaming.
- [x] **CORS.** `CORS_ALLOWED_ORIGINS` — empty by default (no cross-origin
      browser access at all); `validate_production_config()` refuses `*` in
      production.
- [x] **Fail-fast production config check.** `validate_production_config()`
      runs at import time in `app/main.py`; the app process will not start
      with `APP_ENV=production` and unsafe settings (auth off, no keys
      configured, wildcard CORS, or no Langfuse configured). This is what
      makes "Phase 7 is the first point security starts" structurally
      impossible — production without hardening doesn't run, it doesn't
      just run insecurely.
- [x] **Production secrets should come from a secret manager, not `.env`.**
      No code change needed — `pydantic-settings` reads from process
      environment variables already; `.env` is a local-dev convenience only.
      In a real deployment, inject `OPENAI_API_KEY`, `DATABASE_URL`,
      `API_KEYS`, `LANGFUSE_SECRET_KEY`, etc. via the platform's secret
      manager (e.g. a cloud provider's secrets service, or Docker/Kubernetes
      secrets) so they never live in a file on disk or in shell history.

## Deferred, with reasons (not oversights)

- **Multi-instance rate limiting.** The current limiter is in-process memory
  — correct for a single instance, silently *undercounts* abuse across
  multiple instances (`N` instances effectively allow `N × limit`). Fixing
  this needs a shared store (Redis is the standard choice) to coordinate
  counts. Not added now: there is no multi-instance deployment yet to
  justify it, and the approved plan explicitly warns against reaching for
  Redis/Kafka/Temporal-style infrastructure ahead of a real need. Revisit if
  this app is ever actually deployed as more than one process.
- **Streaming upload validation.** `await file.read()` reads the entire
  upload into memory before the size check runs (`app/api/ingestion.py`) —
  a large-enough request still costs memory before being rejected. A
  reverse-proxy/ASGI-server-level max-body-size limit (e.g. in Nginx,
  Caddy, or the ASGI server's own config) is the standard mitigation and
  belongs at the infrastructure layer, not application code — deferred to
  actual deployment configuration, not implemented here.
- **MIME-type/content sniffing beyond file extension.** Uploads are gated by
  file extension only (`.pdf`/`.txt`/`.md`); a file with a spoofed extension
  isn't caught. Low risk given the loaders themselves fail cleanly on
  malformed content (see `app/ingestion/loaders/`), but a stricter
  deployment could add `python-magic`-based content sniffing. Not added:
  the loaders' own parse-failure handling already provides a correctness
  backstop; this would be defense-in-depth, not a gap that currently causes
  incorrect behavior.
- **Malware scanning on uploads.** Out of scope for a project at this scale
  — would require an external scanning service (e.g. ClamAV or a cloud
  provider's scanning API), genuine new infrastructure. Flagged as a real
  gap for a deployment that accepts uploads from untrusted users; this
  project's threat model (a single operator uploading their own documents)
  doesn't currently require it.
- **Prompt injection via ingested documents.** Partially mitigated (the
  system prompt instructs the model to treat context as data, not
  instructions — `app/generation/prompts.py`) but not fully solved; this is
  a known, industry-wide open problem, not something this project claims to
  have closed. Documented here rather than silently assumed away.
- **Per-user quotas / multi-tenant access control.** `API_KEYS` is a flat
  set of static keys, not a user/account system — appropriate for a
  single-operator deployment ("don't let strangers spend my API budget"),
  explicitly not a multi-tenant product auth system. A real multi-user
  product would need considerably more (per-user keys, usage tracking,
  billing) — out of scope for this project.

## Verification

Verified directly in this session, not just described:
- `validate_production_config()` — unit-tested (`tests/unit/test_config_production_check.py`)
  and manually confirmed to refuse to boot (`APP_ENV=production` with
  defaults raises `RuntimeError` at import time; a fully-configured
  production env boots cleanly).
- Auth — unit tests (`tests/unit/test_auth.py`) plus full HTTP-boundary
  tests (`tests/integration/test_deployment_hardening.py`): a request
  without a key gets `401`, `/health` is never gated, a valid key passes.
- Rate limiting — tested directly against the middleware
  (`tests/unit/test_rate_limit.py`): requests within the window pass,
  over-limit requests get `429`, different clients get independent budgets,
  `/health` is exempt.
- Upload limit — configurable and tested
  (`tests/integration/test_deployment_hardening.py::test_upload_limit_is_configurable`).
