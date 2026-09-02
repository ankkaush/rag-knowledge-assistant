"""
Typed application settings, loaded from environment variables (and .env in local dev).

Why pydantic-settings instead of scattered os.environ.get() calls:
- Validation happens once, at startup. A missing/malformed required setting fails
  immediately and loudly, instead of surfacing as a confusing error deep inside a
  request handler minutes later.
- Every setting has a declared type, so a bad value (e.g. a non-integer timeout)
  is caught before the app starts serving traffic.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5434/rag"
    db_pool_size: int = 5
    db_connect_timeout_seconds: int = 5

    # --- Embeddings ---
    embedding_provider: str = "openai"
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_timeout_seconds: int = 15
    embedding_max_retries: int = 3

    # --- Chunking ---
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200

    # --- Generation (LLM) ---
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3
    llm_max_tokens: int = 800
    llm_temperature: float = 0.2

    # --- Retrieval / context construction ---
    default_top_k: int = 5
    # Token budget for the RETRIEVED-CONTEXT portion of the prompt specifically
    # (not the model's full context window) — see app/generation/context_builder.py
    # for why this is a separate, smaller number.
    context_max_tokens: int = 3000

    # --- Local inference (Ollama) + hybrid routing ---
    # Ollama exposes an OpenAI-compatible endpoint at /v1, so OllamaChatProvider
    # reuses the openai SDK client pointed at a different base_url — see
    # app/providers/llm/ollama_provider.py.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout_seconds: int = 60
    ollama_max_retries: int = 2

    # Off by default: routing requires a local model server (Ollama) actually
    # running, which most environments (including CI) won't have. When
    # disabled, `sensitive` documents still get their traces redacted (Phase 4)
    # but generation always uses the single configured LLMProvider, exactly
    # as in Phases 2-5 — this is a fully-supported state, not a degraded one.
    # See app/providers/llm/router.py.
    hybrid_routing_enabled: bool = False
    # Non-sensitive request, API provider unavailable -> fall back to local
    # rather than failing outright. The mirror case (sensitive + local down)
    # is NOT configurable — it always hard-fails. See router.py's docstring.
    hybrid_allow_api_to_local_fallback: bool = True

    # --- Reranking ---
    # Off by default: requires the optional `rerank` extra (sentence-transformers
    # + torch, ~1GB) which most environments (including CI) shouldn't be forced
    # to install just to run the app. See app/retrieval/cross_encoder_reranker.py.
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_multiplier: int = 4

    # --- Observability ---
    # Empty by default -> get_tracer() falls back to LoggingTracer (see
    # app/observability/__init__.py). This means "no Langfuse configured" is
    # not an error state, it's the default local-dev/test state.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Deployment hardening (Phase 7) ---
    # All off/permissive by default — safe and unobtrusive for local dev and
    # tests, exactly like reranker_enabled/hybrid_routing_enabled. See
    # DEPLOYMENT.md for what MUST be turned on before a public deployment,
    # and validate_production_config() below for a fail-fast check of that.
    api_auth_enabled: bool = False
    # Comma-separated list of accepted keys. A caller sends one via the
    # X-API-Key header. Plaintext in .env is fine for local dev; in a real
    # deployment these should come from a secret manager, not a checked-in
    # or shell-exported .env file — see DEPLOYMENT.md.
    api_keys: str = ""

    rate_limit_enabled: bool = False
    rate_limit_requests_per_window: int = 60
    rate_limit_window_seconds: int = 60

    max_upload_mb: int = 20

    # Comma-separated allowed origins for browser cross-origin requests.
    # Empty = no cross-origin access at all (same-origin only) — the safe
    # default. "*" is accepted explicitly but should only be used for local
    # development, never in production (see validate_production_config).
    cors_allowed_origins: str = ""

    # --- App ---
    app_env: str = "local"
    log_level: str = "INFO"


settings = Settings()


def validate_production_config(s: Settings = settings) -> list[str]:
    """
    Fail-fast check: if APP_ENV=production but the deployment-hardening
    settings above look unsafe, return the list of problems (empty = OK).
    Called at startup in app/main.py — raises there, not here, so this
    function stays a pure, independently-testable check rather than a
    side-effecting one.
    """
    if s.app_env != "production":
        return []

    problems = []
    if not s.api_auth_enabled:
        problems.append("API_AUTH_ENABLED must be true in production (no auth = anyone can use your API/budget).")
    if s.api_auth_enabled and not s.api_keys.strip():
        problems.append("API_AUTH_ENABLED is true but API_KEYS is empty — no key would ever be accepted.")
    if s.cors_allowed_origins.strip() == "*":
        problems.append("CORS_ALLOWED_ORIGINS=* in production allows any website to call this API from a browser.")
    if not s.langfuse_public_key or not s.langfuse_secret_key:
        problems.append(
            "Langfuse is not configured in production — LoggingTracer will be used, "
            "meaning no durable observability. This is a warning-level concern, not a hard "
            "blocker, but is flagged since production without any real tracing defeats Phase 4."
        )
    return problems
