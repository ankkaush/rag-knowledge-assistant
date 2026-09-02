"""
Shared retry policy for external dependency calls.

Concept: not all failures deserve a retry.
- Transient errors (network blip, timeout, HTTP 429 rate-limit, HTTP 5xx) are
  usually gone if you try again a moment later — retrying is the correct response.
- Permanent errors (bad API key, malformed request, context too long) will fail
  identically every time — retrying just burns time and, for paid APIs, money.

We use `tenacity` for the *mechanics* of retrying (backoff timing, attempt
counting) because that's a solved, well-tested problem not worth reimplementing.
What we own and must get right ourselves is the *classification*: which
exceptions actually count as transient for a given provider. That classification
function is the part worth reading carefully, not the tenacity call itself.
"""
from __future__ import annotations

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def is_transient_openai_error(exc: BaseException) -> bool:
    """
    Classify an OpenAI SDK exception as transient (retryable) or permanent (not).

    Transient: connection errors, timeouts, rate limits (429), server errors (5xx).
    Permanent: authentication failures (401), invalid request/params (400),
               e.g. input text too long for the embedding model's context window.
    """
    import openai

    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.InternalServerError):
        return True
    # openai.APIStatusError covers everything with an HTTP status; only retry 5xx.
    if isinstance(exc, openai.APIStatusError):
        return 500 <= exc.status_code < 600
    return False


def openai_retry(max_attempts: int):
    """
    Bounded retry with exponential backoff, applied only to transient errors.
    Shared across every OpenAI SDK call site (embeddings, chat completions) —
    the classification above isn't specific to either.
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception(is_transient_openai_error),
    )
