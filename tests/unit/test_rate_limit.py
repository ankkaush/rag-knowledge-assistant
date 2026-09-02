import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.core import config
from app.core.rate_limit import RateLimitMiddleware


def _make_app():
    async def endpoint(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/thing", endpoint), Route("/health", endpoint)])
    app.add_middleware(RateLimitMiddleware)
    return app


@pytest.fixture(autouse=True)
def enable_rate_limiting(monkeypatch):
    monkeypatch.setattr(config.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(config.settings, "rate_limit_requests_per_window", 3)
    monkeypatch.setattr(config.settings, "rate_limit_window_seconds", 60)


def test_requests_within_limit_pass():
    client = TestClient(_make_app())
    for _ in range(3):
        assert client.get("/thing").status_code == 200


def test_requests_over_limit_are_rejected():
    client = TestClient(_make_app())
    for _ in range(3):
        client.get("/thing")
    resp = client.get("/thing")
    assert resp.status_code == 429
    assert resp.json()["error_code"] == "rate_limited"


def test_health_endpoint_is_never_rate_limited():
    client = TestClient(_make_app())
    for _ in range(10):
        assert client.get("/health").status_code == 200


def test_different_clients_have_independent_limits():
    client = TestClient(_make_app())
    for _ in range(3):
        assert client.get("/thing", headers={"X-API-Key": "client-a"}).status_code == 200
    # client-a is now at its limit, but a different key gets its own budget
    assert client.get("/thing", headers={"X-API-Key": "client-b"}).status_code == 200
    assert client.get("/thing", headers={"X-API-Key": "client-a"}).status_code == 429


def test_disabled_rate_limiting_never_blocks(monkeypatch):
    monkeypatch.setattr(config.settings, "rate_limit_enabled", False)
    client = TestClient(_make_app())
    for _ in range(10):
        assert client.get("/thing").status_code == 200
