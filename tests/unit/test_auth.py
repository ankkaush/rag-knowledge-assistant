import pytest

from app.core.auth import require_api_key
from app.core.errors import AuthenticationError


def test_disabled_auth_always_passes(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "api_auth_enabled", False)
    require_api_key(x_api_key=None)  # no raise


def test_enabled_auth_rejects_missing_key(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "api_auth_enabled", True)
    monkeypatch.setattr(config.settings, "api_keys", "secret1,secret2")
    with pytest.raises(AuthenticationError):
        require_api_key(x_api_key=None)


def test_enabled_auth_rejects_wrong_key(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "api_auth_enabled", True)
    monkeypatch.setattr(config.settings, "api_keys", "secret1,secret2")
    with pytest.raises(AuthenticationError):
        require_api_key(x_api_key="wrong-key")


def test_enabled_auth_accepts_valid_key(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "api_auth_enabled", True)
    monkeypatch.setattr(config.settings, "api_keys", "secret1, secret2")
    require_api_key(x_api_key="secret2")  # no raise despite whitespace in config
