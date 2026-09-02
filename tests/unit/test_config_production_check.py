from app.core.config import Settings, validate_production_config


def _prod_settings(**overrides) -> Settings:
    base = dict(
        app_env="production",
        api_auth_enabled=True,
        api_keys="a-real-key",
        cors_allowed_origins="https://example.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
    )
    base.update(overrides)
    return Settings(**base)


def test_non_production_env_always_passes():
    s = Settings(app_env="local", api_auth_enabled=False)
    assert validate_production_config(s) == []


def test_fully_configured_production_passes():
    assert validate_production_config(_prod_settings()) == []


def test_production_without_auth_fails():
    problems = validate_production_config(_prod_settings(api_auth_enabled=False))
    assert any("API_AUTH_ENABLED" in p for p in problems)


def test_production_with_auth_enabled_but_no_keys_fails():
    problems = validate_production_config(_prod_settings(api_keys=""))
    assert any("API_KEYS is empty" in p for p in problems)


def test_production_with_wildcard_cors_fails():
    problems = validate_production_config(_prod_settings(cors_allowed_origins="*"))
    assert any("CORS_ALLOWED_ORIGINS" in p for p in problems)


def test_production_without_langfuse_flags_but_does_not_have_to_be_the_only_problem():
    problems = validate_production_config(_prod_settings(langfuse_public_key="", langfuse_secret_key=""))
    assert any("Langfuse" in p for p in problems)
