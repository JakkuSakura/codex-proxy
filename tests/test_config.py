"""Unit tests for configuration management."""

import json
import os
import pytest
from unittest.mock import patch
from codex_proxy.config import Config, ConfigurationError


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Clear config file path to avoid loading real user config."""
    with patch("codex_proxy.config.Config.config_path", "/dev/null/nonexistent.json"):
        yield


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_host(self):
        cfg = Config()
        assert cfg.host == "127.0.0.1"

    def test_default_port(self):
        cfg = Config()
        assert 1 <= cfg.port <= 65535

    def test_default_port_value(self, monkeypatch):
        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)
        cfg = Config()
        assert cfg.port == 8765

    def test_default_urls(self):
        cfg = Config()
        assert cfg.z_ai_url.startswith("http")
        assert cfg.gemini_api_internal.startswith("http")
        assert cfg.gemini_api_public.startswith("http")

    def test_default_reasoning_config(self):
        cfg = Config()
        assert cfg.reasoning_effort == "medium"
        assert "effort_levels" in cfg.reasoning
        assert "default_effort" in cfg.reasoning

    def test_default_model_prefixes(self):
        cfg = Config()
        assert "gemini" in cfg.model_prefixes
        assert "glm" in cfg.model_prefixes
        assert "zai" in cfg.model_prefixes

    def test_default_compaction_model_is_none(self):
        cfg = Config()
        assert cfg.compaction_model is None

    def test_default_fallback_models_empty(self):
        cfg = Config()
        assert cfg.fallback_models == {}


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_invalid_port_string_raises_error(self, monkeypatch):
        monkeypatch.setenv("CODEX_PROXY_PORT", "invalid")
        with pytest.raises(ConfigurationError):
            Config()

    def test_port_out_of_range_raises_error(self, monkeypatch):
        monkeypatch.setenv("CODEX_PROXY_PORT", "99999")
        with pytest.raises(ConfigurationError):
            Config()

    def test_invalid_url_raises_error(self, monkeypatch):
        monkeypatch.setenv("CODEX_PROXY_ZAI_URL", "not-a-url")
        with pytest.raises(ConfigurationError):
            Config()


class TestConfigEnvOverrides:
    """Test environment variable overrides."""

    def test_port_from_env(self, monkeypatch):
        monkeypatch.setenv("CODEX_PROXY_PORT", "9999")
        cfg = Config()
        assert cfg.port == 9999

    def test_zai_url_from_env(self, monkeypatch):
        custom_url = "https://custom.z.ai/api"
        monkeypatch.setenv("CODEX_PROXY_ZAI_URL", custom_url)
        cfg = Config()
        assert cfg.z_ai_url == custom_url

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("CODEX_PROXY_LOG_LEVEL", "WARNING")
        cfg = Config()
        assert cfg.log_level == "WARNING"

    def test_debug_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("CODEX_PROXY_DEBUG", "false")
        cfg = Config()
        assert cfg.debug_mode is False


class TestConfigFileLoading:
    """Test loading configuration from file."""

    def test_load_config_from_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "port": 9090,
            "log_level": "info",
            "z_ai_api_key": "test-key",
            "compaction_model": "glm-5-turbo",
            "models": ["glm-5-turbo", "gemini-2.5-flash"],
        }))

        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)
        monkeypatch.delenv("CODEX_PROXY_LOG_LEVEL", raising=False)
        monkeypatch.delenv("CODEX_PROXY_ZAI_API_KEY", raising=False)
        monkeypatch.delenv("CODEX_PROXY_MODELS", raising=False)

        with patch("codex_proxy.config.Config.config_path", str(config_file)):
            cfg = Config()

        assert cfg.port == 9090
        assert cfg.log_level == "INFO"
        assert cfg.z_ai_api_key == "test-key"
        assert cfg.compaction_model == "glm-5-turbo"
        assert cfg.models == ["glm-5-turbo", "gemini-2.5-flash"]

    def test_env_takes_precedence_over_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "port": 9090,
            "log_level": "debug",
        }))

        monkeypatch.setenv("CODEX_PROXY_PORT", "7777")

        with patch("codex_proxy.config.Config.config_path", str(config_file)):
            cfg = Config()

        assert cfg.port == 7777  # env wins
        assert cfg.log_level == "DEBUG"  # file value loaded

    def test_missing_config_file_uses_defaults(self, monkeypatch):
        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)
        with patch("codex_proxy.config.Config.config_path", "/dev/null/nonexistent.json"):
            cfg = Config()
        assert cfg.port == 8765


class TestCompactionModelEmptyString:
    """Test that empty string compaction_model is treated as None."""

    def test_empty_string_compaction_model_becomes_none(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "compaction_model": "",
        }))

        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)
        monkeypatch.delenv("CODEX_PROXY_MODELS", raising=False)

        with patch("codex_proxy.config.Config.config_path", str(config_file)):
            cfg = Config()

        assert cfg.compaction_model is None

    def test_valid_compaction_model_preserved(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "compaction_model": "glm-5-turbo",
        }))

        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)
        monkeypatch.delenv("CODEX_PROXY_MODELS", raising=False)

        with patch("codex_proxy.config.Config.config_path", str(config_file)):
            cfg = Config()

        assert cfg.compaction_model == "glm-5-turbo"

    def test_null_compaction_model_stays_none(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "compaction_model": None,
        }))

        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)
        monkeypatch.delenv("CODEX_PROXY_MODELS", raising=False)

        with patch("codex_proxy.config.Config.config_path", str(config_file)):
            cfg = Config()

        assert cfg.compaction_model is None


class TestReasoningConfig:
    """Test reasoning configuration."""

    def test_reasoning_effort_levels(self):
        cfg = Config()
        levels = cfg.reasoning["effort_levels"]
        expected_levels = ["none", "minimal", "low", "medium", "high", "xhigh"]
        for level in expected_levels:
            assert level in levels
            assert "budget" in levels[level]
            assert "level" in levels[level]

    def test_reasoning_effort_from_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "reasoning_effort": "high",
            "reasoning": {
                "effort_levels": {
                    "high": {"budget": 99999, "level": "CUSTOM"},
                },
            },
        }))

        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)

        with patch("codex_proxy.config.Config.config_path", str(config_file)):
            cfg = Config()

        assert cfg.reasoning_effort == "high"
        assert cfg.reasoning["effort_levels"]["high"]["budget"] == 99999


class TestModelConfiguration:
    """Test model-related configuration."""

    def test_models_from_env(self, monkeypatch):
        models_str = "gemini-2.5-flash-lite,gemini-2.5-pro,glm-4"
        monkeypatch.setenv("CODEX_PROXY_MODELS", models_str)
        cfg = Config()
        assert len(cfg.models) == 3
        assert "glm-4" in cfg.models

    def test_custom_model_prefixes_from_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "model_prefixes": {
                "custom-model": "gemini",
                "my-prefix": "zai",
            },
        }))

        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)

        with patch("codex_proxy.config.Config.config_path", str(config_file)):
            cfg = Config()

        assert cfg.model_prefixes["custom-model"] == "gemini"
        assert cfg.model_prefixes["my-prefix"] == "zai"
        # Defaults still present
        assert "gemini" in cfg.model_prefixes
        assert "glm" in cfg.model_prefixes

    def test_fallback_models_from_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "fallback_models": {
                "glm-5-turbo": "glm-4.6",
            },
        }))

        monkeypatch.delenv("CODEX_PROXY_PORT", raising=False)

        with patch("codex_proxy.config.Config.config_path", str(config_file)):
            cfg = Config()

        assert cfg.fallback_models["glm-5-turbo"] == "glm-4.6"
