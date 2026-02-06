import os
import pytest
from codex_proxy.config import Config


def test_config_defaults():
    c = Config(config_path="non_existent.json")
    assert c.port == 8765
    assert c.host == "0.0.0.0"


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("CODEX_PROXY_GEMINI_CLIENT_ID", "test-id")
    c = Config(config_path="non_existent.json")
    assert c.client_id == "test-id"
