"""Test fixtures and utilities."""

import pytest
from codex_proxy.server import ProviderRegistry


@pytest.fixture(autouse=True)
def reset_provider_registry():
    """Reset provider registry before each test to ensure isolation."""
    ProviderRegistry.initialize_from_config()
    yield
    # Clean up after test if needed
