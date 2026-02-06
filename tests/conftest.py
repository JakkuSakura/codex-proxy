"""Test fixtures and utilities."""

import os
import sys

# Add src to sys.path to allow imports of codex_proxy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pytest
from codex_proxy.server import ProviderRegistry


@pytest.fixture(autouse=True)
def reset_provider_registry():
    """Reset provider registry before each test to ensure isolation."""
    ProviderRegistry.initialize_from_config()
    yield
    # Clean up after test if needed
