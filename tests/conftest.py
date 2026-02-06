"""
Shared pytest fixtures for codex-proxy tests.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from codex_proxy.server import RequestHandler


class MockServer:
    """Mock HTTP server for testing."""

    def __init__(self):
        pass


@pytest.fixture
def mock_handler():
    """Create a mock request handler with mocked I/O."""

    def _create_handler(body_dict, path="/v1/responses"):
        body_bytes = json.dumps(body_dict).encode("utf-8")

        request = MagicMock()
        rfile = MagicMock()
        rfile.read.return_value = body_bytes

        wfile = MagicMock()

        from http.server import BaseHTTPRequestHandler

        with patch.object(BaseHTTPRequestHandler, "__init__", return_value=None):
            handler = RequestHandler(request, ("0.0.0.0", 8888), MockServer())

        handler.rfile = rfile
        handler.wfile = wfile
        handler.headers = {"Content-Length": str(len(body_bytes))}
        handler.request_version = "HTTP/1.1"
        handler.path = path

        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.send_error = MagicMock()

        return handler, wfile

    return _create_handler


@pytest.fixture
def reset_config():
    """Reset config to defaults after each test."""
    from codex_proxy.config import Config

    original_instance = None

    # Store original config instance
    import codex_proxy.config as config_module

    if hasattr(config_module, "config"):
        original_instance = config_module.config

    yield

    # Restore or reload config
    if original_instance:
        config_module.config = original_instance
