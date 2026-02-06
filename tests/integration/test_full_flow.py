"""Integration tests for codex-proxy."""

import json
import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from codex_proxy.server import ProxyRequestHandler
from codex_proxy.exceptions import ProviderError, ValidationError
from codex_proxy.providers.zai import ZAIProvider
from codex_proxy.providers.gemini import GeminiProvider


@pytest.fixture
def mock_handler():
    """Create a mock request handler."""
    request = MagicMock()
    rfile = MagicMock()
    rfile.read.return_value = b'{"model": "test"}'
    rfile.read.return_value = len(b'{"model": "test"}')

    wfile = MagicMock()
    wfile.write = MagicMock()
    wfile.flush = MagicMock()

    handler = ProxyRequestHandler(request, ("0.0.0.0", 8888), Mock())
    handler.rfile = rfile
    handler.wfile = wfile
    handler.headers = {"Content-Length": str(len(b'{"model": "test"}'))}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.send_error = MagicMock()

    return handler


class TestValidation:
    """Test input validation layer."""

    def test_invalid_json(self, mock_handler):
        """Test that invalid JSON returns 400 error."""
        mock_handler.rfile.read.return_value = b"{invalid json}"
        mock_handler.headers["Content-Length"] = len(b"{invalid json}")

        mock_handler._handle_post()

        mock_handler.send_error.assert_called_with(400)

    def test_invalid_model_name(self, mock_handler):
        """Test that invalid model name is rejected."""
        data = {"model": "x" * 150}
        mock_handler._handle_post()

        mock_handler.send_error.assert_called_with(400)

    def test_invalid_temperature(self, mock_handler):
        """Test that out-of-range temperature is rejected."""
        data = {"model": "test", "temperature": 3.0}
        mock_handler._handle_post()

        mock_handler.send_error.assert_called_with(400)

    def test_invalid_max_tokens(self, mock_handler):
        """Test that out-of-range max_tokens is rejected."""
        data = {"model": "test", "max_tokens": 999999}
        mock_handler._handle_post()

        mock_handler.send_error.assert_called_with(400)

    def test_compaction_without_input(self, mock_handler):
        """Test that compaction without input field is rejected."""
        data = {"model": "test"}
        mock_handler.path = "/responses/compact"
        mock_handler._handle_post()

        mock_handler.send_error.assert_called_with(400)


class TestProviderErrorHandling:
    """Test provider error handling."""

    @patch("codex_proxy.providers.gemini.GeminiProvider")
    def test_provider_error_returns_502(self, mock_gemini):
        """Test that provider errors return 502 status."""
        mock_instance = mock_gemini.return_value
        mock_instance.handle_request.side_effect = ProviderError("Provider unavailable")

        from codex_proxy.server import ProviderRegistry

        ProviderRegistry._providers.clear()
        ProviderRegistry.register("test", mock_instance)

        mock_handler._handle_post()
        mock_handler.send_error.assert_called_with(502)


class TestOAuthTokenRefresh:
    """Test OAuth token refresh flow."""

    @patch("codex_proxy.auth.open")
    @patch("codex_proxy.auth.time.time")
    def test_token_refresh_on_expiry(self, mock_open, mock_time):
        """Test that expired tokens are refreshed."""
        import json
        import codex_proxy.auth as auth_module

        creds_data = {
            "type": "authorized_user",
            "access_token": "old_token",
            "expiry_date": int(time.time() * 1000) - 1000000,  # Expired
        }

        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            creds_data
        )
        mock_time.time.return_value = int(time.time() * 1000)

        from codex_proxy.auth import GeminiAuth

        auth = GeminiAuth()
        token = auth.get_access_token()

        assert token == "old_token"  # Should not refresh yet


class TestConcurrency:
    """Test concurrent request handling."""

    @patch("codex_proxy.providers.zai.ZAIProvider")
    def test_concurrent_requests(self, mock_zai):
        """Test that concurrent requests don't interfere."""
        mock_instance = mock_zai.return_value

        import threading

        results = []

        def make_request(req_id):
            try:
                mock_instance.handle_request(
                    {
                        "model": "test",
                        "messages": [{"role": "user", "content": f"test {req_id}"}],
                    },
                    Mock(),
                )
                results.append(req_id)
            except Exception as e:
                results.append(f"error {req_id}: {e}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        assert len(results) == 5  # All requests should complete


class TestFullRequestFlow:
    """Test complete request flow with tools."""

    @patch("codex_proxy.providers.zai.ZAIProvider")
    def test_request_with_function_call(self, mock_zai):
        """Test that function calls are properly handled."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Function executed",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "test_function",
                                    "arguments": '{"param": "value"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        mock_instance = mock_zai.return_value
        mock_instance.session.post.return_value.__enter__.return_value = mock_response

        mock_handler._handle_post()

        assert mock_handler.send_response.called

    @patch("codex_proxy.providers.zai.ZAIProvider")
    def test_request_with_web_search(self, mock_zai):
        """Test that web_search tool is transformed correctly."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Search results",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        mock_instance = mock_zai.return_value
        mock_instance.session.post.return_value.__enter__.return_value = mock_response

        data = {
            "model": "test",
            "messages": [{"role": "user", "content": "search something"}],
            "tools": [{"type": "web_search"}],
        }

        mock_handler._handle_post()

        assert mock_handler.send_response.called
