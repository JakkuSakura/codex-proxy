"""Integration tests for the proxy server."""

import json
import pytest
from unittest.mock import MagicMock, patch
from codex_proxy.server import ProxyRequestHandler, ProviderRegistry


class MockRequest:
    """Mock HTTP request."""

    def __init__(self):
        self.rfile = None
        self.wfile = None


class MockServer:
    """Mock HTTP server."""

    pass


@pytest.fixture
def reset_registry():
    """Reset provider registry before each test."""
    ProviderRegistry.initialize_from_config()
    yield


def create_handler(body_dict, path="/v1/responses"):
    """Helper to create a handler with mocked I/O."""
    body_bytes = json.dumps(body_dict).encode("utf-8")

    request = MockRequest()
    rfile = MagicMock()
    rfile.read.return_value = body_bytes

    wfile = MagicMock()

    with patch("http.server.BaseHTTPRequestHandler.__init__", return_value=None):
        handler = ProxyRequestHandler(request, ("0.0.0.0", 8888), MockServer())

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


class TestRequestRouting:
    """Test request routing to providers."""

    def test_gemini_request_routing(self, reset_registry):
        """Test that gemini model routes to Gemini provider."""
        handler, wfile = create_handler(
            {
                "model": "gemini-2.5-flash-lite",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        with patch(
            "codex_proxy.providers.gemini.GeminiProvider.handle_request"
        ) as mock_handle:
            with patch(
                "codex_proxy.providers.zai.ZAIProvider.handle_request"
            ) as mock_zai:
                handler._handle_post()
                mock_handle.assert_called_once()
                mock_zai.assert_not_called()

    def test_glm_request_routing(self, reset_registry):
        """Test that glm model routes to ZAI provider."""
        handler, wfile = create_handler(
            {"model": "glm-4", "messages": [{"role": "user", "content": "Hello"}]}
        )

        with patch(
            "codex_proxy.providers.gemini.GeminiProvider.handle_request"
        ) as mock_gemini:
            with patch(
                "codex_proxy.providers.zai.ZAIProvider.handle_request"
            ) as mock_handle:
                handler._handle_post()
                mock_handle.assert_called_once()
                mock_gemini.assert_not_called()

    def test_zai_request_routing(self, reset_registry):
        """Test that zai model routes to ZAI provider."""
        handler, wfile = create_handler(
            {"model": "zai-custom", "messages": [{"role": "user", "content": "Hello"}]}
        )

        with patch(
            "codex_proxy.providers.gemini.GeminiProvider.handle_request"
        ) as mock_gemini:
            with patch(
                "codex_proxy.providers.zai.ZAIProvider.handle_request"
            ) as mock_handle:
                handler._handle_post()
                mock_handle.assert_called_once()
                mock_gemini.assert_not_called()


class TestRequestValidation:
    """Test request validation in the server."""

    def test_invalid_json_returns_400(self):
        """Test that invalid JSON returns 400 error."""
        body_bytes = b"not valid json"

        request = MockRequest()
        rfile = MagicMock()
        rfile.read.return_value = body_bytes
        wfile = MagicMock()

        with patch("http.server.BaseHTTPRequestHandler.__init__", return_value=None):
            handler = ProxyRequestHandler(request, ("0.0.0.0", 8888), MockServer())

        handler.rfile = rfile
        handler.wfile = wfile
        handler.headers = {"Content-Length": str(len(body_bytes))}
        handler.path = "/v1/responses"  # Manually set path
        handler.send_error = MagicMock()

        handler.do_POST()  # Call do_POST to exercise error handling
        handler.send_error.assert_called_once()
        call_args = handler.send_error.call_args[0]
        assert call_args[0] == 400

    def test_empty_body_returns_400(self):
        """Test that empty body returns 400 error."""
        handler, wfile = create_handler({})

        handler.headers = {"Content-Length": "0"}
        handler._handle_post()
        handler.send_error.assert_called_once()
        call_args = handler.send_error.call_args[0]
        assert call_args[0] == 400

    def test_invalid_endpoint_returns_404(self):
        """Test that invalid endpoint returns 404 error."""
        handler, wfile = create_handler({"model": "test"}, "/invalid")
        handler._handle_post()
        handler.send_error.assert_called_once()
        call_args = handler.send_error.call_args[0]
        assert call_args[0] == 404


class TestCompactionRequests:
    """Test compaction request handling."""

    def test_compact_route_uses_compaction_model(self, reset_registry):
        """Test that compaction requests use configured compaction model."""
        handler, wfile = create_handler(
            {
                "model": "glm-4",  # User's selected model
                "input": "Long conversation...",
                "instructions": "Summarize",
            },
            "/v1/responses/compact",
        )

        with patch(
            "codex_proxy.providers.zai.ZAIProvider.handle_compact"
        ) as mock_compact:
            with patch(
                "codex_proxy.providers.gemini.GeminiProvider.handle_request"
            ) as mock_request:
                handler._handle_post()
                # Should call handle_compact, not handle_request
                mock_compact.assert_called_once()
                mock_request.assert_not_called()

    def test_compact_validation_error(self, reset_registry):
        """Test that invalid compact request returns 400."""
        handler, wfile = create_handler(
            {"model": "test", "input": "content"}, "/v1/responses/compact"
        )

        handler.do_POST()  # Call do_POST to exercise error handling
        handler.send_error.assert_called_once()
        call_args = handler.send_error.call_args[0]
        assert call_args[0] == 400


class TestHeaders:
    """Test that headers are properly forwarded."""

    def test_context_headers_preserved(self, reset_registry):
        """Test that context headers are preserved in request data."""
        handler, wfile = create_handler(
            {
                "model": "gemini-2.5-flash-lite",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        handler.headers = {
            "Content-Length": "100",
            "session_id": "session-123",
            "x-openai-subagent": "true",
            "x-codex-turn-state": "state-456",
            "x-codex-personality": "helpful",
        }

        with patch(
            "codex_proxy.providers.gemini.GeminiProvider.handle_request"
        ) as mock_handle:
            handler._handle_post()
            call_args = mock_handle.call_args[0]
            data = call_args[0]
            assert "_headers" in data
            assert data["_headers"]["session_id"] == "session-123"
            assert data["_headers"]["x-openai-subagent"] == "true"
            assert data["_headers"]["x-codex-turn-state"] == "state-456"
            assert data["_headers"]["x-codex-personality"] == "helpful"


class TestResponsesAPI:
    """Test Responses API specific handling."""

    def test_responses_api_flag_set(self, reset_registry):
        """Test that _is_responses_api flag is set for normal requests."""
        handler, wfile = create_handler(
            {
                "model": "gemini-2.5-flash-lite",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        with patch(
            "codex_proxy.providers.gemini.GeminiProvider.handle_request"
        ) as mock_handle:
            handler._handle_post()
            call_args = mock_handle.call_args[0]
            data = call_args[0]
            assert data.get("_is_responses_api") is True

    def test_compact_no_responses_api_flag(self, reset_registry):
        """Test that _is_responses_api is not set for compact requests."""
        handler, wfile = create_handler(
            {
                "model": "gemini-2.5-flash-lite",
                "input": [{"role": "user", "content": "content"}],
                "instructions": "Summarize",
            },
            "/v1/responses/compact",
        )

        # Force compaction model to ensure Gemini provider is selected
        with patch(
            "codex_proxy.server.config.compaction_model", "gemini-2.5-flash-lite"
        ):
            with patch(
                "codex_proxy.providers.gemini.GeminiProvider.handle_compact"
            ) as mock_handle:
                handler._handle_post()
                call_args = mock_handle.call_args[0]
                data = call_args[0]
                assert (
                    data.get("_is_responses_api") is None
                    or data.get("_is_responses_api") is False
                )


class TestErrorHandling:
    """Test error handling in the server."""

    def test_validation_error_returns_400(self):
        """Test that validation errors return 400."""
        handler, wfile = create_handler(
            {
                "model": "x" * 101,  # Too long
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        handler.do_POST()  # Call do_POST to exercise error handling
        handler.send_error.assert_called_once()
        call_args = handler.send_error.call_args[0]
        assert call_args[0] == 400

    def test_provider_error_returns_502(self, reset_registry):
        """Test that provider errors return 502."""
        from codex_proxy.exceptions import ProviderError

        handler, wfile = create_handler(
            {
                "model": "gemini-2.5-flash-lite",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        # Patch the actual provider instance's method
        provider = ProviderRegistry.get_provider("gemini-2.5-flash-lite")
        with patch.object(
            provider, "handle_request", side_effect=ProviderError("Provider failed")
        ):
            handler.do_POST()
            handler.send_error.assert_called_once()
            call_args = handler.send_error.call_args[0]
            assert call_args[0] == 502

    def test_unexpected_error_returns_500(self):
        """Test that unexpected errors return 500."""
        handler, wfile = create_handler(
            {
                "model": "gemini-2.5-flash-lite",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        with patch(
            "codex_proxy.normalizer.RequestNormalizer.normalize"
        ) as mock_normalize:
            mock_normalize.side_effect = Exception("Unexpected error")
            handler.do_POST()
            handler.send_error.assert_called_once()
            call_args = handler.send_error.call_args[0]
            assert call_args[0] == 500


class TestEndpoints:
    """Test different endpoint paths."""

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/responses",
            "/responses",
        ],
    )
    def test_valid_responses_endpoints(self, reset_registry, path):
        """Test that valid response endpoints are accepted."""
        handler, wfile = create_handler(
            {
                "model": "gemini-2.5-flash-lite",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            path=path,
        )

        with patch(
            "codex_proxy.providers.gemini.GeminiProvider.handle_request"
        ) as mock_handle:
            handler._handle_post()
            mock_handle.assert_called_once()

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/responses/compact",
            "/responses/compact",
        ],
    )
    def test_valid_compact_endpoints(self, reset_registry, path):
        """Test that valid compact endpoints are accepted."""
        handler, wfile = create_handler(
            {
                "model": "gemini-2.5-flash-lite",
                "input": [{"role": "user", "content": "content"}],
                "instructions": "Summarize",
            },
            path=path,
        )

        # Force compaction model to ensure Gemini provider is selected
        with patch(
            "codex_proxy.server.config.compaction_model", "gemini-2.5-flash-lite"
        ):
            with patch(
                "codex_proxy.providers.gemini.GeminiProvider.handle_compact"
            ) as mock_handle:
                handler._handle_post()
                mock_handle.assert_called_once()
