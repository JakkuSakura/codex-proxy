import json
import pytest
from unittest.mock import MagicMock, patch
from codex_proxy.server import RequestHandler


class MockServer:
    def __init__(self):
        pass


@pytest.fixture
def mock_handler_context():
    def _create_handler(body_dict):
        body_bytes = json.dumps(body_dict).encode("utf-8")
        request = MagicMock()
        rfile = MagicMock()
        rfile.read.return_value = body_bytes
        wfile = MagicMock()

        with patch("http.server.BaseHTTPRequestHandler.__init__", return_value=None):
            handler = RequestHandler(request, ("0.0.0.0", 8888), MockServer())

        handler.rfile = rfile
        handler.wfile = wfile
        handler.headers = {"Content-Length": str(len(body_bytes))}
        handler.request_version = "HTTP/1.1"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.send_error = MagicMock()
        return handler, wfile

    yield _create_handler


def test_responses_api_sse_sequence(mock_handler_context):
    payload = {
        "model": "test-gemini-model",
        "instructions": "You are a tester.",
        "input": "Test message",
        "stream": True,
    }
    handler, wfile = mock_handler_context(payload)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5, "totalTokenCount": 10}}}',
        b"data: [DONE]",
    ]

    mock_post_ctx = MagicMock()
    mock_post_ctx.__enter__.return_value = mock_response

    with patch.object(
        handler.gemini_provider.session, "post", return_value=mock_post_ctx
    ):
        with patch.object(
            handler.gemini_provider.auth, "get_access_token", return_value="token"
        ):
            with patch.object(
                handler.gemini_provider.auth, "get_project_id", return_value="pid"
            ):
                handler.path = "/v1/responses"
                handler.do_POST()

                # Capture all writes to wfile.wfile
                written_data = b"".join(
                    [call.args[0] for call in handler.wfile.write.call_args_list]
                ).decode()

                # Check for critical SSE events
                assert "event: response.created" in written_data
                assert "event: response.output_item.added" in written_data
                assert "event: response.output_text.delta" in written_data
                assert "event: response.done" in written_data

                # Check for content (orjson uses no spaces)
                assert '"text":"Hello"' in written_data
                assert '"type":"output_text"' in written_data
