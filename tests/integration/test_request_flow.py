import json
import pytest
from unittest.mock import MagicMock, patch
from codex_proxy.server import RequestHandler
from codex_proxy.config import config


class MockServer:
    def __init__(self):
        pass


@pytest.fixture
def mock_handler_context():
    # Helper to instantiate a handler with mocked IO
    def _create_handler(method, path, body_dict):
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
        handler.path = "/v1/responses"
        handler.request_version = "HTTP/1.1"
        handler.sys_version = ""
        handler.server_version = "CodexProxyTest"
        handler.log_request = MagicMock()

        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.send_error = MagicMock()

        return handler, wfile

    yield _create_handler


def test_gemini_config_and_stream(mock_handler_context):
    create_handler = mock_handler_context
    payload = {
        "model": "test-gemini-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "temperature": 0.7,
    }

    handler, wfile = create_handler("POST", "/", payload)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}], "finishReason": "STOP"}}',
        b"data: [DONE]",
    ]

    mock_post_ctx = MagicMock()
    mock_post_ctx.__enter__.return_value = mock_response
    mock_post_ctx.__exit__.return_value = None

    with patch.object(
        handler.gemini_provider.session, "post", return_value=mock_post_ctx
    ) as mock_post:
        with patch.object(
            handler.gemini_provider.auth, "get_access_token", return_value="fake_token"
        ):
            with patch.object(
                handler.gemini_provider.auth, "get_project_id", return_value="fake_pid"
            ):
                handler.do_POST()

                # Verify Config Payload
                args, kwargs = mock_post.call_args
                req_json = json.loads(kwargs["data"])
                gen_config = req_json["request"]["generationConfig"]
                # Check Thinking Config (Critical for Gemini 3)
                assert gen_config["thinkingConfig"]["includeThoughts"] is True
                assert gen_config["thinkingConfig"]["thinkingLevel"] == "HIGH"
                assert gen_config["temperature"] == 0.7

                # Verify Stream Headers
                handler.send_header.assert_any_call(
                    "Content-Type", "text/event-stream; charset=utf-8"
                )


def test_gemini_non_stream_accumulation(mock_handler_context):
    create_handler = mock_handler_context
    payload = {
        "model": "test-gemini-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,  # Non-stream
        "_is_responses_api": True,
    }

    handler, wfile = create_handler("POST", "/", payload)

    mock_response = MagicMock()
    mock_response.status_code = 200
    # Provider treats everything as stream internally, so we simulate chunks
    mock_response.iter_lines.return_value = [
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Part 1"}]}}], "finishReason": ""}}',
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": " Part 2"}]}}], "finishReason": "STOP"}}',
        b"data: [DONE]",
    ]

    mock_post_ctx = MagicMock()
    mock_post_ctx.__enter__.return_value = mock_response
    mock_post_ctx.__exit__.return_value = None

    with patch.object(
        handler.gemini_provider.session, "post", return_value=mock_post_ctx
    ):
        with patch.object(
            handler.gemini_provider.auth, "get_access_token", return_value="fake_token"
        ):
            with patch.object(
                handler.gemini_provider.auth, "get_project_id", return_value="fake_pid"
            ):
                handler.do_POST()

                # Verify Headers
                handler.send_header.assert_any_call("Content-Type", "application/json")

                # Verify Aggregated Output
                # The provider writes the JSON string to wfile.wfile.write
                written_bytes = b"".join(
                    [call.args[0] for call in handler.wfile.write.call_args_list]
                )
                response_data = json.loads(written_bytes)

                # Responses API uses 'output' instead of 'choices'
                assert (
                    response_data["output"][0]["content"][0]["text"] == "Part 1 Part 2"
                )
                assert response_data["output"][0]["content"][0]["type"] == "output_text"
                assert response_data["status"] == "completed"
