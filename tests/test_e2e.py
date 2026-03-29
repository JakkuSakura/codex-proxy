"""End-to-end tests: start the proxy server, send real HTTP requests, verify responses."""

import json
import os
import socket
import threading
import time

import pytest
import requests

from codex_proxy.server import ThreadedHTTPServer, ProxyRequestHandler
from codex_proxy.config import config


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def proxy_server():
    """Start proxy on a random port, yield base URL, shut down after tests."""
    port = _free_port()
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, ProxyRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    # Give the server a moment to start
    time.sleep(0.3)
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    httpd.shutdown()


# ---------------------------------------------------------------------------
# UI / config endpoints
# ---------------------------------------------------------------------------

class TestUIEndpoints:
    def test_root_returns_html(self, proxy_server):
        r = requests.get(f"{proxy_server}/", timeout=5)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_ui_returns_html(self, proxy_server):
        r = requests.get(f"{proxy_server}/ui", timeout=5)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_config_get_returns_html(self, proxy_server):
        r = requests.get(f"{proxy_server}/config", timeout=5)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_config_post_returns_json(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/config",
            json={"port": 9999},
            timeout=5,
        )
        assert r.status_code == 200
        body = r.json()
        assert "port" in body

    def test_404_for_unknown_endpoint(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/unknown",
            json={},
            timeout=5,
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Validation (400 errors)
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_json_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses",
            data="not json",
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        assert r.status_code == 400

    def test_empty_body_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses",
            data=b"",
            headers={"Content-Length": "0"},
            timeout=5,
        )
        assert r.status_code == 400

    def test_model_name_too_long_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={"model": "x" * 101, "messages": [{"role": "user", "content": "hi"}]},
            timeout=5,
        )
        assert r.status_code == 400

    def test_invalid_temperature_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 5,
            },
            timeout=5,
        )
        assert r.status_code == 400

    def test_invalid_max_tokens_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0,
            },
            timeout=5,
        )
        assert r.status_code == 400

    def test_invalid_stream_flag_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": "yes",
            },
            timeout=5,
        )
        assert r.status_code == 400

    def test_invalid_message_role_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "hacker", "content": "hi"}],
            },
            timeout=5,
        )
        assert r.status_code == 400

    def test_compact_missing_input_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses/compact",
            json={"model": "gemini-2.5-flash", "instructions": "summarize"},
            timeout=5,
        )
        assert r.status_code == 400

    def test_compact_missing_instructions_returns_400(self, proxy_server):
        r = requests.post(
            f"{proxy_server}/v1/responses/compact",
            json={"model": "gemini-2.5-flash", "input": [{"role": "user", "content": "hi"}]},
            timeout=5,
        )
        assert r.status_code == 400

    def test_compact_input_exceeds_limit_returns_400(self, proxy_server):
        messages = [{"role": "user", "content": "x"}] * 101
        r = requests.post(
            f"{proxy_server}/v1/responses/compact",
            json={
                "model": "gemini-2.5-flash",
                "input": messages,
                "instructions": "summarize",
            },
            timeout=5,
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Normalizer (request is accepted and reaches the provider)
# ---------------------------------------------------------------------------

class TestNormalizerReachesProvider:
    """Verify that well-formed requests pass validation and reach the provider."""

    def test_gemini_model_reaches_gemini_provider(self, proxy_server):
        """Gemini model should route to Gemini provider (which will fail with auth, giving 500)."""
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=10,
        )
        # Should NOT be 400/404 — validation passed, it hit the provider
        assert r.status_code not in (400, 404)

    def test_glm_model_reaches_zai_provider(self, proxy_server):
        """GLM model should route to ZAI provider."""
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "glm-5-turbo",
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=10,
        )
        assert r.status_code not in (400, 404)

    def test_responses_api_input_normalized(self, proxy_server):
        """Responses API format (instructions + input) should be accepted."""
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "gemini-2.5-flash",
                "instructions": "You are helpful.",
                "input": [
                    {"type": "message", "role": "user", "content": "hello"},
                ],
            },
            timeout=10,
        )
        assert r.status_code not in (400, 404)

    def test_compact_request_reaches_provider(self, proxy_server):
        """Compact request should pass validation and reach the provider."""
        r = requests.post(
            f"{proxy_server}/v1/responses/compact",
            json={
                "model": "gemini-2.5-flash",
                "input": [{"role": "user", "content": "some context"}],
                "instructions": "Summarize the conversation history concisely.",
            },
            timeout=10,
        )
        assert r.status_code not in (400, 404)

    def test_context_headers_forwarded(self, proxy_server):
        """Verify custom context headers are accepted (we can't inspect internal state, but no 400)."""
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={
                "session_id": "sess-123",
                "x-openai-subagent": "true",
                "x-codex-turn-state": "active",
                "x-codex-personality": "pragmatic",
            },
            timeout=10,
        )
        assert r.status_code not in (400, 404)

    def test_tool_calls_accepted(self, proxy_server):
        """Request with tools should pass validation."""
        r = requests.post(
            f"{proxy_server}/v1/responses",
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "description": "Run a shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"},
                            },
                        },
                    },
                }],
            },
            timeout=10,
        )
        assert r.status_code not in (400, 404)


# ---------------------------------------------------------------------------
# CORS preflight
# ---------------------------------------------------------------------------

class TestCORS:
    def test_options_returns_204(self, proxy_server):
        r = requests.options(f"{proxy_server}/v1/responses", timeout=5)
        assert r.status_code == 204
        assert "Access-Control-Allow-Origin" in r.headers
