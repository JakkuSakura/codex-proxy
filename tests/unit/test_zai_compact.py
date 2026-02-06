import json
import pytest
from unittest.mock import MagicMock, patch
from codex_proxy.providers.zai import ZAIProvider


@pytest.fixture
def zai_provider():
    return ZAIProvider()


def test_zai_compact_success(zai_provider, mock_handler):
    """Test successful compaction request to Z.AI."""
    data = {
        "model": "glm-4.6",
        "input": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
        ],
        "instructions": "Summarize this conversation",
    }
    handler, wfile = mock_handler(data, path="/v1/responses/compact")

    zai_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Summary of conversation: User said hello, assistant responded.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    with patch.object(zai_provider.session, "post") as mock_post:
        mock_post.return_value.__enter__.return_value.status_code = 200
        mock_post.return_value.__enter__.return_value.json.return_value = zai_response

        zai_provider.handle_compact(data, handler)

        # Verify the request was sent
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]

        assert payload["model"] == "glm-4.6"
        assert payload["stream"] == False
        assert payload["temperature"] == 0.1
        assert payload["max_tokens"] == 4096

        # Verify messages include compaction prompt
        messages = payload["messages"]
        assert len(messages) == 3  # 2 original + 1 compaction prompt
        assert (
            messages[2]["content"]
            == "Perform context compaction. instructions: Summarize this conversation"
        )

        # Verify response format
        handler.send_response.assert_called_with(200)
        handler.send_header.assert_called()
        handler.end_headers.assert_called()

        wfile_write_calls = wfile.write.call_args_list
        assert len(wfile_write_calls) > 0

        # Parse the response
        response_data = json.loads(wfile_write_calls[-1][0][0])
        assert "output" in response_data
        assert len(response_data["output"]) == 1
        assert response_data["output"][0]["type"] == "compaction"
        assert "encrypted_content" in response_data["output"][0]


def test_zai_compact_no_instructions(zai_provider, mock_handler):
    """Test compaction with default instructions."""
    data = {
        "model": "glm-4.6",
        "input": [{"role": "user", "content": "Test"}],
    }
    handler, wfile = mock_handler(data, path="/v1/responses/compact")

    zai_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test summary"},
                "finish_reason": "stop",
            }
        ],
    }

    with patch.object(zai_provider.session, "post") as mock_post:
        mock_post.return_value.__enter__.return_value.status_code = 200
        mock_post.return_value.__enter__.return_value.json.return_value = zai_response

        zai_provider.handle_compact(data, handler)

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]

        # Should use default instructions
        assert "Perform context compaction." in payload["messages"][-1]["content"]


def test_zai_compact_empty_history(zai_provider, mock_handler):
    """Test compaction with empty conversation history."""
    data = {
        "model": "glm-4.6",
        "input": [],
        "instructions": "Create summary",
    }
    handler, wfile = mock_handler(data, path="/v1/responses/compact")

    zai_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Empty summary"},
                "finish_reason": "stop",
            }
        ],
    }

    with patch.object(zai_provider.session, "post") as mock_post:
        mock_post.return_value.__enter__.return_value.status_code = 200
        mock_post.return_value.__enter__.return_value.json.return_value = zai_response

        zai_provider.handle_compact(data, handler)

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]

        # Should only have compaction prompt
        assert len(payload["messages"]) == 1
        assert (
            payload["messages"][0]["content"]
            == "Perform context compaction. instructions: Create summary"
        )


def test_zai_compact_no_model(zai_provider, mock_handler):
    """Test compaction when model is not specified (uses first from config)."""
    data = {
        "input": [{"role": "user", "content": "Test"}],
        "instructions": "Summary",
    }
    handler, wfile = mock_handler(data, path="/v1/responses/compact")

    zai_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Summary"},
                "finish_reason": "stop",
            }
        ],
    }

    with patch.object(zai_provider.session, "post") as mock_post:
        mock_post.return_value.__enter__.return_value.status_code = 200
        mock_post.return_value.__enter__.return_value.json.return_value = zai_response

        zai_provider.handle_compact(data, handler)

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]

        # Should use the model from request or config
        assert "model" in payload


def test_zai_compact_api_error(zai_provider, mock_handler):
    """Test compaction when Z.AI API returns an error."""
    data = {
        "model": "glm-4.6",
        "input": [{"role": "user", "content": "Test"}],
    }
    handler, wfile = mock_handler(data, path="/v1/responses/compact")

    with patch.object(zai_provider.session, "post") as mock_post:
        mock_post.return_value.__enter__.return_value.status_code = 500
        mock_post.return_value.__enter__.return_value.text = "Internal Server Error"

        zai_provider.handle_compact(data, handler)

        # Should send error response
        handler.send_error.assert_called_with(500, "Internal Server Error")


def test_zai_compact_exception(zai_provider, mock_handler):
    """Test compaction when an exception occurs."""
    data = {
        "model": "glm-4.6",
        "input": [{"role": "user", "content": "Test"}],
    }
    handler, wfile = mock_handler(data, path="/v1/responses/compact")

    with patch.object(zai_provider.session, "post") as mock_post:
        mock_post.side_effect = Exception("Connection error")

        zai_provider.handle_compact(data, handler)

        # Should send error response
        handler.send_error.assert_called_with(500, "Connection error")


def test_zai_compact_auth_header(zai_provider, mock_handler):
    """Test that auth header is used from handler if available."""
    data = {
        "model": "glm-4.6",
        "input": [{"role": "user", "content": "Test"}],
    }
    handler, wfile = mock_handler(data, path="/v1/responses/compact")
    handler.headers = {"Authorization": "Bearer test-token-123"}

    zai_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Summary"},
                "finish_reason": "stop",
            }
        ],
    }

    with patch.object(zai_provider.session, "post") as mock_post:
        mock_post.return_value.__enter__.return_value.status_code = 200
        mock_post.return_value.__enter__.return_value.json.return_value = zai_response

        zai_provider.handle_compact(data, handler)

        call_args = mock_post.call_args
        headers = call_args.kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token-123"
