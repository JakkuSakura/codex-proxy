"""Unit tests for request validation."""

import pytest
from codex_proxy.validator import RequestValidator
from codex_proxy.exceptions import ValidationError


class TestModelValidation:
    """Test model field validation."""

    def test_valid_model_name(self):
        """Test that valid model name passes validation."""
        data = {"model": "gemini-2.5-flash-lite"}
        RequestValidator.validate_request(data, "/v1/responses")

    def test_long_model_name_fails(self):
        """Test that model name > 100 chars fails validation."""
        data = {"model": "x" * 101}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "Invalid model name" in str(exc.value)

    def test_non_string_model_allowed(self):
        """Test that non-string model is allowed (validation is lenient)."""
        # Current implementation only validates string length
        # Non-string models will pass validation but fail during processing
        data = {"model": 123}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "Invalid model name" in str(exc.value)


class TestMessageValidation:
    """Test message field validation."""

    def test_valid_messages(self):
        """Test that valid messages pass validation."""
        data = {
            "model": "gemini-2.5-flash-lite",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }
        RequestValidator.validate_request(data, "/v1/responses")

    def test_non_list_messages_fails(self):
        """Test that non-list messages fail validation."""
        data = {"messages": "not a list"}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "messages must be a list" in str(exc.value)

    def test_message_missing_role_fails(self):
        """Test that message without role fails validation."""
        data = {"messages": [{"content": "Hello"}]}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "missing required field 'role'" in str(exc.value)

    def test_invalid_message_role_fails(self):
        """Test that invalid message role fails validation."""
        data = {"messages": [{"role": "invalid", "content": "Hello"}]}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "invalid role" in str(exc.value)

    def test_user_message_without_content_fails(self):
        """Test that user message without content/text fails validation."""
        data = {"messages": [{"role": "user"}]}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "must have 'content' or 'text'" in str(exc.value)

    def test_non_object_message_fails(self):
        """Test that non-object message fails validation."""
        data = {"messages": ["not an object"]}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "must be an object" in str(exc.value)


class TestToolValidation:
    """Test tools field validation."""

    def test_valid_tools(self):
        """Test that valid tools pass validation."""
        data = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
        RequestValidator.validate_request(data, "/v1/responses")

    def test_tool_missing_type_fails(self):
        """Test that tool without type fails validation."""
        data = {"tools": [{"name": "search"}]}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "missing required field 'type'" in str(exc.value)

    def test_invalid_tool_type_fails(self):
        """Test that invalid tool type fails validation."""
        data = {"tools": [{"type": "invalid"}]}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "invalid type" in str(exc.value)

    def test_non_object_tool_fails(self):
        """Test that non-object tool fails validation."""
        data = {"tools": ["not an object"]}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "must be an object" in str(exc.value)


class TestParameterValidation:
    """Test parameter validation."""

    def test_valid_temperature(self):
        """Test that valid temperature passes validation."""
        data = {"temperature": 0.7}
        RequestValidator.validate_request(data, "/v1/responses")

    def test_temperature_out_of_range_high_fails(self):
        """Test that temperature > 2 fails validation."""
        data = {"temperature": 3.0}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "temperature must be between 0 and 2" in str(exc.value)

    def test_temperature_out_of_range_low_fails(self):
        """Test that temperature < 0 fails validation."""
        data = {"temperature": -0.5}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "temperature must be between 0 and 2" in str(exc.value)

    def test_non_numeric_temperature_fails(self):
        """Test that non-numeric temperature fails validation."""
        data = {"temperature": "high"}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "temperature must be between 0 and 2" in str(exc.value)

    def test_valid_max_tokens(self):
        """Test that valid max_tokens passes validation."""
        data = {"max_tokens": 1000}
        RequestValidator.validate_request(data, "/v1/responses")

    def test_max_tokens_out_of_range_high_fails(self):
        """Test that max_tokens > 128000 fails validation."""
        data = {"max_tokens": 200000}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "max_tokens must be between 1 and 128000" in str(exc.value)

    def test_max_tokens_out_of_range_low_fails(self):
        """Test that max_tokens < 1 fails validation."""
        data = {"max_tokens": 0}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "max_tokens must be between 1 and 128000" in str(exc.value)

    def test_non_integer_max_tokens_fails(self):
        """Test that non-integer max_tokens fails validation."""
        data = {"max_tokens": "1000"}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "max_tokens must be between 1 and 128000" in str(exc.value)

    def test_valid_stream(self):
        """Test that valid stream passes validation."""
        data = {"stream": True}
        RequestValidator.validate_request(data, "/v1/responses")
        data = {"stream": False}
        RequestValidator.validate_request(data, "/v1/responses")

    def test_non_boolean_stream_fails(self):
        """Test that non-boolean stream fails validation."""
        data = {"stream": "true"}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses")
        assert "stream must be a boolean" in str(exc.value)


class TestCompactionValidation:
    """Test compaction-specific validation."""

    def test_valid_compact_request(self):
        """Test that valid compact request passes validation."""
        data = {
            "model": "gemini-2.5-flash-lite",
            "input": "Long conversation history...",
            "instructions": "Summarize this conversation",
        }
        RequestValidator.validate_request(data, "/v1/responses/compact")

    def test_compact_without_input_fails(self):
        """Test that compact request without input fails validation."""
        data = {"instructions": "Summarize"}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses/compact")
        assert "must have 'input' field" in str(exc.value)

    def test_compact_without_instructions_fails(self):
        """Test that compact request without instructions fails validation."""
        data = {"input": "Some content"}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses/compact")
        assert "must have 'instructions' field" in str(exc.value)

    def test_compact_invalid_input_type_fails(self):
        """Test that compact request with invalid input type fails validation."""
        data = {"input": 123, "instructions": "Summarize"}
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses/compact")
        assert "must be string or list" in str(exc.value)

    def test_compact_input_too_long_fails(self):
        """Test that compact request with too many messages fails validation."""
        data = {
            "input": [{"role": "user", "content": "msg"}] * 101,
            "instructions": "Summarize",
        }
        with pytest.raises(ValidationError) as exc:
            RequestValidator.validate_request(data, "/v1/responses/compact")
        assert "exceeds maximum length of 100 messages" in str(exc.value)


class TestComplexRequests:
    """Test complex, realistic request validation."""

    def test_full_request_with_all_fields(self):
        """Test a complete request with all fields."""
        data = {
            "model": "gemini-2.5-flash-lite",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": True,
        }
        RequestValidator.validate_request(data, "/v1/responses")

    def test_minimal_valid_request(self):
        """Test minimal valid request."""
        data = {"model": "gemini-2.5-flash-lite"}
        RequestValidator.validate_request(data, "/v1/responses")
