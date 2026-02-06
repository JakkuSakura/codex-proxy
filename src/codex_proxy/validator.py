"""Input validation for codex-proxy requests."""

import logging
from typing import Dict, Any, List

from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class RequestValidator:
    """Validates incoming proxy requests."""

    @staticmethod
    def validate_request(data: Dict[str, Any], path: str) -> None:
        """
        Validate incoming request data.

        Raises ValidationError if validation fails.
        """
        # Validate model field
        model = data.get("model")
        if model:
            RequestValidator._validate_model(model)

        # Validate messages
        if "messages" in data:
            messages = data["messages"]
            if not isinstance(messages, list):
                raise ValidationError("messages must be a list")
            RequestValidator._validate_messages(messages)

        # Validate tools if present
        if "tools" in data:
            RequestValidator._validate_tools(data["tools"])

        # Validate temperature
        if "temperature" in data:
            temp = data["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise ValidationError(
                    f"temperature must be between 0 and 2, got: {temp}"
                )

        # Validate max_tokens
        if "max_tokens" in data:
            max_tokens = data["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 128000:
                raise ValidationError(
                    f"max_tokens must be between 1 and 128000, got: {max_tokens}"
                )

        # Validate stream flag
        if "stream" in data:
            stream = data["stream"]
            if not isinstance(stream, bool):
                raise ValidationError(f"stream must be a boolean, got: {type(stream)}")

        # Validate compaction requests
        if "/compact" in path:
            RequestValidator._validate_compact_request(data)

        logger.debug("Request validation passed")

    @staticmethod
    def _validate_model(model: str) -> None:
        """Validate model name format."""
        if not isinstance(model, str) or len(model) > 100:
            raise ValidationError(f"Invalid model name: {model}")

    @staticmethod
    def _validate_messages(messages: List[Dict[str, Any]]) -> None:
        """Validate message structure."""
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValidationError(f"Message {i} must be an object")

            if "role" not in msg:
                raise ValidationError(f"Message {i} missing required field 'role'")

            role = msg["role"]
            valid_roles = ("system", "user", "assistant", "developer")
            if role not in valid_roles:
                raise ValidationError(f"Message {i} has invalid role: {role}")

            if role == "user" and "content" not in msg:
                if "content" not in msg and "text" not in msg:
                    raise ValidationError(
                        f"User message {i} must have 'content' or 'text'"
                    )

    @staticmethod
    def _validate_tools(tools: List[Dict[str, Any]]) -> None:
        """Validate tools structure."""
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValidationError(f"Tool {i} must be an object")

            if "type" not in tool:
                raise ValidationError(f"Tool {i} missing required field 'type'")

            tool_type = tool["type"]
            valid_types = ("function", "web_search", "retrieval")
            if tool_type not in valid_types:
                raise ValidationError(f"Tool {i} has invalid type: {tool_type}")

    @staticmethod
    def _validate_compact_request(data: Dict[str, Any]) -> None:
        """Validate compaction-specific requests."""
        if "input" not in data:
            raise ValidationError("Compaction requests must have 'input' field")

        if "instructions" not in data:
            raise ValidationError("Compaction requests must have 'instructions' field")

        input_data = data["input"]
        if not isinstance(input_data, (str, list)):
            raise ValidationError("Compaction input must be string or list")

        if isinstance(input_data, list) and len(input_data) > 100:
            raise ValidationError(
                "Compaction input exceeds maximum length of 100 messages"
            )
