"""Unit tests for request normalization."""

from codex_proxy.normalizer import RequestNormalizer


class TestInstructionNormalization:
    """Test instruction (system prompt) normalization."""

    def test_string_instruction(self):
        """Test that string instruction is converted to system message."""
        data = {"instructions": "You are a helpful assistant."}
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."

    def test_list_of_string_instructions(self):
        """Test that list of string instructions are concatenated."""
        data = {"instructions": ["Part 1", "Part 2", "Part 3"]}
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Part 1Part 2Part 3"

    def test_list_of_dict_instructions(self):
        """Test that list of dict instructions are concatenated."""
        data = {"instructions": [{"text": "Part 1"}, {"text": "Part 2"}]}
        result = RequestNormalizer.normalize(data)
        assert result["messages"][0]["content"] == "Part 1Part 2"

    def test_mixed_instructions(self):
        """Test that mixed instruction formats work."""
        data = {"instructions": ["String part", {"text": "Dict part"}]}
        result = RequestNormalizer.normalize(data)
        assert "String part" in result["messages"][0]["content"]
        assert "Dict part" in result["messages"][0]["content"]


class TestInputNormalization:
    """Test input (user prompts) normalization."""

    def test_string_input(self):
        """Test that string input is converted to user message."""
        data = {"input": "Hello, world!"}
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, world!"

    def test_list_of_string_inputs(self):
        """Test that list of string inputs creates multiple user messages."""
        data = {"input": ["Message 1", "Message 2", "Message 3"]}
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 3
        assert all(msg["role"] == "user" for msg in result["messages"])
        assert result["messages"][0]["content"] == "Message 1"

    def test_message_type_input(self):
        """Test that message-type input items are normalized."""
        data = {"input": [{"type": "message", "role": "user", "content": "Hello"}]}
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    def test_developer_role_mapping(self):
        """Test that developer role is mapped to system."""
        data = {
            "input": [
                {"type": "message", "role": "developer", "content": "System prompt"}
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert result["messages"][0]["role"] == "system"

    def test_model_role_handling(self):
        """Test that model role is treated as assistant."""
        data = {
            "input": [{"type": "message", "role": "model", "content": "AI response"}]
        }
        result = RequestNormalizer.normalize(data)
        assert result["messages"][0]["role"] == "assistant"


class TestContentPartsNormalization:
    """Test content part normalization within messages."""

    def test_list_of_content_parts(self):
        """Test that list of content parts are concatenated."""
        data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": ["Part 1", "Part 2", "Part 3"],
                }
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert result["messages"][0]["content"] == "Part 1Part 2Part 3"

    def test_dict_content_parts(self):
        """Test that dict content parts are processed."""
        data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Text 1"},
                        {"type": "input_text", "text": "Text 2"},
                    ],
                }
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert "Text 1" in result["messages"][0]["content"]
        assert "Text 2" in result["messages"][0]["content"]

    def test_reasoning_content_extraction(self):
        """Test that reasoning content is extracted from parts."""
        data = {
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Regular content"},
                        {"type": "reasoning_text", "text": "Thinking..."},
                    ],
                }
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][0]["content"] == "Regular content"
        assert result["messages"][0]["reasoning_content"] == "Thinking..."


class TestReasoningNormalization:
    """Test reasoning block normalization."""

    def test_reasoning_block(self):
        """Test that reasoning blocks are added to assistant messages."""
        data = {
            "input": [
                {"type": "message", "role": "assistant", "content": "Response"},
                {
                    "type": "reasoning",
                    "content": ["Thinking step 1", "Thinking step 2"],
                },
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"
        assert "Thinking step 1" in result["messages"][0]["reasoning_content"]

    def test_reasoning_with_dict_content(self):
        """Test reasoning with dict content parts."""
        data = {
            "input": [
                {"type": "message", "role": "assistant", "content": "Response"},
                {
                    "type": "reasoning",
                    "content": [{"text": "Thought 1"}, {"text": "Thought 2"}],
                },
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert "Thought 1" in result["messages"][0]["reasoning_content"]
        assert "Thought 2" in result["messages"][0]["reasoning_content"]


class TestToolCallNormalization:
    """Test tool call normalization."""

    def test_function_call(self):
        """Test that function calls are normalized."""
        data = {
            "input": [
                {
                    "type": "function_call",
                    "name": "search",
                    "arguments": {"query": "test"},
                }
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"
        assert "tool_calls" in result["messages"][0]
        assert result["messages"][0]["tool_calls"][0]["function"]["name"] == "search"
        assert (
            '"query": "test"'
            in result["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )

    def test_command_execution(self):
        """Test that command execution is normalized to function call."""
        data = {
            "input": [
                {"type": "commandExecution", "command": ["ls", "-la"], "cwd": "/home"}
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert (
            result["messages"][0]["tool_calls"][0]["function"]["name"]
            == "run_shell_command"
        )

    def test_local_shell_call(self):
        """Test that local shell call is normalized."""
        data = {
            "input": [
                {
                    "type": "local_shell_call",
                    "action": {
                        "exec": {
                            "command": ["echo", "hello"],
                            "working_directory": "/tmp",
                        }
                    },
                }
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert (
            result["messages"][0]["tool_calls"][0]["function"]["name"]
            == "local_shell_command"
        )

    def test_web_search_call(self):
        """Test that web search call is normalized."""
        data = {
            "input": [{"type": "web_search_call", "action": {"query": "test search"}}]
        }
        result = RequestNormalizer.normalize(data)
        assert (
            result["messages"][0]["tool_calls"][0]["function"]["name"] == "web_search"
        )


class TestToolOutputNormalization:
    """Test tool output normalization."""

    def test_function_call_output(self):
        """Test that function call outputs are normalized."""
        data = {
            "input": [
                {"type": "message", "role": "assistant"},
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "Function result",
                },
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 2
        assert result["messages"][1]["role"] == "tool"
        assert result["messages"][1]["tool_call_id"] == "call_123"
        assert result["messages"][1]["content"] == "Function result"

    def test_tool_output_with_stdout(self):
        """Test that tool output with stdout is normalized."""
        data = {
            "input": [
                {"type": "message", "role": "assistant"},
                {
                    "type": "commandExecutionOutput",
                    "call_id": "call_456",
                    "stdout": "Command output",
                },
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert result["messages"][1]["content"] == "Command output"

    def test_tool_output_with_content_list(self):
        """Test that tool output with content list is normalized."""
        data = {
            "input": [
                {"type": "message", "role": "assistant"},
                {
                    "type": "function_call_output",
                    "call_id": "call_789",
                    "content": [
                        {"type": "text", "text": "Part 1"},
                        {"type": "input_text", "text": "Part 2"},
                    ],
                },
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert "Part 1" in result["messages"][1]["content"]
        assert "Part 2" in result["messages"][1]["content"]


class TestToolNormalization:
    """Test tools definition normalization."""

    def test_flat_tool_normalization(self):
        """Test that flat tool definitions are wrapped."""
        data = {
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search function",
                    "parameters": {"type": "object"},
                }
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert result["tools"][0]["type"] == "function"
        assert "function" in result["tools"][0]
        assert result["tools"][0]["function"]["name"] == "search"

    def test_wrapped_tool_unchanged(self):
        """Test that already wrapped tools are unchanged."""
        data = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert result["tools"][0]["function"]["name"] == "search"


class TestResponsesAPIFields:
    """Test Responses API specific fields."""

    def test_previous_response_id_preserved(self):
        """Test that previous_response_id is preserved."""
        data = {"previous_response_id": "resp_123"}
        result = RequestNormalizer.normalize(data)
        assert result["previous_response_id"] == "resp_123"

    def test_store_field_preserved(self):
        """Test that store field is preserved and defaults to False."""
        data = {}
        result = RequestNormalizer.normalize(data)
        assert result["store"] is False

        data = {"store": True}
        result = RequestNormalizer.normalize(data)
        assert result["store"] is True

    def test_metadata_preserved(self):
        """Test that metadata is preserved."""
        data = {"metadata": {"key": "value"}}
        result = RequestNormalizer.normalize(data)
        assert result["metadata"] == {"key": "value"}


class TestComplexScenarios:
    """Test complex normalization scenarios."""

    def test_multi_turn_conversation(self):
        """Test normalization of multi-turn conversation."""
        data = {
            "instructions": "You are helpful",
            "input": [
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi"},
                {
                    "type": "function_call",
                    "name": "search",
                    "arguments": {"query": "test"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Results",
                },
                {"type": "message", "role": "user", "content": "Thanks!"},
            ],
        }
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 5
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"
        assert result["messages"][2]["tool_calls"]
        assert result["messages"][3]["role"] == "tool"
        assert result["messages"][4]["role"] == "user"

    def test_assistant_content_accumulation(self):
        """Test that assistant content is accumulated correctly."""
        data = {
            "input": [
                {"type": "message", "role": "assistant", "content": "First part"},
                {"type": "reasoning", "content": ["Thinking"]},
                {"type": "message", "role": "assistant", "content": "Second part"},
            ]
        }
        result = RequestNormalizer.normalize(data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"
        assert "First part" in result["messages"][0]["content"]
        assert "Second part" in result["messages"][0]["content"]
        assert "Thinking" in result["messages"][0]["reasoning_content"]
