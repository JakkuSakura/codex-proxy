import json
import pytest
from unittest.mock import MagicMock, patch
from codex_proxy.providers.gemini import GeminiProvider
from codex_proxy.providers.zai import ZAIProvider
from codex_proxy.providers.gemini_utils import map_messages

@pytest.fixture
def gemini_provider():
    return GeminiProvider()

@pytest.fixture
def zai_provider():
    return ZAIProvider()

def test_gemini_map_messages_reasoning(gemini_provider):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there.", "reasoning_content": "I should say hi."}
    ]
    # Use map_messages from gemini_utils
    contents, sys_instruction = map_messages(messages, "gemini-3-pro-preview")
    
    assert len(contents) == 2
    
    # 2. Assistant message parsing
    assert contents[1]["role"] == "model"
    # Parts: reasoning part + text part
    assert len(contents[1]["parts"]) == 2
    assert contents[1]["parts"][0]["text"] == "I should say hi."
    assert contents[1]["parts"][0]["thought"] is True
    assert contents[1]["parts"][1]["text"] == "Hi there."

def test_gemini_map_system_instruction(gemini_provider):
    messages = [
        {"role": "system", "content": "You are a coding bot."},
        {"role": "user", "content": "Hello"}
    ]
    contents, sys_instruction = map_messages(messages, "gemini-3-pro-preview")
    
    # System should be extracted, not in contents
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    
    # System Instruction should be populated
    assert sys_instruction is not None
    assert sys_instruction["parts"][0]["text"] == "You are a coding bot."

def test_zai_role_fix(zai_provider):
    data = {
        "messages": [{"role": "developer", "content": "system prompt"}]
    }
    handler = MagicMock()
    
    with patch.object(zai_provider.session, 'post') as mock_post:
        # Mocking the context manager
        mock_post.return_value.__enter__.return_value.status_code = 200
        mock_post.return_value.__enter__.return_value.content = b'{"id": "1"}'
        
        zai_provider.handle_request(data, handler)
        
        # Verify role changed
        assert data["messages"][0]["role"] == "system"