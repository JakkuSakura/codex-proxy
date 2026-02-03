import json
from src.codex_proxy.providers.gemini_utils import map_messages, get_gpt52_instructions

def test_gpt52_instruction_assembly():
    # Test base assembly
    instr = get_gpt52_instructions("pragmatic")
    assert "You are Codex, based on GPT-5" in instr
    assert "Personality" in instr
    assert "deeply pragmatic" in instr
    assert "High-Quality Output Rules" in instr
    assert "Title Case for section headers" in instr

def test_personality_injection():
    messages = [
        {"role": "system", "content": "personality:friendly"},
        {"role": "user", "content": "Hello"}
    ]
    contents, system_instruction = map_messages(messages, "gemini-3")
    
    instr_text = system_instruction['parts'][0]['text']
    assert "Personality" in instr_text
    assert "supportive teammate" in instr_text
    if "pragmatic" in instr_text:
        # Check if it's because of the high-quality rules
        assert "Adopt 'Senior Engineer Energy': be pragmatic" in instr_text
        # But the personality section should NOT be pragmatic
        assert "deeply pragmatic" not in instr_text

def test_message_mapping_parity():
    messages = [
        {"role": "user", "content": "Calculate 2+2"},
        {"role": "assistant", "content": "It's 4", "reasoning_content": "Simple math."}
    ]
    contents, _ = map_messages(messages, "gemini-3")
    
    assert len(contents) == 2
    assert contents[0]['role'] == 'user'
    assert contents[1]['role'] == 'model'
    assert contents[1]['parts'][0]['thought'] is True
    assert contents[1]['parts'][0]['text'] == "Simple math."

if __name__ == "__main__":
    test_gpt52_instruction_assembly()
    test_personality_injection()
    test_message_mapping_parity()
    print("All parity tests passed!")
