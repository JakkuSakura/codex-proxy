
import requests
import json
import sys

PROXY_URL = "http://127.0.0.1:8765/v1/responses"
MODEL = 'gemini-3-pro-preview'

def test_multi_tool():
    print(f"Testing Multiple Tool Calls in One Turn on {MODEL}...")
    
    payload = {
        "model": MODEL,
        "input": "Read README.md AND ALSO list the files in src/ using TWO separate tool calls.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"dir": {"type": "string"}},
                        "required": ["dir"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": False
    }
    
    resp = requests.post(PROXY_URL, json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"[FAILED]: {resp.status_code}")
        print(resp.text)
        return False
    
    data = resp.json()
    output = data.get('output', [])
    tool_calls = [item for item in output if item.get('type') == 'function_call']
    
    print(f"Triggered {len(tool_calls)} tool calls.")
    for tc in tool_calls:
        print(f" - {tc['name']}: {tc['arguments']}")
        
    if len(tool_calls) < 2:
        print("[FAILED] Expected at least 2 tool calls for this prompt.")
        # Sometimes models don't parallelize even if asked, but Gemini 3 usually does.
        return False

    # Verify signatures are present in ALL calls if thoughts were returned
    has_reasoning = any(item.get('type') == 'reasoning' for item in output)
    if has_reasoning:
        for tc in tool_calls:
            if not tc.get('thought_signature'):
                print(f"[FAILED] Tool call {tc['id']} missing thought_signature")
                return False
        print("[OK] All tool calls have signatures.")

    return True

if __name__ == "__main__":
    if test_multi_tool():
        sys.exit(0)
    else:
        sys.exit(1)
