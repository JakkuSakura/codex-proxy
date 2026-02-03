
import requests
import json
import sys

PROXY_URL = "http://127.0.0.1:8765/v1/responses"
MODEL = 'gemini-2.5-flash-lite'

def test_tool_chaining():
    print(f"Testing Tool Chaining on {MODEL}...")
    
    # 1. First request to trigger tool
    print("Step 1: Requesting tool call...")
    payload1 = {
        "model": MODEL,
        "input": "What is the content of secret.txt?",
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
            }
        ],
        "tool_choice": "auto",
        "stream": False
    }
    
    resp1 = requests.post(PROXY_URL, json=payload1, timeout=60)
    if resp1.status_code != 200:
        print(f"[FAILED] Step 1: {resp1.status_code}")
        print(resp1.text)
        return False
    
    data1 = resp1.json()
    output1 = data1.get('output', [])
    tool_call = next((item for item in output1 if item.get('type') == 'function_call'), None)
    
    if not tool_call:
        print("[FAILED] No tool call triggered")
        print(json.dumps(data1, indent=2))
        return False
    
    print(f"[OK] Tool call triggered: {tool_call['name']} id={tool_call['id']}")
    
    # 2. Second request providing tool output
    print("Step 2: Providing tool output...")
    # The Responses API input can include Item Objects
    input_history = [{"type": "message", "role": "user", "content": "What is the content of secret.txt?"}]
    input_history.extend(output1) # Include reasoning, function_call, etc.
    input_history.append({
        "type": "function_call_output", 
        "call_id": tool_call['id'],
        "output": "The secret code is 12345."
    })
    
    payload2 = {
        "model": MODEL,
        "input": input_history,
        "stream": False
    }
    
    resp2 = requests.post(PROXY_URL, json=payload2, timeout=60)
    if resp2.status_code != 200:
        print(f"[FAILED] Step 2: {resp2.status_code}")
        print(resp2.text)
        return False
        
    data2 = resp2.json()
    final_text = ""
    for item in data2.get('output', []):
        if item.get('type') == 'message':
            for content in item.get('content', []):
                if content.get('type') == 'output_text':
                    final_text += content.get('text', '')
                    
    print(f"Final model response: {final_text}")
    if "12345" in final_text:
        print("[SUCCESS] Model correctly used the tool output!")
        return True
    else:
        print("[FAILED] Model did not seem to use the tool output.")
        return False

if __name__ == "__main__":
    if test_tool_chaining():
        sys.exit(0)
    else:
        sys.exit(1)
