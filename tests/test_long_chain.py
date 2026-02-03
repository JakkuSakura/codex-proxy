
import requests
import json
import sys
import time

PROXY_URL = "http://127.0.0.1:8765/v1/responses"
MODEL = 'gemini-3-pro-preview'

def run_turn(input_history, expected_tool=None):
    payload = {
        "model": MODEL,
        "input": input_history,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "stream": False
    }
    
    resp = requests.post(PROXY_URL, json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"[FAILED] HTTP {resp.status_code}: {resp.text}")
        return None
    
    return resp.json()

def test_long_chain():
    print(f"Testing 3-Turn Tool Chaining on {MODEL}...")
    
    # Turn 1: Initial Prompt
    print("\n--- Turn 1 ---")
    input_history = [{"type": "message", "role": "user", "content": "Tell me what OS this is, then list files, then tell me what is inside README.md. Use tools for each step separately."}]
    data1 = run_turn(input_history)
    if not data1: return False
    
    output1 = data1.get('output', [])
    tool_call1 = next((item for item in output1 if item.get('type') == 'function_call'), None)
    if not tool_call1:
        print("[FAILED] Turn 1: No tool call")
        return False
    print(f"[OK] Turn 1 call: {tool_call1['name']} args={tool_call1['arguments']}")
    
    # Turn 2: Provide OS info, Expect File List
    print("\n--- Turn 2 ---")
    input_history.extend(output1)
    input_history.append({
        "type": "function_call_output", 
        "call_id": tool_call1['id'],
        "output": "Linux host 6.18.3-arch1-1 #1 SMP PREEMPT_DYNAMIC x86_64 GNU/Linux"
    })
    
    data2 = run_turn(input_history)
    if not data2: return False
    
    output2 = data2.get('output', [])
    tool_call2 = next((item for item in output2 if item.get('type') == 'function_call'), None)
    if not tool_call2:
        print("[FAILED] Turn 2: No tool call")
        print(json.dumps(data2, indent=2))
        return False
    print(f"[OK] Turn 2 call: {tool_call2['name']} args={tool_call2['arguments']}")
    
    # Turn 3: Provide File List, Expect README content
    print("\n--- Turn 3 ---")
    input_history.extend(output2)
    input_history.append({
        "type": "function_call_output", 
        "call_id": tool_call2['id'],
        "output": "README.md\nsrc/\ntests/"
    })
    
    data3 = run_turn(input_history)
    if not data3: return False
    
    output3 = data3.get('output', [])
    tool_call3 = next((item for item in output3 if item.get('type') == 'function_call'), None)
    if not tool_call3:
        print("[FAILED] Turn 3: No tool call")
        print(json.dumps(data3, indent=2))
        return False
    print(f"[OK] Turn 3 call: {tool_call3['name']} args={tool_call3['arguments']}")

    # Turn 4: Final response
    print("\n--- Turn 4 ---")
    input_history.extend(output3)
    input_history.append({
        "type": "function_call_output", 
        "call_id": tool_call3['id'],
        "output": "# Codex Proxy\nA high-performance proxy for Gemini and other models."
    })
    
    data3 = resp3.json()
    final_text = ""
    for item in data3.get('output', []):
        if item.get('type') == 'message':
            for content in item.get('content', []):
                if content.get('type') == 'output_text':
                    final_text += content.get('text', '')
    
    print(f"Final model response: {final_text}")
    if "Codex Proxy" in final_text:
        print("\n[SUCCESS] 4-Turn Chain Completed!")
        return True
    else:
        print("\n[FAILED] Final response didn't contain expected text.")
        return False

if __name__ == "__main__":
    if test_long_chain():
        sys.exit(0)
    else:
        sys.exit(1)
