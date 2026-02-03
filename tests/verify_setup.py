import requests
import json
import sys

PROXY_URL = "http://127.0.0.1:8765/v1/responses"

MODELS = ['gemini-2.5-flash-lite']

def test_model(model_name):
    print(f"Testing {model_name}...", end=" ")
    try:
        resp = requests.post(
            PROXY_URL,
            json={
                "model": model_name,
                "input": "Hi",
                "stream": False
            },
            timeout=60
        )
        if resp.status_code == 200:
            try:
                data = resp.json()
                if "output" in data and len(data["output"]) > 0:
                    print("[OK]")
                    return True
                else:
                    print("[FAILED] Invalid Responses API Structure")
                    print(json.dumps(data, indent=2)[:500])
                    return False
            except Exception as e:
                print(f"[FAILED] Not JSON: {e}")
                print(resp.text[:200])
                return False
        else:
            print(f"[FAILED] ({resp.status_code})")
            print(resp.text[:200])
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_tool_use(model_name):
    print(f"Testing Tool Use on {model_name}...", end=" ")
    try:
        resp = requests.post(
            PROXY_URL,
            json={
                "model": model_name,
                "input": "What is the weather in Paris?",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"]
                            }
                        }
                    }
                ],
                "tool_choice": "auto"
            },
            timeout=60
        )
        if resp.status_code == 200:
            # Responses API might return tool_calls as an item in output
            # For Gemini, we might just check if any item has type 'function_call' (if implemented)
            # or if it's in the text.
            if "function_call" in resp.text or "tool_call" in resp.text:
                print("[OK] (Tool Triggered)")
            else:
                print("[OK] (No Tool Triggered)")
            return True
        else:
            print(f"[FAILED] ({resp.status_code})")
            print(resp.text[:200])
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_reasoning(model_name):
    print(f"Testing Reasoning on {model_name}...", end=" ")
    try:
        resp = requests.post(
            PROXY_URL,
            json={
                "model": model_name,
                "input": "How many Rs in strawberry?",
                "stream": False
            },
            timeout=60
        )
        if resp.status_code == 200:
            data = resp.json()
            output = data.get('output', [])
            has_reasoning = any(item.get('type') == 'reasoning' for item in output)
            if has_reasoning:
                print("[OK] (Reasoning Captured)")
            else:
                print("[OK] (No Reasoning Item in Output)")
            return True
        else:
            print(f"[FAILED] ({resp.status_code})")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    print("--- Starting Proxy Responses API Verification ---")
    success = True
    # Test only first model for speed
    if not test_model(MODELS[0]):
        success = False

    if success:
        test_tool_use(MODELS[0])
        test_reasoning(MODELS[0])

    if not success:
        sys.exit(1)
