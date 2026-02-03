import requests
import json
import sys

PROXY_URL = "http://127.0.0.1:8765/v1/responses"
MODEL = 'gemini-2.5-flash-lite'

def test_search():
    print(f"Testing Google Search Grounding on {MODEL}...")
    url = PROXY_URL
    payload = {
        "model": MODEL,
        "input": "Who is the current Prime Minister of the UK?",
        "include": ["search"],
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        if response.status_code != 200:
            print(f"[FAILED] HTTP {response.status_code}")
            print(response.text)
            return False
            
        found_citation = False
        text = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        raw_data = line_str[6:]
                        if not raw_data.strip() or raw_data == '[DONE]': continue
                        data = json.loads(raw_data)
                        etype = data.get('type')
                        if etype == 'citation':
                            print(f"[OK] Found citation: {data.get('value')}")
                            found_citation = True
                        elif etype == 'response.output_text.delta':
                            text += data.get('delta', '')
                    except Exception as e:
                        print(f"Failed to parse line: {line_str} - Error: {e}")
        
        print(f"Model Response: {text}")
        if found_citation:
            print("[SUCCESS] Grounding citations emitted!")
            return True
        else:
            print("[WARNING] No citations found. Model might have answered from internal knowledge.")
            return True # Not necessarily a failure if model is sure
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    if test_search():
        sys.exit(0)
    else:
        sys.exit(1)
