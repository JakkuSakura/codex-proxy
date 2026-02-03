from typing import Dict, Any
from .base import BaseProvider
from ..utils import create_session
from ..config import config

class ZAIProvider(BaseProvider):
    def __init__(self):
        self.session = create_session()

    def handle_request(self, data: Dict[str, Any], handler: Any) -> None:
        # Fix Role
        if 'messages' in data:
            for m in data['messages']:
                if m.get('role') == 'developer':
                    m['role'] = 'system'
        
        stream = data.get('stream', False)
        auth_header = handler.headers.get('Authorization', '')
        
        try:
            with self.session.post(
                config.z_ai_url, 
                json=data, 
                headers={'Authorization': auth_header}, 
                stream=stream, 
                timeout=(10, 600)
            ) as resp:
                
                # Forward Status
                handler.send_response(resp.status_code)
                handler.send_header('Content-Type', 'text/event-stream' if stream else 'application/json')
                handler.send_header('Connection', 'keep-alive')
                handler.end_headers()
                
                if stream:
                    for line in resp.iter_lines():
                        if line:
                            # If Responses API requested, we should ideally map ZAI stream too
                            # For now, let's keep it simple or implement mapping
                            handler.wfile.write(line + b'\n')
                            handler.wfile.flush()
                else:
                    if data.get('_is_responses_api'):
                        try:
                            z_data = resp.json()
                            content = z_data['choices'][0]['message']['content']
                            usage = z_data.get('usage', {})
                            resp_obj = {
                                "id": f"zai_{z_data.get('id')}",
                                "object": "response",
                                "created": z_data.get('created'),
                                "model": z_data.get('model'),
                                "status": "completed",
                                "usage": {
                                    "prompt_tokens": usage.get('prompt_tokens', 0),
                                    "completion_tokens": usage.get('completion_tokens', 0),
                                    "total_tokens": usage.get('total_tokens', 0)
                                },
                                "output": [{
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": content}]
                                }]
                            }
                            import orjson
                            handler.wfile.write(orjson.dumps(resp_obj))
                        except Exception:
                            handler.wfile.write(resp.content)
                    else:
                        handler.wfile.write(resp.content)
        except Exception as e:
            # Let the server handler deal with connection errors if not already started
            raise e
