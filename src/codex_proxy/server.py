import json
import logging
import socket
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

try:
    import orjson
except ImportError:
    import json as orjson

from .config import config
from .providers.gemini import GeminiProvider
from .providers.zai import ZAIProvider

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    
    def server_bind(self):
        """Override to enable TCP_NODELAY for lower latency."""
        HTTPServer.server_bind(self)
        try:
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            logger.warning(f"Failed to set TCP_NODELAY: {e}")

class RequestHandler(BaseHTTPRequestHandler):
    gemini_provider = GeminiProvider()
    zai_provider = ZAIProvider()

    def _normalize_request(self, data):
        """Normalize 'responses' wire API to OpenAI Chat API structure for internal processing."""
        messages = []
        
        # 1. Handle Instructions (System Prompt)
        if 'instructions' in data:
            inst = data['instructions']
            content = ""
            if isinstance(inst, str):
                content = inst
            elif isinstance(inst, list):
                for block in inst:
                    if isinstance(block, str):
                        content += block
                    elif isinstance(block, dict):
                        content += block.get('text', '')
            
            if content:
                messages.append({"role": "system", "content": content})
        
        # 2. Handle Input (User Prompts & History)
        if 'input' in data:
            inp = data['input']
            if isinstance(inp, str):
                inp = [inp]
                
            if isinstance(inp, list):
                for item in inp:
                    if isinstance(item, str):
                         messages.append({"role": "user", "content": item})
                    elif isinstance(item, dict):
                        item_type = item.get('type', 'message')
                        
                        # Helper to get/create last assistant message for merging
                        def get_last_assistant():
                            if messages and messages[-1]['role'] == 'assistant':
                                return messages[-1]
                            msg = {"role": "assistant", "content": None}
                            messages.append(msg)
                            return msg

                        if item_type in ('message', 'agentMessage'):
                            role = item.get('role', 'user')
                            if role == 'developer': role = 'system'
                            
                            content_raw = item.get('content')
                            content = ""
                            reasoning_content = item.get('reasoning_content', "")
                            if isinstance(content_raw, str):
                                content = content_raw
                            elif isinstance(content_raw, list):
                                for part in content_raw:
                                    if isinstance(part, str): content += part
                                    elif isinstance(part, dict):
                                        ptype = part.get('type')
                                        if ptype in ('input_text', 'text', 'output_text'):
                                            content += part.get('text', '')
                                        elif ptype == 'reasoning_text':
                                            reasoning_content += part.get('text', '')
                            
                            if role == 'assistant' or role == 'model':
                                amsg = get_last_assistant()
                                if content:
                                    amsg['content'] = (amsg['content'] or "") + content
                                if reasoning_content:
                                    amsg['reasoning_content'] = (amsg.get('reasoning_content') or "") + reasoning_content
                                if item.get('thought_signature'):
                                    amsg['thought_signature'] = item.get('thought_signature')
                            else:
                                messages.append({"role": role, "content": content or ""})

                        elif item_type == "reasoning":
                            content_list = item.get('content', [])
                            content = ""
                            if isinstance(content_list, list):
                                for cp in content_list:
                                    if isinstance(cp, str): content += cp
                                    elif isinstance(cp, dict): content += cp.get('text', '')
                            
                            amsg = get_last_assistant()
                            amsg['reasoning_content'] = (amsg.get('reasoning_content') or "") + content
                            if item.get('thought_signature'):
                                amsg['thought_signature'] = item.get('thought_signature')

                        elif item_type in ('function_call', 'commandExecution', 'local_shell_call', 'fileChange', 'custom_tool_call', 'web_search_call'):
                            call_id = item.get('call_id') or item.get('id') or f"call_{len(messages)}"
                            name = item.get('name')
                            if not name:
                                if item_type == 'commandExecution': name = 'run_shell_command'
                                elif item_type == 'local_shell_call': name = 'local_shell_command'
                                elif item_type == 'fileChange': name = 'write_file'
                                elif item_type == 'web_search_call': name = 'web_search'
                            
                            args = item.get('arguments') or item.get('input') or {}
                            if not args and item_type == 'web_search_call':
                                args = item.get('action') or {}

                            if not args:
                                if item_type == 'commandExecution': args = {'command': item.get('command', ''), 'dir_path': item.get('cwd', '.')}
                                elif item_type == 'local_shell_call': 
                                    action = item.get('action', {})
                                    exec_data = action.get('exec', {})
                                    args = {'command': exec_data.get('command', []), 'working_directory': exec_data.get('working_directory')}
                                elif item_type == 'fileChange':
                                    changes = item.get('changes', [])
                                    path = changes[0].get('path') if changes else 'unknown'
                                    args = {'file_path': path}
                            
                            if isinstance(args, dict): args = json.dumps(args)
                            
                            if name:
                                amsg = get_last_assistant()
                                if 'tool_calls' not in amsg: amsg['tool_calls'] = []
                                amsg['tool_calls'].append({"id": call_id, "type": "function", "function": {"name": name, "arguments": args}})
                                
                                # Signature/Thought recovery
                                it_sig = item.get('thought_signature')
                                it_th = item.get('thought')
                                if it_sig: amsg['thought_signature'] = it_sig
                                if it_th: amsg['reasoning_content'] = (amsg.get('reasoning_content') or "") + it_th

                        elif item_type in ('function_call_output', 'commandExecutionOutput', 'fileChangeOutput', 'custom_tool_call_output'):
                            call_id = item.get('call_id') or item.get('id')
                            output_raw = item.get('output') or item.get('content') or item.get('stdout', '')
                            
                            content = ""
                            if isinstance(output_raw, str):
                                content = output_raw
                            elif isinstance(output_raw, dict):
                                content = output_raw.get('content', '')
                                if not content and output_raw.get('success') is False:
                                    content = "Error: Tool execution failed"
                            elif isinstance(output_raw, list):
                                # Handle structured tool output items
                                for part in output_raw:
                                    if isinstance(part, str): content += part
                                    elif isinstance(part, dict):
                                        if part.get('type') in ('input_text', 'text'):
                                            content += part.get('text', '')
                            
                            if not content and item.get('stderr'): content = f"Error: {item['stderr']}"
                            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

        # Use the normalized messages
        data['messages'] = messages
        
        # 3. Handle Responses API specific fields
        data['previous_response_id'] = data.get('previous_response_id')
        data['store'] = data.get('store', False)
        data['metadata'] = data.get('metadata', {})
        
        # Map tools (flat -> wrapped)
        if 'tools' in data:
            normalized_tools = []
            for t in data['tools']:
                if t.get('type') == 'function' and 'function' not in t:
                    normalized_tools.append({
                        'type': 'function',
                        'function': {
                            'name': t.get('name'),
                            'description': t.get('description'),
                            'parameters': t.get('parameters'),
                            'strict': t.get('strict', False)
                        }
                    })
                else:
                    normalized_tools.append(t)
            data['tools'] = normalized_tools
            
        return data

    def do_POST(self):
        try:
            logger.info(f"POST {self.path}")
            
            # 1. Enforce supported endpoints
            if self.path not in ('/v1/responses', '/responses', '/v1/responses/compact', '/responses/compact'):
                self.send_error(404, f"Endpoint {self.path} not supported.")
                return

            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self.send_error(400, "Empty body")
                return

            body = self.rfile.read(length)
            body_str = body.decode('utf-8', errors='replace')
            if config.debug_mode:
                logger.debug(f"RAW CODEX REQUEST: {body_str}")
            
            try:
                data = orjson.loads(body)
            except Exception:
                data = json.loads(body)
            
            # Extract important headers for providers
            data['_headers'] = {
                'session_id': self.headers.get('session_id'),
                'x-openai-subagent': self.headers.get('x-openai-subagent'),
                'x-codex-turn-state': self.headers.get('x-codex-turn-state')
            }

            # 2. Normalize Schema (only for standard responses, compact has its own schema)
            if '/compact' not in self.path:
                data = self._normalize_request(data)
                data['_is_responses_api'] = True
            
            model = data.get('model', '')
            
            if model.startswith('gemini'):
                if '/compact' in self.path:
                    self.gemini_provider.handle_compact(data, self)
                else:
                    self.gemini_provider.handle_request(data, self)
            else:
                self.zai_provider.handle_request(data, self)
            
            self.close_connection = True
                
        except Exception as e:
            logger.error(f"Handler Error: {e}", exc_info=True)
            try:
                self.send_error(500, str(e))
            except Exception:
                pass  # Connection might be closed

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def log_message(self, format, *args):
        # Disable default HTTP logging to stdout/stderr for speed
        pass

if __name__ == "__main__":
    port = config.port
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, RequestHandler)
    logger.info(f"Starting Codex Proxy on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        httpd.server_close()
