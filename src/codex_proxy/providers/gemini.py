import json
import time
import logging
import requests
from typing import Dict, Any, List, Optional

try:
    import orjson
except ImportError:
    import json as orjson

from ..config import config
from ..auth import GeminiAuth, AuthError
from ..utils import create_session
from .base import BaseProvider
from .gemini_utils import map_messages, sanitize_params
from .gemini_stream import stream_responses_loop, handle_responses_api_sync

logger = logging.getLogger(__name__)

class GeminiProvider(BaseProvider):
    def __init__(self):
        self.auth = GeminiAuth()
        self.session = create_session()
        self.rate_limit_cache: Dict[str, float] = {}

    def handle_request(self, data: Dict[str, Any], handler: Any) -> None:
        """Entry point for Gemini requests."""
        if config.debug_mode:
            try:
                with open("/tmp/last_proxy_request.json", "w") as f:
                    json.dump(data, f)
            except: pass
        self._stream_request(data, handler)

    def handle_compact(self, data: Dict[str, Any], wfile: Any) -> None:
        """Handle context compaction requests using a fast Flash model."""
        try:
            token = self.auth.get_access_token()
            pid = self.auth.get_project_id(token)
            
            # Map messages from CompactionInput.input
            contents, system_instruction = map_messages(data.get('input', []), "gemini-2.5-flash-lite")
            
            # Add explicit compaction instructions
            compaction_prompt = data.get('instructions', "Summarize the conversation history concisely.")
            contents.append({
                'role': 'user', 
                'parts': [{'text': f"Perform context compaction. instructions: {compaction_prompt}"}]
            })
            
            request_body = {
                "model": "gemini-2.5-flash-lite", "project": pid,
                "request": {
                    "contents": contents,
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}
                }
            }
            if system_instruction: request_body['request']['systemInstruction'] = system_instruction

            url = f"{config.gemini_api_base}/v1internal:streamGenerateContent?alt=sse"
            headers = {'Authorization': f"Bearer {token}"}
            
            resp = self.session.post(url, data=orjson.dumps(request_body), headers=headers, stream=True, timeout=60)
            resp.raise_for_status()
            
            # Parse streaming response to collect final text
            final_text = ""
            for line in resp.iter_lines():
                if line.startswith(b'data: '):
                    try:
                        d = orjson.loads(line[6:])
                        parts = d.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [])
                        for p in parts:
                            if 'text' in p: final_text += p['text']
                    except: continue
            
            # Construct the CompactHistoryResponse
            # We return a single Compaction item (using the compaction_summary alias)
            compacted_item = {
                "type": "compaction_summary",
                "encrypted_content": final_text
            }
            
            result = {"output": [compacted_item]}
            
            wfile.send_response(200)
            wfile.send_header('Content-Type', 'application/json')
            wfile.end_headers()
            wfile.wfile.write(orjson.dumps(result))
            
        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            wfile.send_error(500, str(e))

    def _stream_request(self, req_data: Dict[str, Any], wfile: Any) -> None:
        is_stream = req_data.get('stream', req_data.get('_is_responses_api', False))
        requested_model = req_data.get('model', config.gemini_models[0])
        
        # Header deferral is managed inside _execute_stream
        for attempt in range(2):
            try:
                self._execute_stream(requested_model, req_data, wfile, headers_sent=False, display_model=requested_model)
                return
            except Exception as e:
                is_429 = False
                retry_delay = 0.5
                
                if getattr(e, 'response', None) is not None:
                    if e.response.status_code == 429:
                        is_429 = True
                        try:
                            error_data = e.response.json()
                            for detail in error_data.get('error', {}).get('details', []):
                                if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                    delay_str = detail.get('retryDelay', '0.5s')
                                    retry_delay = float(delay_str.rstrip('s'))
                                    break
                        except: pass
                
                if is_429 and attempt == 0:
                    wait_time = min(retry_delay, 2.0)
                    logger.warning(f"Model {requested_model} rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if not is_429:
                    self._report_error(wfile, e, is_stream)
                    return
                break

        fallback_model = 'gemini-2.5-flash'
        if requested_model == fallback_model:
            fallback_model = 'gemini-2.5-flash-lite'
            
        logger.info(f"Trying fallback model: {fallback_model}")
        try:
            self._execute_stream(fallback_model, req_data, wfile, headers_sent=False, display_model=requested_model)
        except Exception as e:
            setattr(e, 'model', requested_model)
            self._report_error(wfile, e, is_stream)

    def _report_error(self, wfile, error, is_stream):
        logger.error(f"Request failed: {error}")
        err_msg = str(error)
        
        # Determine error code
        codex_err = "internal_server_error"
        status_code = 500
        headers = {}
        
        if getattr(error, 'response', None) is not None:
            status_code = error.response.status_code
            if status_code == 429:
                codex_err = "usage_limit_exceeded"
                # Add model cap headers for deep parity
                headers["x-codex-model-cap-model"] = getattr(error, 'model', 'gemini')
                headers["x-codex-model-cap-reset-after-seconds"] = "60"
            elif status_code == 400:
                codex_err = "bad_request"
            
            # Forward promo message if present in upstream headers
            promo = error.response.headers.get('x-codex-promo-message')
            if promo: headers['x-codex-promo-message'] = promo

        if is_stream:
            try:
                try:
                    wfile.send_response(200)
                    wfile.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                    for k, v in headers.items(): wfile.send_header(k, v)
                    wfile.end_headers()
                except: pass
                
                fail_evt = {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {"code": codex_err, "message": err_msg}
                    }
                }
                wfile.wfile.write(b'data: ' + orjson.dumps(fail_evt) + b'\n\n')
                wfile.wfile.flush()
            except: pass
        else:
            try:
                wfile.send_response(status_code)
                for k, v in headers.items(): wfile.send_header(k, v)
                wfile.end_headers()
                wfile.wfile.write(err_msg.encode())
            except: pass

    def _execute_stream(self, model: str, req_data: Dict[str, Any], wfile: Any, headers_sent: bool = False, display_model: str = None) -> None:
        token = self.auth.get_access_token()
        pid = self.auth.get_project_id(token)
        
        # Subagent optimized model selection
        headers_dict = req_data.get('_headers') or {}
        subagent = headers_dict.get('x-openai-subagent')
        if subagent in ('compact', 'review') and not model.startswith('gemini-2.5-flash'):
            model = 'gemini-2.5-flash-lite'
            logger.info(f"Subagent '{subagent}' detected. Using faster model: {model}")

        # Native Parity: Personality Injection
        # We inject a 'virtual' system message that gemini_utils.map_messages understands
        personality = headers_dict.get('x-codex-personality') or config.default_personality
        messages = req_data.get('messages') or []
        # Insert at the beginning to ensure it's processed first
        messages.insert(0, {"role": "system", "content": f"personality:{personality}"})

        display_model = display_model or model
        contents, system_instruction = map_messages(messages, model)
        if not contents: contents = [{'role': 'user', 'parts': [{'text': '...'}]}]
        
        tools = []
        for t in (req_data.get('tools') or []):
            f = t.get('function') if t.get('type') == 'function' else (t if 'name' in t else None)
            if f: tools.append({'name': f['name'], 'description': f.get('description', ''), 'parameters': sanitize_params(f.get('parameters', {}))})
        
        max_tokens = req_data.get('max_tokens') or req_data.get('maxOutputTokens') or 8192
        gen_config = {"temperature": req_data.get('temperature', 1.0), "topP": 0.95, "topK": 64, "maxOutputTokens": max_tokens}
        
        # Thinking Config Parity (4-tier mapping)
        effort = (req_data.get('reasoning') or {}).get('effort', 'medium')
        budget = config.default_thinking_budget
        level = config.default_thinking_level
        
        if effort == 'low':
            budget, level = 4096, "LOW"
        elif effort == 'medium':
            budget, level = 16384, "MEDIUM"
        elif effort == 'high':
            budget, level = 32768, "HIGH"
        elif effort == 'xhigh':
            budget, level = 65536, "HIGH"
            # XHigh also injects a prompt hint via system instruction
            if system_instruction:
                system_instruction['parts'][0]['text'] += "\n\nCRITICAL: Provide an extremely thorough, step-by-step reasoning process before providing the final answer."

        if model.startswith('gemini-3'):
            gen_config["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": level}
        else:
            if budget > 0:
                gen_config["thinkingConfig"] = {"thinkingBudget": budget, "includeThoughts": True}

        request_body = {
            "model": model, "project": pid, "user_prompt_id": f"u-{int(time.time())}",
            "request": {
                "contents": contents, 
                "generationConfig": gen_config,
                "session_id": str(headers_dict.get('session_id') or req_data.get('conversation_id') or f"s-{int(time.time())}")
            }
        }
        
        # Grounding / Search
        if 'search' in (req_data.get('include') or []):
            if 'tools' not in request_body['request']: request_body['request']['tools'] = []
            request_body['request']['tools'].append({'googleSearch': {}})

        # Prompt Caching resource name validation
        cache_key = req_data.get('prompt_cache_key')
        if cache_key and cache_key.startswith('projects/'):
            request_body['request']['cachedContent'] = cache_key

        # Verbosity
        verbosity = (req_data.get('text') or {}).get('verbosity', 'medium')
        v_instr = None
        if verbosity == 'low': v_instr = "Respond very concisely and briefly."
        elif verbosity == 'high': v_instr = "Respond with very detailed and comprehensive information."
        
        if v_instr and system_instruction:
            system_instruction['parts'][0]['text'] += f"\n{v_instr}"

        # JSON Schema - Deep Support
        text_ctrl = req_data.get('text') or {}
        if text_ctrl.get('format', {}).get('type') == 'json_schema':
            schema = text_ctrl['format'].get('schema')
            if schema:
                gen_config["responseMimeType"] = "application/json"
                gen_config["responseSchema"] = schema
                # Force JSON instructions if not strictly enforced by API version
                json_prompt = "\nOutput the response strictly as JSON matching the requested schema."
                if system_instruction:
                    system_instruction['parts'][0]['text'] += json_prompt

        if system_instruction: request_body['request']['systemInstruction'] = system_instruction
        
        # Tools & Smart Proactivity
        if tools:
            if 'tools' not in request_body['request']: request_body['request']['tools'] = []
            request_body['request']['tools'].append({'functionDeclarations': tools})
            
            # GPT-5.2 Parity: Enforce ANY mode for tool calls to ensure proactivity
            tc = req_data.get('tool_choice', 'auto')
            mode = "AUTO" if tc == "auto" else ("NONE" if tc == "none" else "ANY")
            
            # Native parity: if it's a first turn or tool choice is auto, favor ANY to match gpt-5.2 proactivity
            if mode == "AUTO": mode = "ANY"
            
            request_body['request']['toolConfig'] = {"functionCallingConfig": {"mode": mode}}

        url = f"{config.gemini_api_base}/v1internal:streamGenerateContent?alt=sse"
        headers = {'Authorization': f"Bearer {token}", 'User-Agent': f'GeminiCLI/0.26.0/{model} (linux; x64)'}
        
        # Turn State forward/backward persistence
        if req_data.get('store'):
            headers['x-codex-store'] = 'true'
            ts = headers_dict.get('x-codex-turn-state')
            if ts: headers['x-codex-turn-state'] = ts

        logger.debug(f"Gemini Request Body: {orjson.dumps(request_body).decode()}")
        
        with self.session.post(url, data=orjson.dumps(request_body), headers=headers, stream=True, timeout=(10, 600)) as resp:
            if resp.status_code != 200:
                logger.error(f"Gemini API Error {resp.status_code}: {resp.text}")
                raise requests.exceptions.HTTPError(f"{resp.status_code} Error", response=resp)

            # --- DEEP PARITY: Forward turn state header ---
            turn_state = resp.headers.get('x-codex-turn-state')

            if not headers_sent:
                wfile.send_response(200)
                wfile.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                wfile.send_header('Cache-Control', 'no-cache')
                wfile.send_header('Connection', 'keep-alive')
                # Forward x-codex-turn-state for stateless parity
                if turn_state: wfile.send_header('x-codex-turn-state', turn_state)
                # Signal reasoning support
                wfile.send_header('x-reasoning-included', 'true')
                wfile.end_headers()
                headers_sent = True

            created_ts = int(time.time())
            if req_data.get('_is_responses_api'):
                return stream_responses_loop(resp, wfile, display_model, created_ts, headers_sent)
            
            return self._stream_chat_loop(resp, wfile, display_model, created_ts, headers_sent)

    def _stream_chat_loop(self, resp, wfile, model, created_ts, headers_sent):
        write, flush = wfile.wfile.write, wfile.wfile.flush
        full_response_content, finish_reason, sent_role = "", None, False
        chunk_base = {"id": "", "object": "chat.completion.chunk", "created": created_ts, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
        
        for line in resp.iter_lines():
            if not line:
                if headers_sent:
                    try: write(b': keep-alive\n\n'); flush()
                    except Exception: break
                continue
            if line.startswith(b'data: '):
                if line == b'data: [DONE]':
                    continue
                try:
                    data = orjson.loads(line[6:])
                    cand = data.get('response', {}).get('candidates', [{}])[0]
                    parts, finish = cand.get('content', {}).get('parts', []), cand.get('finishReason')
                    buf, tcs = "", None
                    for p in parts:
                        if 'text' in p: buf += p['text']
                        elif 'functionCall' in p:
                            if tcs is None: tcs = []
                            fc = p['functionCall']
                            tcs.append({"index": 0, "id": f"call_{created_ts}", "type": "function", "function": {"name": fc['name'], "arguments": orjson.dumps(fc['args']).decode()}})
                    
                    o_finish = {'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter', 'RECITATION': 'content_filter'}.get(finish, 'stop') if finish else None
                    if tcs: o_finish = 'tool_calls'
                    if not headers_sent:
                        if buf: full_response_content += buf
                        if o_finish: finish_reason = o_finish
                        continue

                    chunk_base["id"] = f"chatcmpl-{data.get('traceId', created_ts)}"
                    delta = chunk_base["choices"][0]["delta"]
                    delta.clear()
                    chunk_base["choices"][0]["finish_reason"] = o_finish
                    if not sent_role:
                        delta["role"] = "assistant"; sent_role = True
                    if buf: delta["content"] = buf
                    if tcs: delta["tool_calls"] = tcs
                    if delta or o_finish:
                        write(b'data: ' + orjson.dumps(chunk_base) + b'\n\n'); flush()
                except Exception: continue

        if not headers_sent:
            resp_obj = {"id": f"chatcmpl-{created_ts}", "object": "chat.completion", "created": created_ts, "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": full_response_content}, "finish_reason": finish_reason or "stop"}]}
            wfile.send_response(200)
            wfile.send_header('Content-Type', 'application/json')
            wfile.end_headers()
            wfile.wfile.write(orjson.dumps(resp_obj))