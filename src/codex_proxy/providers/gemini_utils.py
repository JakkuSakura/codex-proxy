import json
import logging
from typing import Dict, Any, List, Optional, Tuple

try:
    import orjson
except ImportError:
    import json as orjson

logger = logging.getLogger(__name__)

def sanitize_params(params: Dict) -> Dict:
    if not isinstance(params, dict): return params
    return {k: sanitize_params(v) for k, v in params.items() if k not in {'additionalProperties', 'title', 'default', 'minItems', 'maxItems', 'uniqueItems'}}

def map_messages(messages: List[Dict], model_name: str) -> Tuple[List[Dict], Optional[Dict]]:
    contents = []
    system_parts = []
    
    # Map to store tool_call_id -> function_name from HISTORY
    tool_call_map = {}
    for m in messages:
        if m.get('tool_calls'):
            for tc in m['tool_calls']:
                tc_id = tc.get('id')
                fn_name = tc.get('function', {}).get('name')
                if tc_id and fn_name:
                    tool_call_map[tc_id] = fn_name

    for m in messages:
        role = m.get('role')
        content_raw = m.get('content')
        reasoning = m.get('reasoning_content')
        
        if role == 'system' or role == 'developer':
            if isinstance(content_raw, str) and content_raw:
                system_parts.append({'text': content_raw})
            elif isinstance(content_raw, list):
                for part in content_raw:
                    if part.get('type') in ('text', 'input_text'):
                        system_parts.append({'text': part.get('text', '')})
            continue
        
        parts = []
        if reasoning:
            # IMPORTANT: Do not strip reasoning text
            parts.append({'text': reasoning, 'thought': True})
        
        if content_raw:
            if isinstance(content_raw, str):
                parts.append({'text': content_raw})
            elif isinstance(content_raw, list):
                for cp in content_raw:
                    ctype = cp.get('type')
                    if ctype in ('text', 'input_text', 'output_text'):
                        parts.append({'text': cp.get('text', '')})
                    elif ctype in ('image', 'input_image'):
                        url = cp.get('image_url')
                        if url and url.startswith('data:'):
                            try:
                                # Native parity: wrap images with <image> tags in text parts
                                parts.append({'text': '<image>'})
                                # Parse data:image/png;base64,....
                                header, data = url.split(',', 1)
                                mime = header.split(':', 1)[1].split(';', 1)[0]
                                parts.append({'inlineData': {'mimeType': mime, 'data': data}})
                                parts.append({'text': '</image>'})
                            except: pass
        
        if m.get('tool_calls'):
            # Extract thought signature from the message
            msg_thought_sig = m.get('thought_signature') or m.get('thoughtSignature')
            
            for tc in m['tool_calls']:
                try:
                    args = tc['function']['arguments']
                    if isinstance(args, str): args = orjson.loads(args)
                except: args = {}
                
                fn_name = tc['function']['name']
                part = {'functionCall': {'name': fn_name, 'args': args}}
                
                # Use provided signature or a synthetic skip-validator
                sig_to_use = msg_thought_sig or "skip_thought_signature_validator"
                part['thoughtSignature'] = sig_to_use
                
                parts.append(part)
        
        if role == 'tool':
            tc_id = m.get('tool_call_id')
            fn_name = tool_call_map.get(tc_id, m.get('name', 'unknown'))
            # Native FR support
            resp_part = {'functionResponse': {'name': fn_name, 'response': {'content': content_raw or ''}}}
            
            # Check if we can append to the last turn
            if contents and contents[-1]['role'] == 'user' and any('functionResponse' in p for p in contents[-1]['parts']):
                contents[-1]['parts'].append(resp_part)
                continue
            else:
                role = 'user'
                parts = [resp_part]
        
        if role == 'assistant':
            role = 'model'
        else:
            role = 'user'
        
        if parts:
            # Check if we can merge with the previous turn if it has the same role
            if contents and contents[-1]['role'] == role:
                contents[-1]['parts'].extend(parts)
            else:
                content_obj = {'role': role, 'parts': parts}
                contents.append(content_obj)
    
    system_instruction = {'parts': system_parts} if system_parts else None
    return contents, system_instruction