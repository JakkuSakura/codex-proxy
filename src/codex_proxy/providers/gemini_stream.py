import json
import time
import logging
import re
from typing import Dict, Any, List, Optional

try:
    import orjson
except ImportError:
    import json as orjson

from ..config import config

logger = logging.getLogger(__name__)

# Enhanced pattern to split reasoning into professional summaries
REASONING_HEADER_PATTERN = re.compile(r'\*\*(.*?)\*\*')

def stream_responses_loop(resp, wfile, model, created_ts, headers_sent=False):
    """
    State-aware Multi-Item Streaming Loop with Deep Native Parity.
    Handles reasoning summaries, content split, telemetry events, and quota snapshots.
    """
    write = wfile.wfile.write
    flush = wfile.wfile.flush
    json_loads = orjson.loads
    json_dumps = orjson.dumps
    
    resp_id = f"resp_{created_ts}"
    seq_num = 0
    
    def send_evt(evt_type: str, data: Dict):
        nonlocal seq_num
        seq_num += 1
        evt = {
            "id": f"evt_{int(time.time()*1000)}_{seq_num}",
            "object": "response.event",
            "type": evt_type,
            "created_at": int(time.time()),
            "sequence_number": seq_num,
            **data
        }
        if config.debug_mode:
            logger.debug(f"SSE OUT: {evt_type} - {json_dumps(evt).decode()}")
        # Fully aligned with eventsource_stream and codex-rs serde
        write(b'event: ' + evt_type.encode() + b'\ndata: ' + json_dumps(evt) + b'\n\n')
        flush()

    # 1. Start response
    send_evt("response.created", {
        "response": {
            "id": resp_id, "object": "response", "created_at": created_ts,
            "model": model, "status": "in_progress"
        }
    })
    
    # 2. Native Telemetry: Models Etag
    send_evt("models_etag", {"etag": "v1-gemini-deep-parity"})
    
    # 3. Native Telemetry: Server Reasoning Included
    # Gemini internal API always accounts for its own reasoning tokens in candidatesTokenCount
    send_evt("server_reasoning_included", {"included": True})

    # 4. Native Telemetry: Rate Limits (Beset efforts parsing)
    # We map Gemini's quota state if headers are present
    h = resp.headers
    snapshot = {
        "primary": None, "secondary": None, 
        "credits": {"has_credits": True, "unlimited": False, "balance": None}
    }
    try:
        # Map BESPOKE headers if the internal API provides them (simulated for parity if missing)
        p_used = h.get('x-codex-primary-used-percent')
        if p_used:
            snapshot["primary"] = {
                "used_percent": float(p_used),
                "window_minutes": int(h.get('x-codex-primary-window-minutes', 60)),
                "resets_at": int(h.get('x-codex-primary-reset-at', time.time() + 3600))
            }
    except: pass
    send_evt("rate_limits", snapshot)

    active_items = {} # type -> {item, index}
    output_items = []
    final_usage = None
    priority_call_checked = False
    all_citations = set()
    search_triggered = False
    
    summary_buffer = ""
    detected_headers = set()
    current_summary_index = -1
    
    global_next_idx = 0

    def _ensure_item(itype, item_id_prefix='msg'):
        nonlocal global_next_idx
        if itype in active_items:
            return active_items[itype]
        
        idx = global_next_idx
        global_next_idx += 1
        
        if itype == 'reasoning':
            item = {
                "id": f"rs_{int(time.time()*1000)}_{idx}", 
                "type": "reasoning", 
                "summary": [], 
                "content": [{"type": "reasoning_text", "text": ""}]
            }
        elif itype == 'message':
            item = {
                "id": f"{item_id_prefix}_{int(time.time()*1000)}_{idx}", 
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": ""}]
            }
        else: return None
        
        active_items[itype] = {"item": item, "index": idx}
        send_evt("response.output_item.added", {"response_id": resp_id, "output_index": idx, "item": item})
        return active_items[itype]

    def _close_item(itype):
        if itype not in active_items: return
        data = active_items.pop(itype)
        item, idx = data['item'], data['index']
        item['status'] = 'completed'
        send_evt("response.output_item.done", {"response_id": resp_id, "output_index": idx, "item": item})
        output_items.append(item)

    for line in resp.iter_lines():
        if not line or not line.startswith(b'data: ') or line == b'data: [DONE]': continue
            
        try:
            raw_json = line[6:]
            data = json_loads(raw_json)
            resp_part = data.get('response', {})
            
            # Usage
            usage = resp_part.get('usageMetadata')
            if usage:
                it, ot, rt = usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0), usage.get("thinkingTokenCount", 0)
                final_usage = {
                    "input_tokens": it, 
                    "input_tokens_details": {"cached_tokens": usage.get("cachedContentTokenCount", 0)},
                    "output_tokens": ot, 
                    "output_tokens_details": {"reasoning_tokens": rt},
                    "total_tokens": it + ot
                }
            
            cand = resp_part.get('candidates', [{}])[0]
            parts = cand.get('content', {}).get('parts', [])
            
            # Grounding Calls
            g_meta = cand.get('groundingMetadata', {})
            queries = g_meta.get('queries', [])
            if queries and not search_triggered:
                idx = global_next_idx
                global_next_idx += 1
                ws_item = {"id": f"ws_{int(time.time()*1000)}_{idx}", "type": "web_search_call", "status": "completed", "action": {"type": "search", "queries": queries}}
                send_evt("response.output_item.added", {"response_id": resp_id, "output_index": idx, "item": ws_item})
                send_evt("response.output_item.done", {"response_id": resp_id, "output_index": idx, "item": ws_item})
                output_items.append(ws_item)
                search_triggered = True

            # Citations
            c_meta = cand.get('citationMetadata', {})
            for c in c_meta.get('citations', []):
                if c.get('uri'): all_citations.add(f"({c.get('title', 'Source')}) {c.get('uri')}")
            for chunk in g_meta.get('groundingChunks', []):
                if chunk.get('web', {}).get('uri'): all_citations.add(f"({chunk['web'].get('title') or 'Search Result'}) {chunk['web']['uri']}")

            for p in parts:
                if 'functionCall' in p:
                    # Do NOT close reasoning/message immediately to avoid UI status flickering
                    fc = p['functionCall']
                    idx = global_next_idx
                    global_next_idx += 1
                    cid = f"call_{int(time.time()*1000)}_{idx}"
                    itype = "local_shell_call" if fc['name'] in ('shell', 'container.exec', 'shell_command') else "function_call"
                    fc_item = {"id": cid, "type": itype, "status": "completed", "name": fc['name'], "arguments": json_dumps(fc['args']).decode(), "call_id": cid}
                    if itype == "local_shell_call": fc_item["action"] = {"type": "exec", "command": fc['args'].get('command', [])}
                    send_evt("response.output_item.added", {"response_id": resp_id, "output_index": idx, "item": fc_item})
                    send_evt("response.output_item.done", {"response_id": resp_id, "output_index": idx, "item": fc_item})
                    output_items.append(fc_item)
                
                else:
                    tv, tx = p.get('thought'), p.get('text')
                    is_r = (isinstance(tv, str) or (tv is True and tx))
                    t_chunk = tv if isinstance(tv, str) else (tx if tv is True else None)
                    if tv is True: tx = None
                    
                    if is_r and t_chunk:
                        idata = _ensure_item('reasoning')
                        item, idx = idata['item'], idata['index']
                        
                        # Auto-Header injection for professional status
                        if not item['summary'] and not t_chunk.strip().startswith('**'):
                            initial_h = "Thinking"
                            item['summary'].append({"type": "summary_text", "text": initial_h})
                            send_evt("response.reasoning_summary_part.added", {"response_id": resp_id, "item_id": item["id"], "output_index": idx, "summary_index": 0})
                            send_evt("response.reasoning_summary_text.delta", {"response_id": resp_id, "item_id": item["id"], "output_index": idx, "summary_index": 0, "delta": initial_h})
                            detected_headers.add(initial_h)
                            current_summary_index = 0

                        item['content'][0]['text'] += t_chunk
                        send_evt("response.reasoning_text.delta", {"response_id": resp_id, "item_id": item["id"], "output_index": idx, "content_index": 0, "delta": t_chunk})
                        
                        summary_buffer += t_chunk
                        # Better header detection: only look for full **Header** tags
                        headers = re.findall(r'\*\*(.*?)\*\*', summary_buffer)
                        for h in headers:
                            if h and h not in detected_headers:
                                detected_headers.add(h)
                                current_summary_index += 1
                                item['summary'].append({"type": "summary_text", "text": h})
                                send_evt("response.reasoning_summary_part.added", {"response_id": resp_id, "item_id": item["id"], "output_index": idx, "summary_index": current_summary_index})
                                send_evt("response.reasoning_summary_text.delta", {"response_id": resp_id, "item_id": item["id"], "output_index": idx, "summary_index": current_summary_index, "delta": h})

                    if tx:
                        idata = _ensure_item('message')
                        item, idx = idata['item'], idata['index']
                        item['content'][0]['text'] += tx
                        send_evt("response.output_text.delta", {"response_id": resp_id, "item_id": item["id"], "output_index": idx, "content_index": 0, "delta": tx})
                        
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
            continue

    for itype in list(active_items.keys()):
        _close_item(itype)

    if all_citations and output_items:
        for item in reversed(output_items):
            if item.get('type') == 'message':
                item['content'][0]['text'] += ("\n\nSources:\n" + "\n".join(sorted(all_citations)))
                break

    # 5. Final Token Usage Summary
    if final_usage:
        usage_info = {
            "total_token_usage": {
                "input_tokens": final_usage['input_tokens'],
                "cached_input_tokens": final_usage['input_tokens_details']['cached_tokens'],
                "output_tokens": final_usage['output_tokens'],
                "reasoning_output_tokens": final_usage['output_tokens_details']['reasoning_tokens'],
                "total_tokens": final_usage['total_tokens']
            },
            "last_token_usage": {
                "input_tokens": final_usage['input_tokens'],
                "cached_input_tokens": final_usage['input_tokens_details']['cached_tokens'],
                "output_tokens": final_usage['output_tokens'],
                "reasoning_output_tokens": final_usage['output_tokens_details']['reasoning_tokens'],
                "total_tokens": final_usage['total_tokens']
            },
            "model_context_window": 1000000 
        }
        send_evt("token_count", {"info": usage_info})

    # 6. Final response.completed
    final_resp_data = {
        "id": resp_id, "object": "response", "status": "completed", "model": model,
        "created_at": created_ts, "completed_at": int(time.time()), 
        "usage": final_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 
        "output": output_items
    }
    send_evt("response.completed", {"response": final_resp_data})
    flush()

def handle_responses_api_sync(resp, wfile, model, created_ts):
    """Deep Parity Sync handler."""
    json_loads, json_dumps = orjson.loads, orjson.dumps
    all_parts, final_usage, all_citations, search_queries = [], None, set(), []
    for line in resp.iter_lines():
        if not line or not line.startswith(b'data: ') or line == b'data: [DONE]': continue
        try:
            data = json_loads(line[6:])
            resp_part = data.get('response', {})
            usage = resp_part.get('usageMetadata')
            if usage:
                it, ot, rt = usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0), usage.get("thinkingTokenCount", 0)
                final_usage = {"input_tokens": it, "input_tokens_details": {"cached_tokens": usage.get("cachedContentTokenCount", 0)}, "output_tokens": ot, "output_tokens_details": {"reasoning_tokens": rt}, "total_tokens": it + ot}
            all_parts.extend(resp_part.get('candidates', [{}])[0].get('content', {}).get('parts', []))
            for c in resp_part.get('candidates', [{}])[0].get('citationMetadata', {}).get('citations', []):
                if c.get('uri'): all_citations.add(f"({c.get('title', 'Source')}) {c.get('uri')}")
            g_meta = resp_part.get('candidates', [{}])[0].get('groundingMetadata', {})
            search_queries.extend(g_meta.get('queries', []))
            for chunk in g_meta.get('groundingChunks', []):
                if chunk.get('web', {}).get('uri'): all_citations.add(f"({chunk['web'].get('title') or 'Search Result'}) {chunk['web']['uri']}")
        except Exception: continue
    output_items = []
    if search_queries: output_items.append({"id": f"ws_{created_ts}", "type": "web_search_call", "status": "completed", "action": {"type": "search", "queries": search_queries}})
    curr_msg, curr_rs = None, None
    for p in all_parts:
        if 'functionCall' in p:
            fc = p['functionCall']
            itype = "local_shell_call" if fc['name'] in ('shell', 'container.exec', 'shell_command') else "function_call"
            item = {"id": f"call_{int(time.time()*1000)}_{len(output_items)}", "type": itype, "status": "completed", "name": fc['name'], "arguments": json_dumps(fc['args']).decode(), "call_id": f"call_{created_ts}"}
            if itype == "local_shell_call": item["action"] = {"type": "exec", "command": fc['args'].get('command', [])}
            output_items.append(item)
        else:
            tv, tx = p.get('thought'), p.get('text')
            is_r = (isinstance(tv, str) or (tv is True and tx))
            t_chunk = tv if isinstance(tv, str) else (tx if tv is True else None)
            if tv is True: tx = None
            if is_r and t_chunk:
                if not curr_rs: curr_rs = {"id": f"rs_{created_ts}", "type": "reasoning", "status": "completed", "content": [{"type": "reasoning_text", "text": ""}], "summary": []}
                curr_rs['content'][0]['text'] += t_chunk
                h = re.findall(r'\*\*(.*?)\*\*', curr_rs['content'][0]['text'])
                if h: curr_rs['summary'] = [{"type": "summary_text", "text": x} for x in h]
                if curr_rs not in output_items: output_items.append(curr_rs)
            if tx:
                if not curr_msg: curr_msg = {"id": f"msg_{created_ts}", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}
                curr_msg['content'][0]['text'] += tx
                if curr_msg not in output_items: output_items.append(curr_msg)
    if all_citations and curr_msg: curr_msg['content'][0]['text'] += ("\n\nSources:\n" + "\n".join(sorted(all_citations)))
    resp_obj = {"id": f"resp_{created_ts}", "object": "response", "created": created_ts, "model": model, "status": "completed", "usage": final_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "output": output_items}
    wfile.send_response(200)
    wfile.send_header('Content-Type', 'application/json')
    wfile.end_headers()
    wfile.wfile.write(json_dumps(resp_obj))