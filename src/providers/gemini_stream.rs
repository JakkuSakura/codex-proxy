use bytes::Bytes;
use futures::stream::Stream;
use futures::StreamExt;
use regex::Regex;
use serde_json::{Value, json};
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::debug;

use crate::config::CONFIG;

pub fn stream_responses_sse(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    resp_id: &str, model: &str, created_ts: i64, request_metadata: &Value,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let resp_id = resp_id.to_string();
    let model = model.to_string();
    let metadata = request_metadata.clone();
    let re_header = Regex::new(r"\*\*(.*?)\*\*").unwrap();

    Box::pin(async_stream::stream! {
        let mut seq_num: u64 = 0;
        let mut global_next_idx: usize = 0;
        let mut active_items: std::collections::HashMap<String, (Value, usize)> = std::collections::HashMap::new();
        let mut output_items: Vec<Value> = Vec::new();
        let mut final_usage: Option<Value> = None;

        fn now_ms() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0) }
        fn now_secs() -> i64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0) }

        let mut send_evt = |evt_type: &str, data: Value| -> Option<Bytes> {
            seq_num += 1;
            let mut evt = json!({"id": format!("evt_{}_{seq_num}", now_ms()), "object": "response.event", "type": evt_type, "created_at": now_secs(), "sequence_number": seq_num});
            if let Value::Object(map) = data { for (k, v) in map { evt.as_object_mut().unwrap().insert(k, v); } }
            if CONFIG.debug_mode { debug!("SSE OUT: {} - {}", evt_type, serde_json::to_string(&evt).unwrap_or_default()); }
            Some(Bytes::from(format!("event: {}\ndata: {}\n\n", evt_type, serde_json::to_string(&evt).unwrap_or_default())))
        };

        if let Some(bytes) = send_evt("response.created", json!({"response": {
            "id": &resp_id, "object": "response", "created_at": created_ts, "model": &model, "status": "in_progress",
            "temperature": metadata.get("temperature").and_then(|t| t.as_f64()).unwrap_or(1.0),
            "top_p": metadata.get("top_p").and_then(|t| t.as_f64()).unwrap_or(1.0),
            "tool_choice": metadata.get("tool_choice").and_then(|v| v.as_str()).unwrap_or("auto"),
            "tools": metadata.get("tools").cloned().unwrap_or(json!([])), "parallel_tool_calls": true,
            "store": metadata.get("store").and_then(|s| s.as_bool()).unwrap_or(true),
            "metadata": metadata.get("metadata").cloned().unwrap_or(json!({})), "output": [],
        }})) { yield Ok(bytes); }

        if let Some(bytes) = send_evt("models_etag", json!({"etag": "v1-gemini-gpt-5-2-parity"})) { yield Ok(bytes); }
        if let Some(bytes) = send_evt("server_reasoning_included", json!({"included": true})) { yield Ok(bytes); }
        if let Some(bytes) = send_evt("rate_limits", json!({"primary": null, "secondary": null, "credits": {"has_credits": true, "unlimited": false, "balance": null}})) { yield Ok(bytes); }

        let mut byte_stream = std::pin::pin!(byte_stream);
        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result { Ok(c) => c, Err(e) => { debug!("Stream error: {}", e); break; } };
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if !line.starts_with("data: ") || line == "data: [DONE]" { continue; }
                let data: Value = match serde_json::from_str(&line[6..]) { Ok(d) => d, Err(_) => continue };
                let resp_part = data.get("response").cloned().unwrap_or(data.clone());

                if let Some(usage) = resp_part.get("usageMetadata") {
                    let it = usage.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
                    let ot = usage.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
                    let rt = usage.get("thinkingTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
                    final_usage = Some(json!({"input_tokens": it, "input_tokens_details": {"cached_tokens": usage.get("cachedContentTokenCount").and_then(|v| v.as_u64()).unwrap_or(0)}, "output_tokens": ot, "output_tokens_details": {"reasoning_tokens": rt}, "total_tokens": it + ot}));
                }

                let candidates = match resp_part.get("candidates").and_then(|c| c.as_array()) { Some(c) if !c.is_empty() => c, _ => continue };
                let cand = &candidates[0];
                let parts = match cand.get("content").and_then(|c| c.get("parts")).and_then(|p| p.as_array()) { Some(p) => p, None => continue };
                let mut text_buf = String::new();
                let mut reasoning_text = String::new();

                for p in parts {
                    if let Some(fc) = p.get("functionCall") {
                        let fc_name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                        let fc_args = fc.get("args").cloned().unwrap_or(json!({}));
                        let args_str = serde_json::to_string(&fc_args).unwrap_or_default();
                        let itype = match fc_name { "shell" | "container.exec" | "shell_command" => "local_shell_call", _ => "function_call" };
                        let call_id = format!("call_{}_{}", now_ms(), global_next_idx);
                        let mut item = json!({"id": &call_id, "type": itype, "status": "in_progress", "name": fc_name, "arguments": &args_str, "call_id": &call_id});
                        if itype == "local_shell_call" { item.as_object_mut().unwrap().insert("action".into(), json!({"type": "exec", "command": fc_args.get("command").cloned().unwrap_or(json!([]))})); }
                        if let Some(sig) = p.get("thoughtSignature").and_then(|s| s.as_str()) { item.as_object_mut().unwrap().insert("thought_signature".into(), json!(sig)); }
                        let idx = global_next_idx; global_next_idx += 1;
                        if let Some(bytes) = send_evt("response.output_item.added", json!({"response_id": &resp_id, "output_index": idx, "item": &item})) { yield Ok(bytes); }
                        item.as_object_mut().unwrap().insert("status".into(), json!("completed"));
                        if let Some(bytes) = send_evt("response.output_item.done", json!({"response_id": &resp_id, "output_index": idx, "item": item})) { yield Ok(bytes); }
                        output_items.push(item);
                    } else {
                        let tv = p.get("thought");
                        let tx = p.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        let is_reasoning = tv.map(|v| v.is_string()).unwrap_or(false) || tv == Some(&Value::Bool(true));
                        let t_chunk = tv.and_then(|v| v.as_str()).unwrap_or(tx);
                        if is_reasoning && !t_chunk.is_empty() { reasoning_text.push_str(t_chunk); }
                        else if tv != Some(&Value::Bool(true)) && !tx.is_empty() { text_buf.push_str(tx); }
                    }
                }

                if !reasoning_text.is_empty() {
                    let entry = active_items.entry("reasoning".to_string()).or_insert_with(|| {
                        let idx = global_next_idx; global_next_idx += 1;
                        (json!({"id": format!("rs_{}_{}", now_ms(), idx), "type": "reasoning", "status": "in_progress", "summary": [], "content": [{"type": "reasoning_text", "text": ""}]}), idx)
                    });
                    let (ref mut item, _) = *entry;
                    let current = item["content"][0]["text"].as_str().unwrap_or("");
                    item["content"][0]["text"] = json!(format!("{current}{reasoning_text}"));
                }

                if !text_buf.is_empty() {
                    let entry = active_items.entry("message".to_string()).or_insert_with(|| {
                        let idx = global_next_idx; global_next_idx += 1;
                        let item = json!({"id": format!("msg_{}_{}", now_ms(), idx), "type": "message", "role": "assistant", "status": "in_progress", "content": [{"type": "output_text", "text": ""}]});
                        (item, idx)
                    });
                    let (ref mut item, idx) = *entry;
                    let current = item["content"][0]["text"].as_str().unwrap_or("");
                    item["content"][0]["text"] = json!(format!("{current}{text_buf}"));
                    if let Some(bytes) = send_evt("response.output_text.delta", json!({"response_id": &resp_id, "item_id": item["id"].clone(), "output_index": idx, "content_index": 0, "delta": text_buf})) { yield Ok(bytes); }

                    if let Some(r_entry) = active_items.get("reasoning") {
                        let full = r_entry.0["content"][0]["text"].as_str().unwrap_or("");
                        let headers: Vec<&str> = re_header.captures_iter(full).filter_map(|c| c.get(1)).map(|m| m.as_str()).collect();
                        if !headers.is_empty() {
                            let summaries: Vec<Value> = headers.iter().map(|h| json!({"type": "summary_text", "text": h})).collect();
                            let r_entry_mut = active_items.get_mut("reasoning").unwrap();
                            r_entry_mut.0.as_object_mut().unwrap().insert("summary".into(), json!(summaries));
                        }
                    }
                }

                if let Some(citations) = cand.get("citationMetadata").and_then(|c| c.get("citations")).and_then(|c| c.as_array()) {
                    for c in citations {
                        if let (Some(title), Some(uri)) = (c.get("title").and_then(|t| t.as_str()), c.get("uri").and_then(|u| u.as_str()))
                            && let Some(entry) = active_items.get_mut("message") {
                                let item = &mut entry.0;
                                let current = item["content"][0]["text"].as_str().unwrap_or("").to_string();
                                item["content"][0]["text"] = json!(format!("{current}\n({title}) {uri}"));
                            }
                    }
                }
            }
        }

        let keys: Vec<String> = active_items.keys().cloned().collect();
        for key in keys {
            if let Some((mut item, idx)) = active_items.remove(&key) {
                item.as_object_mut().unwrap().insert("status".into(), json!("completed"));
                if let Some(bytes) = send_evt("response.output_item.done", json!({"response_id": &resp_id, "output_index": idx, "item": item})) { yield Ok(bytes); }
                output_items.push(item);
            }
        }

        if let Some(bytes) = send_evt("response.completed", json!({"response": {
            "id": &resp_id, "object": "response", "status": "completed", "model": &model,
            "created_at": created_ts, "completed_at": now_secs(),
            "usage": final_usage.unwrap_or(json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})),
            "output": output_items,
            "temperature": metadata.get("temperature").and_then(|t| t.as_f64()).unwrap_or(1.0),
            "top_p": metadata.get("top_p").and_then(|t| t.as_f64()).unwrap_or(1.0),
            "tool_choice": metadata.get("tool_choice").and_then(|v| v.as_str()).unwrap_or("auto"),
            "tools": metadata.get("tools").cloned().unwrap_or(json!([])), "parallel_tool_calls": true,
            "store": metadata.get("store").and_then(|s| s.as_bool()).unwrap_or(true),
            "metadata": metadata.get("metadata").cloned().unwrap_or(json!({})),
        }})) { yield Ok(bytes); }
    })
}
