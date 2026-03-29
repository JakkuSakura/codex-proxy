use bytes::Bytes;
use futures::stream::Stream;
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::debug;

pub fn stream_responses_sse(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: &str,
    created_ts: i64,
    request_metadata: &Value,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let model = model.to_string();
    let metadata = request_metadata.clone();
    let resp_id = format!("resp_{created_ts}");

    Box::pin(async_stream::stream! {
        let mut seq_num: u64 = 0;
        let mut full_content = String::new();
        let mut message: Option<Value> = None;
        let mut message_idx: i64 = -1;
        let mut next_idx: usize = 0;
        let mut tool_calls: HashMap<usize, (Value, usize)> = HashMap::new();

        fn now_ms() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0) }
        fn now_secs() -> i64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0) }

        let mut send_event = |evt_type: &str, data: Value| -> Bytes {
            seq_num += 1;
            let mut evt = json!({
                "id": format!("evt_{}_{seq_num}", now_ms()),
                "object": "response.event",
                "type": evt_type,
                "created_at": now_secs(),
                "sequence_number": seq_num,
            });
            if let Value::Object(map) = data {
                for (k, v) in map {
                    evt.as_object_mut().unwrap().insert(k, v);
                }
            }
            let payload = format!("event: {}\ndata: {}\n\n", evt_type, serde_json::to_string(&evt).unwrap_or_default());
            Bytes::from(payload)
        };

        let response_obj = json!({
            "id": &resp_id, "object": "response", "created_at": created_ts, "model": &model,
            "status": "in_progress",
            "temperature": metadata.get("temperature").and_then(|t| t.as_f64()).unwrap_or(1.0),
            "top_p": metadata.get("top_p").and_then(|t| t.as_f64()).unwrap_or(1.0),
            "tool_choice": metadata.get("tool_choice").and_then(|v| v.as_str()).unwrap_or("auto"),
            "tools": metadata.get("tools").cloned().unwrap_or(json!([])),
            "parallel_tool_calls": true,
            "store": metadata.get("store").and_then(|s| s.as_bool()).unwrap_or(true),
            "metadata": metadata.get("metadata").cloned().unwrap_or(json!({})),
            "output": [],
        });

        yield Ok(send_event("response.created", json!({"response": response_obj.clone()})));

        let mut byte_stream = std::pin::pin!(byte_stream);
        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => { debug!("ZAI stream error: {}", e); break; }
            };
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if !line.starts_with("data: ") { continue; }
                if line == "data: [DONE]" { break; }
                let data_str = &line[6..];
                let data: Value = match serde_json::from_str(data_str) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                debug!("ZAI STREAM DELTA: {}", data_str);

                let choices = match data.get("choices").and_then(|c| c.as_array()) {
                    Some(c) if !c.is_empty() => c,
                    _ => continue,
                };
                let choice = &choices[0];
                let delta = choice.get("delta").cloned().unwrap_or(json!({}));

                // Handle tool calls
                if let Some(tc_deltas) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc_delta in tc_deltas {
                        let idx = tc_delta.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                        if let std::collections::hash_map::Entry::Vacant(e) = tool_calls.entry(idx) {
                            let output_idx = next_idx;
                            next_idx += 1;
                            let call_id = tc_delta.get("id")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| format!("call_{}_{}", now_ms(), output_idx));
                            let tool_call = json!({
                                "id": &call_id, "type": "function_call", "status": "in_progress",
                                "name": "", "arguments": "", "call_id": &call_id,
                            });
                            yield Ok(send_event("response.output_item.added", json!({
                                "response_id": &resp_id, "output_index": output_idx, "item": tool_call.clone(),
                            })));
                            e.insert((tool_call, output_idx));
                        }

                        if let Some(tc_entry) = tool_calls.get_mut(&idx) {
                            let tc = &mut tc_entry.0;
                            if let Some(fn_delta) = delta.get("function") {
                                if let Some(name) = fn_delta.get("name").and_then(|n| n.as_str()) {
                                    let current = tc.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                                    let obj = tc.as_object_mut().unwrap();
                                    obj.insert("name".into(), json!(format!("{current}{name}")));
                                }
                                if let Some(args_part) = fn_delta.get("arguments") {
                                    let args_str = if args_part.is_string() {
                                        args_part.as_str().unwrap().to_string()
                                    } else {
                                        serde_json::to_string(args_part).unwrap_or_default()
                                    };
                                    let current = tc.get("arguments").and_then(|a| a.as_str()).unwrap_or("").to_string();
                                    let obj = tc.as_object_mut().unwrap();
                                    obj.insert("arguments".into(), json!(format!("{current}{args_str}")));
                                }
                            }
                        }
                    }
                }

                // Handle content
                if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                    && !content.is_empty() {
                        full_content.push_str(content);

                        if message.is_none() {
                            message_idx = next_idx as i64;
                            next_idx += 1;
                            let item_id = format!("msg_{}_{}", now_ms(), message_idx);
                            let msg = json!({
                                "id": &item_id, "type": "message", "role": "assistant",
                                "status": "in_progress", "content": [{"type": "output_text", "text": ""}],
                            });
                            yield Ok(send_event("response.output_item.added", json!({
                                "response_id": &resp_id, "output_index": message_idx, "item": msg.clone(),
                            })));
                            message = Some(msg);
                        }

                        if let Some(ref msg) = message {
                            let item_id = msg["id"].as_str().unwrap_or("");
                            yield Ok(send_event("response.output_text.delta", json!({
                                "response_id": &resp_id, "item_id": item_id,
                                "output_index": message_idx, "content_index": 0, "delta": content,
                            })));
                            if let Some(ref mut m) = message
                                && let Some(content_arr) = m.get_mut("content").and_then(|c| c.as_array_mut())
                                    && let Some(first) = content_arr.first_mut()
                                        && let Some(obj) = first.as_object_mut() {
                                            obj.insert("text".into(), json!(full_content.clone()));
                                        }
                        }
                    }
            }
        }

        // Finalize
        let mut final_output: Vec<Value> = Vec::new();
        let mut items_to_close: Vec<(usize, Value)> = Vec::new();

        if let Some(msg) = message {
            items_to_close.push((message_idx as usize, msg));
        }
        for (item, output_idx) in tool_calls.values() {
            items_to_close.push((*output_idx, item.clone()));
        }
        items_to_close.sort_by_key(|(idx, _)| *idx);

        for (out_idx, mut item) in items_to_close {
            item.as_object_mut().unwrap().insert("status".into(), json!("completed"));

            if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                if name == "shell" || name == "container.exec" || name == "shell_command" {
                    item.as_object_mut().unwrap().insert("type".into(), json!("local_shell_call"));
                    if let Ok(args) = serde_json::from_str::<Value>(
                        item.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}")
                    ) {
                        item.as_object_mut().unwrap().insert("action".into(), json!({
                            "type": "exec", "command": args.get("command").cloned().unwrap_or(json!([]))
                        }));
                    }
                }
            }

            yield Ok(send_event("response.output_item.done", json!({
                "response_id": &resp_id, "output_index": out_idx, "item": &item,
            })));
            final_output.push(item);
        }

        let mut final_resp = response_obj.clone();
        let obj = final_resp.as_object_mut().unwrap();
        obj.insert("status".into(), json!("completed"));
        obj.insert("completed_at".into(), json!(now_secs()));
        obj.insert("usage".into(), json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}));
        obj.insert("output".into(), json!(final_output));

        yield Ok(send_event("response.completed", json!({"response": final_resp})));
    })
}
