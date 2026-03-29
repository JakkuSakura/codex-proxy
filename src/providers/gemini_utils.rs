use serde_json::{Value, json};

pub fn sanitize_params(params: &Value) -> Value {
    match params {
        Value::Object(map) => {
            let filtered: serde_json::Map<String, Value> = map.iter()
                .filter(|(k, _)| !["additionalProperties", "title", "default", "minItems", "maxItems", "uniqueItems"].contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), sanitize_params(v)))
                .collect();
            Value::Object(filtered)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(sanitize_params).collect()),
        _ => params.clone(),
    }
}

pub fn map_messages(messages: &[Value], _model_name: &str) -> (Vec<Value>, Option<Value>) {
    let mut contents: Vec<Value> = Vec::new();
    let mut system_parts: Vec<Value> = Vec::new();
    let mut tool_call_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for m in messages {
        if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                if let (Some(id), Some(fn_obj)) = (tc.get("id").and_then(|i| i.as_str()), tc.get("function"))
                    && let Some(name) = fn_obj.get("name").and_then(|n| n.as_str()) { tool_call_map.insert(id.to_string(), name.to_string()); }
            }
        }
    }
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content_raw = m.get("content");
        let reasoning = m.get("reasoning_content").and_then(|r| r.as_str()).unwrap_or("");
        if role == "system" || role == "developer" {
            match content_raw {
                Some(Value::String(s)) if !s.is_empty() => { system_parts.push(json!({"text": s})); }
                Some(Value::Array(arr)) => {
                    for part in arr {
                        if let Some(ptype) = part.get("type").and_then(|t| t.as_str()) {
                            if (ptype == "text" || ptype == "input_text") && !part.get("text").and_then(|t| t.as_str()).unwrap_or("").is_empty() {
                                system_parts.push(json!({"text": part["text"]}));
                            }
                        } else if let Some(s) = part.as_str() {
                            system_parts.push(json!({"text": s}));
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        let mut parts: Vec<Value> = Vec::new();
        if !reasoning.is_empty() { parts.push(json!({"text": reasoning, "thought": true})); }
        if let Some(content) = content_raw {
            match content {
                Value::String(s) => parts.push(json!({"text": s})),
                Value::Array(arr) => {
                    for cp in arr {
                        let ctype = cp.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if ctype == "text" || ctype == "input_text" || ctype == "output_text" {
                            if let Some(text) = cp.get("text").and_then(|t| t.as_str()) { parts.push(json!({"text": text})); }
                        } else if (ctype == "image" || ctype == "input_image")
                            && let Some(url) = cp.get("image_url").and_then(|u| u.as_str())
                                && url.starts_with("data:") { parts.push(json!({"text": "<image>"})); parts.push(json!({"text": "</image>"})); }
                    }
                }
                _ => {}
            }
        }
        if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
            let thought_sig = m.get("thought_signature").or_else(|| m.get("thoughtSignature")).and_then(|s| s.as_str()).unwrap_or("skip_thought_signature_validator");
            for tc in tcs {
                let fn_name = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).unwrap_or("unknown");
                let args_str = tc.get("function").and_then(|f| f.get("arguments")).and_then(|a| a.as_str()).unwrap_or("{}");
                let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                parts.push(json!({"functionCall": {"name": fn_name, "args": args}, "thoughtSignature": thought_sig}));
            }
        }
        let gemini_role = if role == "assistant" { "model" } else { "user" };
        if role == "tool" {
            let tc_id = m.get("tool_call_id").and_then(|i| i.as_str()).unwrap_or("unknown");
            let fn_name = tool_call_map.get(tc_id).map(|s| s.as_str()).or_else(|| m.get("name").and_then(|n| n.as_str())).unwrap_or("unknown");
            let resp_part = json!({"functionResponse": {"name": fn_name, "response": {"content": content_raw.unwrap_or(&Value::String(String::new())).clone()}}});
            let tool_role = "user";
            if let Some(last) = contents.last_mut()
                && last.get("role").and_then(|r| r.as_str()) == Some(tool_role)
                    && let Some(last_parts) = last.get_mut("parts").and_then(|p| p.as_array_mut())
                        && last_parts.iter().any(|p| p.get("functionResponse").is_some()) { last_parts.push(resp_part); continue; }
            contents.push(json!({"role": tool_role, "parts": [resp_part]}));
            continue;
        }
        if !parts.is_empty() {
            if let Some(last) = contents.last_mut()
                && last.get("role").and_then(|r| r.as_str()) == Some(gemini_role)
                    && let Some(last_parts) = last.get_mut("parts").and_then(|p| p.as_array_mut()) { last_parts.extend(parts.clone()); continue; }
            contents.push(json!({"role": gemini_role, "parts": parts}));
        }
    }
    let system_instruction = if system_parts.is_empty() { None } else { Some(json!({"parts": system_parts})) };
    (contents, system_instruction)
}
