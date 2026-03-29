use serde_json::{Value, json};

pub fn normalize(data: &mut Value) {
    let mut messages = Vec::new();

    if let Some(instructions) = data.get("instructions") {
        let content = extract_text(instructions);
        if !content.is_empty() {
            messages.push(json!({"role": "system", "content": content}));
        }
    }

    if let Some(input) = data.get("input") {
        let items = match input {
            Value::String(s) => vec![json!(s)],
            Value::Array(arr) => arr.clone(),
            _ => Vec::new(),
        };
        for item in &items {
            process_input_item(item, &mut messages);
        }
    }

    data["messages"] = Value::Array(messages);
    if data.get("previous_response_id").is_none() { data["previous_response_id"] = Value::Null; }
    if data.get("store").is_none() { data["store"] = Value::Bool(false); }
    if data.get("metadata").is_none() { data["metadata"] = json!({}); }

    if let Some(tools) = data.get("tools").and_then(|t| t.as_array()) {
        data["tools"] = Value::Array(normalize_tools(tools));
    }
}

fn extract_text(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr.iter().map(|item| match item {
            Value::String(s) => s.clone(),
            Value::Object(obj) => obj.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(),
            _ => String::new(),
        }).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

fn process_input_item(item: &Value, messages: &mut Vec<Value>) {
    let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("message");
    match item_type {
        "message" | "agentMessage" => process_message(item, messages),
        "reasoning" => process_reasoning(item, messages),
        "function_call" | "commandExecution" | "local_shell_call" | "fileChange"
        | "custom_tool_call" | "web_search_call" => process_tool_call(item, messages),
        "function_call_output" | "commandExecutionOutput" | "fileChangeOutput"
        | "custom_tool_call_output" => process_tool_output(item, messages),
        _ => {}
    }
}

fn process_message(item: &Value, messages: &mut Vec<Value>) {
    let role = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
    let role = if role == "developer" { "system" } else { role };
    let content_raw = item.get("content");
    let reasoning_content = item.get("reasoning_content").and_then(|r| r.as_str()).unwrap_or("");
    let content = extract_content_text(content_raw);

    if role == "assistant" || role == "model" {
        let idx = ensure_last_assistant(messages);
        let amsg = &mut messages[idx];
        let obj = amsg.as_object_mut().unwrap();
        let current_content = obj.get("content").and_then(|c| c.as_str()).unwrap_or("");
        obj.insert("content".into(), Value::String(format!("{current_content}{content}")));
        if !reasoning_content.is_empty() {
            let current_rc = obj.get("reasoning_content").and_then(|c| c.as_str()).unwrap_or("");
            obj.insert("reasoning_content".into(), Value::String(format!("{current_rc}{reasoning_content}")));
        }
        if let Some(sig) = item.get("thought_signature").and_then(|s| s.as_str()) {
            obj.insert("thought_signature".into(), Value::String(sig.into()));
        }
    } else {
        messages.push(json!({"role": role, "content": content}));
    }
}

fn process_reasoning(item: &Value, messages: &mut Vec<Value>) {
    let content_list = item.get("content").and_then(|c| c.as_array());
    let content = content_list.map(|arr| arr.iter().map(|cp| match cp {
        Value::String(s) => s.clone(),
        Value::Object(obj) => obj.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(),
        _ => String::new(),
    }).collect::<Vec<_>>().join("")).unwrap_or_default();

    let idx = ensure_last_assistant(messages);
    let amsg = &mut messages[idx];
    let obj = amsg.as_object_mut().unwrap();
    let current = obj.get("reasoning_content").and_then(|c| c.as_str()).unwrap_or("");
    obj.insert("reasoning_content".into(), Value::String(format!("{current}{content}")));
    if let Some(sig) = item.get("thought_signature").and_then(|s| s.as_str()) {
        obj.insert("thought_signature".into(), Value::String(sig.into()));
    }
}

fn process_tool_call(item: &Value, messages: &mut Vec<Value>) {
    let call_id = item.get("call_id").or_else(|| item.get("id"))
        .and_then(|v| v.as_str()).unwrap_or("call_unknown");
    let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("function_call");
    let name = item.get("name").and_then(|n| n.as_str()).map(|n| match item_type {
        "commandExecution" => "run_shell_command",
        "local_shell_call" => "local_shell_command",
        "fileChange" => "write_file",
        "web_search_call" => "web_search",
        _ => n,
    }).or_else(|| item.get("name").and_then(|n| n.as_str())).unwrap_or("unknown");

    let mut args = item.get("arguments").or_else(|| item.get("input")).cloned().unwrap_or(Value::Null);
    if args.is_null() && item_type == "web_search_call" { args = item.get("action").cloned().unwrap_or(json!({})); }
    if args.is_null() {
        match item_type {
            "commandExecution" => { args = json!({"command": item.get("command").and_then(|c| c.as_str()).unwrap_or(""), "dir_path": item.get("cwd").and_then(|c| c.as_str()).unwrap_or(".")}); }
            "local_shell_call" => { let action = item.get("action").and_then(|a| a.get("exec")); args = json!({"command": action.and_then(|e| e.get("command")).cloned().unwrap_or(json!([])), "working_directory": action.and_then(|e| e.get("working_directory"))}); }
            "fileChange" => { let changes = item.get("changes").and_then(|c| c.as_array()); let path = changes.and_then(|arr| arr.first()).and_then(|c| c.get("path")).and_then(|p| p.as_str()).unwrap_or("unknown"); args = json!({"file_path": path}); }
            _ => {}
        }
    }
    let args_str = if args.is_string() { args.as_str().unwrap().to_string() } else { serde_json::to_string(&args).unwrap_or_default() };

    let idx = ensure_last_assistant(messages);
    let amsg = &mut messages[idx];
    let obj = amsg.as_object_mut().unwrap();
    let tool_calls = obj.entry("tool_calls").or_insert_with(|| json!([]));
    if let Some(arr) = tool_calls.as_array_mut() {
        arr.push(json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": args_str}}));
    }
    if let Some(sig) = item.get("thought_signature") { obj.insert("thought_signature".into(), Value::String(sig.as_str().unwrap_or("").into())); }
    if let Some(thought) = item.get("thought").and_then(|t| t.as_str()) {
        let current = obj.get("reasoning_content").and_then(|r| r.as_str()).unwrap_or("");
        obj.insert("reasoning_content".into(), Value::String(format!("{current}{thought}")));
    }
}

fn process_tool_output(item: &Value, messages: &mut Vec<Value>) {
    let call_id = item.get("call_id").or_else(|| item.get("id")).and_then(|v| v.as_str()).unwrap_or("call_unknown");
    let output_raw = item.get("output").or_else(|| item.get("content")).or_else(|| item.get("stdout")).cloned().unwrap_or(Value::Null);
    let content = match &output_raw {
        Value::String(s) => s.clone(),
        Value::Object(obj) => obj.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string(),
        Value::Array(arr) => arr.iter().filter_map(|part| {
            let ptype = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if ptype == "input_text" || ptype == "text" { part.get("text").and_then(|t| t.as_str()).map(String::from) } else { None }
        }).collect::<Vec<_>>().join(""),
        _ => String::new(),
    };
    let content = if content.is_empty() {
        if let Some(stderr) = item.get("stderr").and_then(|s| s.as_str()) { format!("Error: {stderr}") } else { String::new() }
    } else { content };
    messages.push(json!({"role": "tool", "tool_call_id": call_id, "content": content}));
}

fn extract_content_text(content_raw: Option<&Value>) -> String {
    match content_raw {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr.iter().map(|part| {
            let ptype = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match ptype { "input_text" | "text" | "output_text" => part.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(), _ => String::new() }
        }).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

fn ensure_last_assistant(messages: &mut Vec<Value>) -> usize {
    if let Some(last) = messages.last()
        && last.get("role").and_then(|r| r.as_str()) == Some("assistant") { return messages.len() - 1; }
    messages.push(json!({"role": "assistant", "content": null}));
    messages.len() - 1
}

fn normalize_tools(tools: &[Value]) -> Vec<Value> {
    tools.iter().map(|t| {
        if t.get("type").and_then(|v| v.as_str()) == Some("function") && t.get("function").is_none() {
            json!({"type": "function", "function": {"name": t.get("name"), "description": t.get("description"), "parameters": t.get("parameters"), "strict": t.get("strict").unwrap_or(&Value::Bool(false))}})
        } else { t.clone() }
    }).collect()
}
