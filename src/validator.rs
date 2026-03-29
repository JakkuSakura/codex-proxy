use serde_json::Value;
use tracing::debug;

use crate::error::ProxyError;

pub fn validate_request(data: &Value, path: &str) -> Result<(), ProxyError> {
    if let Some(model) = data.get("model") {
        validate_model(model)?;
    }

    if let Some(messages) = data.get("messages") {
        let msgs = messages.as_array()
            .ok_or_else(|| ProxyError::Validation("messages must be a list".into()))?;
        validate_messages(msgs)?;
    }

    if let Some(tools) = data.get("tools") {
        let tools_list = tools.as_array()
            .ok_or_else(|| ProxyError::Validation("tools must be a list".into()))?;
        validate_tools(tools_list)?;
    }

    if let Some(temp) = data.get("temperature") {
        let t = temp.as_f64()
            .ok_or_else(|| ProxyError::Validation("temperature must be a number".into()))?;
        if !(0.0..=2.0).contains(&t) {
            return Err(ProxyError::Validation(format!("temperature must be between 0 and 2, got: {t}")));
        }
    }

    if let Some(max_tokens) = data.get("max_tokens") {
        let mt = max_tokens.as_u64()
            .ok_or_else(|| ProxyError::Validation("max_tokens must be a positive integer".into()))?;
        if !(1..=128_000).contains(&mt) {
            return Err(ProxyError::Validation(format!("max_tokens must be between 1 and 128000, got: {mt}")));
        }
    }

    if let Some(stream) = data.get("stream")
        && !stream.is_boolean() {
            return Err(ProxyError::Validation(format!("stream must be a boolean, got: {stream}")));
        }

    if path.contains("/compact") {
        validate_compact_request(data)?;
    }

    debug!("Request validation passed");
    Ok(())
}

fn validate_model(model: &Value) -> Result<(), ProxyError> {
    let s = model.as_str()
        .ok_or_else(|| ProxyError::Validation("model must be a string".into()))?;
    if s.len() > 100 {
        return Err(ProxyError::Validation(format!("Invalid model name (too long): {s}")));
    }
    Ok(())
}

fn validate_messages(messages: &[Value]) -> Result<(), ProxyError> {
    for (i, msg) in messages.iter().enumerate() {
        if !msg.is_object() {
            return Err(ProxyError::Validation(format!("Message {i} must be an object")));
        }
        let role = msg.get("role")
            .ok_or_else(|| ProxyError::Validation(format!("Message {i} missing required field 'role'")))?
            .as_str()
            .ok_or_else(|| ProxyError::Validation(format!("Message {i} role must be a string")))?;

        match role {
            "system" | "user" | "assistant" | "developer" => {}
            other => return Err(ProxyError::Validation(format!("Message {i} has invalid role: {other}"))),
        }

        if role == "user" && msg.get("content").is_none() && msg.get("text").is_none() {
            return Err(ProxyError::Validation(format!("User message {i} must have 'content' or 'text'")));
        }
    }
    Ok(())
}

fn validate_tools(tools: &[Value]) -> Result<(), ProxyError> {
    for (i, tool) in tools.iter().enumerate() {
        if !tool.is_object() {
            return Err(ProxyError::Validation(format!("Tool {i} must be an object")));
        }
        let ttype = tool.get("type")
            .ok_or_else(|| ProxyError::Validation(format!("Tool {i} missing required field 'type'")))?
            .as_str()
            .ok_or_else(|| ProxyError::Validation(format!("Tool {i} type must be a string")))?;

        match ttype {
            "function" | "web_search" | "retrieval" => {}
            other => return Err(ProxyError::Validation(format!("Tool {i} has invalid type: {other}"))),
        }
    }
    Ok(())
}

fn validate_compact_request(data: &Value) -> Result<(), ProxyError> {
    if data.get("input").is_none() {
        return Err(ProxyError::Validation("Compaction requests must have 'input' field".into()));
    }
    if data.get("instructions").is_none() {
        return Err(ProxyError::Validation("Compaction requests must have 'instructions' field".into()));
    }

    let input = &data["input"];
    match input {
        Value::String(_) => {}
        Value::Array(arr) if arr.len() <= 100 => {}
        Value::Array(_) => return Err(ProxyError::Validation("Compaction input exceeds maximum length of 100 messages".into())),
        _ => return Err(ProxyError::Validation("Compaction input must be string or list".into())),
    }
    Ok(())
}
