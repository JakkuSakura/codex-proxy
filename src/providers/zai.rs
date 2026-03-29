use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::providers::base::Provider;

pub struct ZAIProvider { client: reqwest::Client }
impl Default for ZAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ZAIProvider { pub fn new() -> Self { Self { client: reqwest::Client::new() } }

    async fn execute_request(&self, data: &Value, headers: HeaderMap) -> Result<Response<Body>, ProxyError> {
        let mut payload = json!({"model": data.get("model"), "messages": data.get("messages").cloned().unwrap_or(json!([])), "stream": data.get("stream").and_then(|s| s.as_bool()).unwrap_or(false)});
        if let Some(tools) = data.get("tools") { payload["tools"] = tools.clone(); }
        if let Some(tc) = data.get("tool_choice") { payload["tool_choice"] = tc.clone(); }
        if let Some(temp) = data.get("temperature") { payload["temperature"] = temp.clone(); }
        if let Some(top_p) = data.get("top_p") { payload["top_p"] = top_p.clone(); }
        if let Some(max_tokens) = data.get("max_tokens") { payload["max_tokens"] = max_tokens.clone(); }
        transform_payload(&mut payload);
        let auth_header = headers.get("Authorization").and_then(|v| v.to_str().ok()).map(|s| s.to_string());
        let auth = if !CONFIG.z_ai_api_key.is_empty() { Some(format!("Bearer {}", CONFIG.z_ai_api_key)) } else { auth_header };
        let stream = payload.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
        let mut req_builder = self.client.post(&CONFIG.z_ai_url).json(&payload)
            .timeout(std::time::Duration::from_secs(CONFIG.request_timeout_read));
        if let Some(auth) = &auth { req_builder = req_builder.header("Authorization", auth); }
        let resp = req_builder.send().await?;
        info!("Z.AI response status: {}", resp.status());
        if stream { self.handle_stream_response(resp, &payload).await } else { self.handle_sync_response(resp, data).await }
    }

    async fn handle_stream_response(&self, resp: reqwest::Response, payload: &Value) -> Result<Response<Body>, ProxyError> {
        let model = payload.get("model").and_then(|m| m.as_str()).unwrap_or("unknown").to_string();
        let created_ts = now_seconds();
        let metadata = json!({"temperature": payload.get("temperature").and_then(|t| t.as_f64()).unwrap_or(1.0), "top_p": payload.get("top_p").and_then(|t| t.as_f64()).unwrap_or(1.0), "tool_choice": payload.get("tool_choice").and_then(|v| v.as_str()).unwrap_or("auto"), "tools": payload.get("tools").cloned().unwrap_or(json!([])), "store": true, "metadata": json!({})});
        let sse_stream = crate::providers::zai_stream::stream_responses_sse(resp.bytes_stream(), &model, created_ts, &metadata);
        let body = Body::from_stream(sse_stream);
        Ok(Response::builder().status(200).header("Content-Type", "text/event-stream; charset=utf-8").header("Connection", "keep-alive").body(body).unwrap())
    }

    async fn handle_sync_response(&self, resp: reqwest::Response, original_data: &Value) -> Result<Response<Body>, ProxyError> {
        let status = resp.status();
        let body_bytes = resp.bytes().await?;
        if original_data.get("_is_responses_api").and_then(|v| v.as_bool()).unwrap_or(false) && status.is_success() {
            return Ok(write_mapped_response(&body_bytes).into_response());
        }
        Ok(Response::builder().status(status).header("Content-Type", "application/json").body(Body::from(body_bytes)).unwrap())
    }

    async fn handle_compact_impl(&self, data: &Value, headers: HeaderMap) -> Result<Response<Body>, ProxyError> {
        let compaction_model = CONFIG.compaction_model.as_deref().unwrap_or_else(|| CONFIG.models.first().map(|s| s.as_str()).unwrap_or("glm-4.6"));
        let input = data.get("input").cloned().unwrap_or(json!([]));
        let mut messages = match &input { Value::String(s) => vec![json!({"role": "user", "content": s})], Value::Array(arr) => arr.clone(), _ => vec![] };
        let compaction_prompt = data.get("instructions").and_then(|i| i.as_str()).unwrap_or("Summarize the conversation history concisely.");
        messages.push(json!({"role": "user", "content": format!("Perform context compaction. instructions: {compaction_prompt}")}));
        let payload = json!({"model": compaction_model, "messages": messages, "stream": false, "temperature": CONFIG.compaction_temperature, "max_tokens": 4096});
        let auth_header = headers.get("Authorization").and_then(|v| v.to_str().ok()).map(|s| s.to_string());
        let auth = if !CONFIG.z_ai_api_key.is_empty() { Some(format!("Bearer {}", CONFIG.z_ai_api_key)) } else { auth_header };
        let mut req_builder = self.client.post(&CONFIG.z_ai_url).json(&payload).timeout(std::time::Duration::from_secs(CONFIG.request_timeout_read));
        if let Some(a) = &auth { req_builder = req_builder.header("Authorization", a); }
        let resp = req_builder.send().await?;
        if !resp.status().is_success() { let status = resp.status(); let body = resp.text().await.unwrap_or_default(); return Err(ProxyError::Provider(format!("Compaction error ({}): {}", status, body))); }
        let z_data: Value = resp.json().await?;
        let final_text = z_data.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first()).and_then(|c| c.get("message")).and_then(|m| m.get("content")).and_then(|c| c.as_str()).unwrap_or("");
        Ok(axum::Json(json!({"summary_text": final_text})).into_response())
    }
}

fn transform_payload(payload: &mut Value) {
    if let Some(messages) = payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for m in messages { if m.get("role").and_then(|r| r.as_str()) == Some("developer") { m["role"] = json!("system"); } }
    }
    if let Some(tools) = payload.get_mut("tools").and_then(|t| t.as_array()) {
        let mut transformed = Vec::new();
        for tool in tools {
            let ttype = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match ttype {
                "function" => { let mut t = tool.clone(); if let Some(obj) = t.as_object_mut() { obj.remove("strict"); } transformed.push(t); }
                "web_search" => { transformed.push(json!({"type": "web_search", "web_search": {"enable": true, "search_engine": "search_pro_jina"}})); }
                _ => {}
            }
        }
        payload["tools"] = json!(transformed);
    }
}

fn write_mapped_response(body_bytes: &[u8]) -> Response<Body> {
    let z_data: Value = serde_json::from_slice(body_bytes).unwrap_or(json!({}));
    let choice = z_data.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first());
    let message = choice.and_then(|c| c.get("message"));
    let usage = z_data.get("usage").cloned().unwrap_or(json!({}));
    let mut output_items: Vec<Value> = Vec::new();
    if let Some(msg) = message {
        if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                let name = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).unwrap_or("");
                let args = tc.get("function").and_then(|f| f.get("arguments")).cloned().unwrap_or(json!({}));
                let call_id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
                let itype = match name { "shell" | "container.exec" | "shell_command" => "local_shell_call", _ => "function_call" };
                let mut item = json!({"id": call_id, "type": itype, "status": "completed", "name": name, "arguments": serde_json::to_string(&args).unwrap_or_default(), "call_id": call_id});
                if itype == "local_shell_call" { let cmd = args.get("command").cloned().unwrap_or(json!([])); item.as_object_mut().unwrap().insert("action".into(), json!({"type": "exec", "command": cmd})); }
                output_items.push(item);
            }
        }
        if let Some(content) = msg.get("content").and_then(|c| c.as_str())
            && !content.is_empty() { output_items.push(json!({"id": format!("msg_{}", now_ms()), "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "text", "text": content}]})); }
    }
    let resp_obj = json!({"id": format!("zai_{}", z_data.get("id").and_then(|i| i.as_str()).unwrap_or("unknown")), "object": "response", "created": z_data.get("created").and_then(|c| c.as_i64()).unwrap_or(0), "model": z_data.get("model").and_then(|m| m.as_str()).unwrap_or("unknown"), "status": "completed",
        "usage": {"prompt_tokens": usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0), "completion_tokens": usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0), "total_tokens": usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0)}, "output": output_items});
    axum::Json(resp_obj).into_response()
}

impl Provider for ZAIProvider {
    fn handle_request(&self, data: Value, headers: HeaderMap) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>> { Box::pin(async move { self.execute_request(&data, headers).await }) }
    fn handle_compact(&self, data: Value, headers: HeaderMap) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>> { Box::pin(async move { self.handle_compact_impl(&data, headers).await }) }
    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> { Box::new(ZAIProvider { client: self.client.clone() }) }
}

fn now_ms() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0) }
fn now_seconds() -> i64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0) }
