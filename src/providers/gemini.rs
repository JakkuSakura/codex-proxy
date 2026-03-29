use axum::body::Body;
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::auth::{AuthType, GEMINI_AUTH};
use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::providers::base::Provider;
use super::gemini_utils::{map_messages, sanitize_params};

pub struct GeminiProvider {
    client: reqwest::Client,
}

impl Default for GeminiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiProvider {
    pub fn new() -> Self { Self { client: reqwest::Client::new() } }

    async fn execute_stream(&self, model: &str, req_data: &Value, _headers: HeaderMap) -> Result<Response<Body>, ProxyError> {
        let auth_ctx = GEMINI_AUTH.get_auth_context(false).await?;
        let (contents, system_instruction) = map_messages(
            req_data.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default().as_slice(),
            model,
        );
        let temperature = req_data.get("temperature").and_then(|t| t.as_f64()).unwrap_or(1.0);
        let top_p = req_data.get("top_p").and_then(|t| t.as_f64()).unwrap_or(1.0);
        let max_tokens = req_data.get("max_tokens").and_then(|m| m.as_u64());
        let store = req_data.get("store").and_then(|s| s.as_bool()).unwrap_or(true);
        let reasoning_effort = &CONFIG.reasoning_effort;
        let reasoning_cfg = CONFIG.reasoning.effort_levels.get(reasoning_effort);

        let mut gen_config = json!({"temperature": temperature, "topP": top_p});
        if let Some(max_t) = max_tokens { gen_config["maxOutputTokens"] = json!(max_t); }
        if let Some(rc) = reasoning_cfg
            && rc.budget > 0 {
                gen_config["thinkingConfig"] = json!({"thinkingBudget": rc.budget, "includeThoughts": true});
            }

        let mut request_body = json!({"contents": contents, "generationConfig": gen_config});
        if let Some(ref sys_inst) = system_instruction { request_body["systemInstruction"] = sys_inst.clone(); }
        apply_tools(req_data, &mut request_body);

        let (url, req_headers, body_field) = match &auth_ctx.auth_type {
            AuthType::Public => {
                let key = auth_ctx.api_key.as_ref().unwrap();
                (format!("{}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={key}", CONFIG.gemini_api_public),
                 header::HeaderMap::new(), "public")
            }
            AuthType::Internal => {
                let token = auth_ctx.access_token.as_ref().unwrap();
                let mut h = header::HeaderMap::new();
                h.insert("Authorization", format!("Bearer {token}").parse().unwrap());
                h.insert("User-Agent", "GeminiCLI/0.26.0 (linux; x64)".parse().unwrap());
                (format!("{}/v1internal:streamGenerateContent?alt=sse", CONFIG.gemini_api_internal), h, "internal")
            }
        };

        let actual_body = if body_field == "internal" {
            json!({"model": model, "project": auth_ctx.project_id, "request": request_body})
        } else { request_body.clone() };

        let resp = self.client.post(&url).headers(req_headers).json(&actual_body)
            .timeout(std::time::Duration::from_secs(CONFIG.request_timeout_read)).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ProxyError::Provider(format!("Gemini API error ({}): {}", status, body)));
        }

        let created_ts = now_seconds();
        let resp_id = format!("resp_{created_ts}");
        let model_owned = model.to_string();
        let request_metadata = json!({
            "temperature": temperature, "top_p": top_p,
            "tool_choice": req_data.get("tool_choice").unwrap_or(&json!("auto")),
            "tools": req_data.get("tools").unwrap_or(&json!([])),
            "store": store, "metadata": req_data.get("metadata").unwrap_or(&json!({})),
        });

        let byte_stream = resp.bytes_stream();
        let sse_stream = crate::providers::gemini_stream::stream_responses_sse(byte_stream, &resp_id, &model_owned, created_ts, &request_metadata);
        let body = Body::from_stream(sse_stream);
        Ok(Response::builder().status(200)
            .header("Content-Type", "text/event-stream; charset=utf-8")
            .header("Connection", "keep-alive").body(body).unwrap())
    }

    async fn handle_compact_impl(&self, data: &Value, _headers: HeaderMap) -> Result<Response<Body>, ProxyError> {
        let auth_ctx = GEMINI_AUTH.get_auth_context(false).await?;
        let compaction_model = CONFIG.compaction_model.as_deref()
            .unwrap_or_else(|| CONFIG.models.first().map(|s| s.as_str()).unwrap_or("gemini-2.5-flash-lite"));
        let input = data.get("input").cloned().unwrap_or(json!([]));
        let input_list = match &input {
            Value::String(s) => vec![json!({"role": "user", "content": s})],
            Value::Array(arr) => arr.clone(),
            _ => vec![],
        };
        let (mut contents, system_instruction) = map_messages(&input_list, compaction_model);
        let compaction_prompt = data.get("instructions").and_then(|i| i.as_str()).unwrap_or("Summarize the conversation history concisely.");
        contents.push(json!({"role": "user", "parts": [{"text": format!("Perform context compaction. instructions: {compaction_prompt}")}]}));
        let gen_config = json!({"temperature": 0.1, "maxOutputTokens": 4096});

        let (url, req_headers, body) = match &auth_ctx.auth_type {
            AuthType::Public => {
                let key = auth_ctx.api_key.as_ref().unwrap();
                let mut rb = json!({"contents": contents, "generationConfig": gen_config});
                if let Some(ref sys) = system_instruction { rb["systemInstruction"] = sys.clone(); }
                (format!("{}/v1beta/models/{compaction_model}:streamGenerateContent?alt=sse&key={key}", CONFIG.gemini_api_public),
                 header::HeaderMap::new(), rb)
            }
            AuthType::Internal => {
                let token = auth_ctx.access_token.as_ref().unwrap();
                let pid = auth_ctx.project_id.as_ref().unwrap();
                let mut rb = json!({"contents": contents, "generationConfig": gen_config});
                if let Some(ref sys) = system_instruction { rb["systemInstruction"] = sys.clone(); }
                let mut h = header::HeaderMap::new();
                h.insert("Authorization", format!("Bearer {token}").parse().unwrap());
                (format!("{}/v1internal:streamGenerateContent?alt=sse", CONFIG.gemini_api_internal), h,
                 json!({"model": compaction_model, "project": pid, "request": rb}))
            }
        };

        let resp = self.client.post(&url).headers(req_headers).json(&body)
            .timeout(std::time::Duration::from_secs(60)).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ProxyError::Provider(format!("Compaction error ({}): {}", status, body)));
        }

        let mut final_text = String::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(ProxyError::Http)?;
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if !line.starts_with("data: ") { continue; }
                let data_str = &line[6..];
                if let Ok(d) = serde_json::from_str::<Value>(data_str) {
                    let candidates = d.get("candidates").or_else(|| d.get("response").and_then(|r| r.get("candidates")));
                    if let Some(cands) = candidates.and_then(|c| c.as_array())
                        && let Some(cand) = cands.first()
                            && let Some(parts) = cand.get("content").and_then(|c| c.get("parts")).and_then(|p| p.as_array()) {
                                for p in parts {
                                    if let Some(t) = p.get("text").and_then(|t| t.as_str()) { final_text.push_str(t); }
                                }
                            }
                }
            }
        }
        let result = json!({"output": [{"type": "compaction", "encrypted_content": final_text}]});
        Ok(axum::Json(result).into_response())
    }
}


impl Provider for GeminiProvider {
    fn handle_request(&self, data: Value, headers: HeaderMap)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>> {
        Box::pin(async move {
            let model = data.get("model").and_then(|m| m.as_str())
                .unwrap_or_else(|| CONFIG.models.first().map(|s| s.as_str()).unwrap_or("gemini-2.5-flash"));
            self.execute_stream(model, &data, headers).await
        })
    }

    fn handle_compact(&self, data: Value, headers: HeaderMap)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>> {
        Box::pin(async move { self.handle_compact_impl(&data, headers).await })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(GeminiProvider { client: self.client.clone() })
    }
}

fn apply_tools(req_data: &Value, target: &mut Value) {
    let tools = req_data.get("tools").and_then(|t| t.as_array());
    if let Some(tools) = tools {
        if tools.is_empty() { return; }
        let mut tool_decls = Vec::new();
        for t in tools {
            let f = if t.get("type").and_then(|v| v.as_str()) == Some("function") {
                t.get("function")
            } else if t.get("name").is_some() { Some(t) } else { None };
            if let Some(func) = f {
                let params = func.get("parameters").cloned().unwrap_or(json!({}));
                tool_decls.push(json!({"name": func.get("name"), "description": func.get("description").unwrap_or(&Value::Null), "parameters": sanitize_params(&params)}));
            }
        }
        if !tool_decls.is_empty() {
            if target.get("tools").is_none() { target.as_object_mut().unwrap().insert("tools".into(), json!([])); }
            if let Some(arr) = target.get_mut("tools").and_then(|v| v.as_array_mut()) {
                arr.push(json!({"functionDeclarations": tool_decls}));
            }
            let tc = req_data.get("tool_choice").and_then(|t| t.as_str()).unwrap_or("auto");
            let mode = match tc { "none" => "NONE", _ => "ANY" };
            target["toolConfig"] = json!({"functionCallingConfig": {"mode": mode}});
        }
        if let Some(include) = req_data.get("include").and_then(|i| i.as_array())
            && include.iter().any(|v| v.as_str() == Some("search")) {
                if target.get("tools").is_none() { target.as_object_mut().unwrap().insert("tools".into(), json!([])); }
                if let Some(arr) = target.get_mut("tools").and_then(|v| v.as_array_mut()) {
                    arr.push(json!({"googleSearch": {}}));
                }
            }
    }
}

fn now_seconds() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}
