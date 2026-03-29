use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::providers::base::Provider;
use crate::schema::json_value::JsonValue;
use crate::schema::openai::{
    ChatContent, ChatMessage, ChatRequest, CompactRequest, Tool, ToolCall,
};
use crate::schema::sse::{
    FunctionCallItem, LocalShellCallItem, MessageItem, OutputContentPart, OutputItem,
    ResponseObject, Usage,
};

pub struct ZAIProvider {
    client: reqwest::Client,
}

impl Default for ZAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ZAIProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn execute_request(
        &self,
        req: &ChatRequest,
        headers: HeaderMap,
    ) -> Result<Response<Body>, ProxyError> {
        let auth_header = headers
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let auth = if !CONFIG.z_ai_api_key.is_empty() {
            Some(format!("Bearer {}", CONFIG.z_ai_api_key))
        } else {
            auth_header
        };
        if auth.is_none() {
            return Err(ProxyError::Auth(
                "Missing Z.AI API key. Set CODEX_PROXY_ZAI_API_KEY (recommended) or pass an Authorization header to the proxy."
                    .into(),
            ));
        }

        let zai_req = to_zai_chat_request(req);
        let mut req_builder = self
            .client
            .post(&CONFIG.z_ai_url)
            .json(&zai_req)
            .timeout(std::time::Duration::from_secs(CONFIG.request_timeout_read));
        if let Some(a) = &auth {
            req_builder = req_builder.header("Authorization", a);
        }

        let resp = req_builder.send().await?;
        let status = resp.status();
        info!("Z.AI response status: {}", status);

        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProxyError::Auth(format!(
                    "Z.AI request unauthorized ({}). Check CODEX_PROXY_ZAI_API_KEY. Body: {}",
                    status, body
                )));
            }
            return Err(ProxyError::Provider(format!(
                "Z.AI request failed ({}): {}",
                status, body
            )));
        }

        if req.stream {
            self.handle_stream_response(resp, req).await
        } else {
            self.handle_sync_response(resp).await
        }
    }

    async fn handle_stream_response(
        &self,
        resp: reqwest::Response,
        req: &ChatRequest,
    ) -> Result<Response<Body>, ProxyError> {
        let model = req.model.clone();
        let created_ts = now_seconds();
        let sse_stream = crate::providers::zai_stream::stream_responses_sse(
            resp.bytes_stream(),
            &model,
            created_ts,
            req,
        );
        let body = Body::from_stream(sse_stream);
        Ok(Response::builder()
            .status(200)
            .header("Content-Type", "text/event-stream; charset=utf-8")
            .header("Connection", "keep-alive")
            .body(body)
            .unwrap())
    }

    async fn handle_sync_response(
        &self,
        resp: reqwest::Response,
    ) -> Result<Response<Body>, ProxyError> {
        let body_bytes = resp.bytes().await?;
        let z_data: ZaiChatResponse = serde_json::from_slice(&body_bytes).map_err(|e| {
            ProxyError::Provider(format!("Failed to decode Z.AI response JSON: {e}"))
        })?;

        let out = map_zai_response_to_responses_api(&z_data);
        Ok(axum::Json(out).into_response())
    }

    async fn handle_compact_impl(
        &self,
        data: &CompactRequest,
        headers: HeaderMap,
    ) -> Result<Response<Body>, ProxyError> {
        let compaction_model = CONFIG.compaction_model.as_deref().unwrap_or_else(|| {
            CONFIG
                .models
                .first()
                .map(|s| s.as_str())
                .unwrap_or("glm-4.6")
        });

        let mut messages = match &data.input {
            crate::schema::openai::ResponsesInput::Text(s) => vec![ZaiMessage {
                role: "user".into(),
                content: Some(s.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            crate::schema::openai::ResponsesInput::Items(items) => {
                let mut req = ChatRequest {
                    model: compaction_model.to_string(),
                    messages: Vec::new(),
                    tools: Vec::new(),
                    tool_choice: None,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    stream: false,
                    store: false,
                    metadata: Default::default(),
                    previous_response_id: None,
                    include: Vec::new(),
                };
                // Reuse normalizer semantics for compaction input items.
                req.messages =
                    crate::normalizer::normalize(crate::schema::openai::ResponsesRequest {
                        model: compaction_model.to_string(),
                        input: Some(crate::schema::openai::ResponsesInput::Items(items.clone())),
                        instructions: None,
                        previous_response_id: None,
                        store: None,
                        metadata: None,
                        tools: None,
                        tool_choice: None,
                        temperature: None,
                        top_p: None,
                        max_tokens: None,
                        stream: None,
                        include: None,
                    })
                    .messages;
                req.messages
                    .into_iter()
                    .map(|m| to_zai_message(&m))
                    .collect::<Vec<_>>()
            }
        };

        let compaction_prompt = instructions_to_text(&data.instructions);
        let compaction_prompt = if compaction_prompt.is_empty() {
            "Summarize the conversation history concisely.".to_string()
        } else {
            compaction_prompt
        };
        messages.push(ZaiMessage {
            role: "user".into(),
            content: Some(format!(
                "Perform context compaction. instructions: {compaction_prompt}"
            )),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });

        let payload = ZaiChatRequest {
            model: compaction_model.to_string(),
            messages,
            stream: false,
            tools: None,
            tool_choice: None,
            temperature: Some(CONFIG.compaction_temperature),
            top_p: None,
            max_tokens: Some(4096),
        };

        let auth_header = headers
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let auth = if !CONFIG.z_ai_api_key.is_empty() {
            Some(format!("Bearer {}", CONFIG.z_ai_api_key))
        } else {
            auth_header
        };
        if auth.is_none() {
            return Err(ProxyError::Auth(
                "Missing Z.AI API key. Set CODEX_PROXY_ZAI_API_KEY (recommended) or pass an Authorization header to the proxy."
                    .into(),
            ));
        }

        let mut req_builder = self
            .client
            .post(&CONFIG.z_ai_url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(CONFIG.request_timeout_read));
        if let Some(a) = &auth {
            req_builder = req_builder.header("Authorization", a);
        }
        let resp = req_builder.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProxyError::Auth(format!(
                    "Z.AI compaction unauthorized ({}). Check CODEX_PROXY_ZAI_API_KEY. Body: {}",
                    status, body
                )));
            }
            return Err(ProxyError::Provider(format!(
                "Z.AI compaction failed ({}): {}",
                status, body
            )));
        }
        let z_data: ZaiChatResponse = resp.json().await?;
        let final_text = z_data
            .choices
            .first()
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.content.as_deref())
            .unwrap_or("")
            .to_string();
        Ok(axum::Json(CompactResponse {
            summary_text: final_text,
        })
        .into_response())
    }
}

impl Provider for ZAIProvider {
    fn handle_request(
        &self,
        data: ChatRequest,
        headers: HeaderMap,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move { self.execute_request(&data, headers).await })
    }

    fn handle_compact(
        &self,
        data: CompactRequest,
        headers: HeaderMap,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move { self.handle_compact_impl(&data, headers).await })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(ZAIProvider {
            client: self.client.clone(),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct CompactResponse {
    summary_text: String,
}

#[derive(Clone, Debug, Serialize)]
struct ZaiChatRequest {
    model: String,
    messages: Vec<ZaiMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct ZaiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

fn to_zai_chat_request(req: &ChatRequest) -> ZaiChatRequest {
    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(req.tools.iter().cloned().map(transform_tool).collect())
    };
    ZaiChatRequest {
        model: req.model.clone(),
        messages: req.messages.iter().map(to_zai_message).collect(),
        stream: req.stream,
        tools,
        tool_choice: req.tool_choice.clone(),
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
    }
}

fn to_zai_message(msg: &ChatMessage) -> ZaiMessage {
    let role = if msg.role == "developer" {
        "system".to_string()
    } else {
        msg.role.clone()
    };
    let content = msg.content.as_ref().map(chat_content_to_string);
    let tool_calls = if msg.tool_calls.is_empty() {
        None
    } else {
        Some(msg.tool_calls.clone())
    };
    ZaiMessage {
        role,
        content,
        tool_calls,
        tool_call_id: msg.tool_call_id.clone(),
        name: msg.name.clone(),
    }
}

fn chat_content_to_string(c: &ChatContent) -> String {
    match c {
        ChatContent::Text(s) => s.clone(),
        ChatContent::Parts(parts) => parts
            .iter()
            .map(|p| p.text.clone().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn instructions_to_text(i: &crate::schema::openai::Instructions) -> String {
    match i {
        crate::schema::openai::Instructions::Text(s) => s.clone(),
        crate::schema::openai::Instructions::Parts(parts) => parts
            .iter()
            .map(|p| match p {
                crate::schema::openai::TextPart::Text(s) => s.clone(),
                crate::schema::openai::TextPart::Obj { text } => text.clone(),
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn transform_tool(mut tool: Tool) -> Tool {
    if tool.tool_type == "function" {
        tool.strict = None;
    }
    if tool.tool_type == "web_search" {
        tool.web_search = Some(crate::schema::openai::WebSearchConfig {
            enable: Some(true),
            search_engine: Some("search_pro_jina".into()),
        });
    }
    tool
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiChatResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    created: Option<i64>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ZaiChoice>,
    #[serde(default)]
    usage: Option<ZaiUsage>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiChoice {
    #[serde(default)]
    message: Option<ZaiChoiceMessage>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiChoiceMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ZaiToolCall>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ZaiToolCallFn>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiToolCallFn {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<JsonValue>,
}

#[derive(Clone, Debug, Deserialize)]
struct ZaiUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

fn map_zai_response_to_responses_api(z: &ZaiChatResponse) -> ResponseObject {
    let created = z.created.unwrap_or(0);
    let model = z.model.clone().unwrap_or_else(|| "unknown".into());
    let resp_id = format!("zai_{}", z.id.clone().unwrap_or_else(|| "unknown".into()));

    let mut output: Vec<OutputItem> = Vec::new();

    if let Some(choice) = z.choices.first() {
        if let Some(msg) = &choice.message {
            for (idx, tc) in msg.tool_calls.iter().enumerate() {
                let call_id = tc
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("call_{}_{}", now_ms(), idx));
                let name = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_default();
                let args = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.as_ref())
                    .map(|a| serde_json::to_string(a).unwrap_or_default())
                    .unwrap_or_else(|| "{}".into());

                let item = if name == "shell" || name == "container.exec" || name == "shell_command"
                {
                    let cmd = extract_command_from_args(&args);
                    OutputItem::LocalShellCall(LocalShellCallItem {
                        id: call_id.clone(),
                        status: "completed".into(),
                        name,
                        arguments: args,
                        call_id: call_id.clone(),
                        action: crate::schema::sse::ShellAction {
                            action_type: "exec",
                            command: cmd,
                        },
                        thought_signature: None,
                    })
                } else {
                    OutputItem::FunctionCall(FunctionCallItem {
                        id: call_id.clone(),
                        status: "completed".into(),
                        name,
                        arguments: args,
                        call_id: call_id.clone(),
                        thought_signature: None,
                    })
                };
                output.push(item);
            }

            if let Some(content) = msg.content.as_deref() {
                if !content.is_empty() {
                    output.push(OutputItem::Message(MessageItem {
                        id: format!("msg_{}", now_ms()),
                        role: "assistant",
                        status: "completed".into(),
                        content: vec![OutputContentPart::OutputText {
                            text: content.to_string(),
                        }],
                    }));
                }
            }
        }
    }

    let usage = z.usage.as_ref().map(|u| {
        let prompt = u.prompt_tokens.unwrap_or(0);
        let completion = u.completion_tokens.unwrap_or(0);
        let total = u.total_tokens.unwrap_or(prompt + completion);
        Usage {
            input_tokens: prompt,
            output_tokens: completion,
            total_tokens: total,
            input_tokens_details: None,
            output_tokens_details: None,
        }
    });

    ResponseObject {
        id: resp_id,
        object: "response",
        created_at: created,
        completed_at: None,
        model,
        status: "completed".into(),
        temperature: 1.0,
        top_p: 1.0,
        tool_choice: "auto".into(),
        tools: Vec::new(),
        parallel_tool_calls: true,
        store: false,
        metadata: Default::default(),
        output,
        usage,
    }
}

fn extract_command_from_args(args: &str) -> Vec<String> {
    let parsed: Result<JsonValue, _> = serde_json::from_str(args);
    match parsed {
        Ok(JsonValue::Object(map)) => match map.get("command") {
            Some(JsonValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
