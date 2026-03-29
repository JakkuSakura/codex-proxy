use axum::body::Body;
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use super::gemini_utils::{
    GeminiContent, GeminiPart, GeminiSystemInstruction, map_messages, sanitize_params,
};
use crate::auth::{AuthType, GEMINI_AUTH_MANAGER};
use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::providers::base::{Provider, ProviderExecutionContext};
use crate::schema::json_value::JsonValue;
use crate::schema::openai::{ChatRequest, CompactRequest, ResponsesRequest};

pub struct GeminiProvider {
    client: reqwest::Client,
}

impl Default for GeminiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn execute_stream(
        &self,
        req_data: &ChatRequest,
        context: &ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let auth_ctx = GEMINI_AUTH_MANAGER
            .get_auth_context(&context.account, false)
            .await?;
        let model = context.upstream_model();

        let (contents, system_instruction) = map_messages(&req_data.messages, model);
        let temperature = req_data.temperature.unwrap_or(1.0);
        let top_p = req_data.top_p.unwrap_or(1.0);
        let max_tokens = req_data.max_tokens;
        let reasoning_effort = &CONFIG.reasoning.default_effort;
        let reasoning_cfg = CONFIG.reasoning.effort_levels.get(reasoning_effort);

        let mut gen_config = GeminiGenerationConfig {
            temperature,
            top_p,
            max_output_tokens: None,
            thinking_config: None,
        };
        if let Some(max_t) = max_tokens {
            gen_config.max_output_tokens = Some(max_t);
        }
        if let Some(rc) = reasoning_cfg
            && rc.budget > 0
        {
            gen_config.thinking_config = Some(GeminiThinkingConfig {
                thinking_budget: rc.budget,
                include_thoughts: true,
            });
        }

        let (tools, tool_config) = apply_tools(req_data);
        let request_body = GeminiRequestBody {
            contents,
            generation_config: gen_config,
            system_instruction,
            tools,
            tool_config,
        };

        let (url, req_headers, body_field) = match &auth_ctx.auth_type {
            AuthType::Public => {
                let key = auth_ctx.api_key.as_ref().unwrap();
                (
                    format!(
                        "{}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={key}",
                        CONFIG.providers.gemini.api_public
                    ),
                    header::HeaderMap::new(),
                    GeminiBodyKind::Public,
                )
            }
            AuthType::Internal => {
                let token = auth_ctx.access_token.as_ref().unwrap();
                let mut h = header::HeaderMap::new();
                h.insert("Authorization", format!("Bearer {token}").parse().unwrap());
                h.insert(
                    "User-Agent",
                    "GeminiCLI/0.26.0 (linux; x64)".parse().unwrap(),
                );
                (
                    format!(
                        "{}/v1internal:streamGenerateContent?alt=sse",
                        CONFIG.providers.gemini.api_internal
                    ),
                    h,
                    GeminiBodyKind::Internal,
                )
            }
        };

        let resp = match body_field {
            GeminiBodyKind::Public => {
                self.client
                    .post(&url)
                    .headers(req_headers)
                    .json(&request_body)
                    .timeout(std::time::Duration::from_secs(CONFIG.timeouts.read_seconds))
                    .send()
                    .await?
            }
            GeminiBodyKind::Internal => {
                let token_project = auth_ctx.project_id.as_ref().unwrap();
                let internal = GeminiInternalBody {
                    model: model.to_string(),
                    project: token_project.clone(),
                    request: request_body,
                };
                self.client
                    .post(&url)
                    .headers(req_headers)
                    .json(&internal)
                    .timeout(std::time::Duration::from_secs(CONFIG.timeouts.read_seconds))
                    .send()
                    .await?
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProxyError::Auth(format!(
                    "Gemini API unauthorized ({}): {}",
                    status, body
                )));
            }
            return Err(ProxyError::Provider(format!(
                "Gemini API error ({}): {}",
                status, body
            )));
        }

        let created_ts = now_seconds();
        let resp_id = format!("resp_{created_ts}");
        let model_owned = model.to_string();

        let byte_stream = resp.bytes_stream();
        let sse_stream = crate::providers::gemini_stream::stream_responses_sse(
            byte_stream,
            &resp_id,
            &model_owned,
            created_ts,
            req_data,
        );
        let body = Body::from_stream(sse_stream);
        Ok(Response::builder()
            .status(200)
            .header("Content-Type", "text/event-stream; charset=utf-8")
            .header("Connection", "keep-alive")
            .body(body)
            .unwrap())
    }

    async fn handle_compact_impl(
        &self,
        data: &CompactRequest,
        context: &ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let auth_ctx = GEMINI_AUTH_MANAGER
            .get_auth_context(&context.account, false)
            .await?;
        let compaction_model = context.upstream_model();

        let chat_req = crate::normalizer::normalize(ResponsesRequest {
            model: compaction_model.to_string(),
            input: Some(data.input.clone()),
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
        });

        let (mut contents, system_instruction) = map_messages(&chat_req.messages, compaction_model);
        let compaction_prompt = instructions_to_text(&data.instructions);
        let compaction_prompt = if compaction_prompt.is_empty() {
            "Summarize the conversation history concisely.".to_string()
        } else {
            compaction_prompt
        };
        contents.push(GeminiContent {
            role: "user".into(),
            parts: vec![GeminiPart {
                text: Some(format!(
                    "Perform context compaction. instructions: {compaction_prompt}"
                )),
                thought: None,
                function_call: None,
                function_response: None,
                thought_signature: None,
            }],
        });

        let body = GeminiRequestBody {
            contents,
            generation_config: GeminiGenerationConfig {
                temperature: CONFIG.compaction.temperature,
                top_p: 1.0,
                max_output_tokens: Some(4096),
                thinking_config: None,
            },
            system_instruction,
            tools: None,
            tool_config: None,
        };

        let (url, req_headers, send_body) = match &auth_ctx.auth_type {
            AuthType::Public => {
                let key = auth_ctx.api_key.as_ref().unwrap();
                (
                    format!(
                        "{}/v1beta/models/{compaction_model}:streamGenerateContent?alt=sse&key={key}",
                        CONFIG.providers.gemini.api_public
                    ),
                    header::HeaderMap::new(),
                    GeminiSendBody::Public(body),
                )
            }
            AuthType::Internal => {
                let token = auth_ctx.access_token.as_ref().unwrap();
                let pid = auth_ctx.project_id.as_ref().unwrap();
                let mut h = header::HeaderMap::new();
                h.insert("Authorization", format!("Bearer {token}").parse().unwrap());
                (
                    format!(
                        "{}/v1internal:streamGenerateContent?alt=sse",
                        CONFIG.providers.gemini.api_internal
                    ),
                    h,
                    GeminiSendBody::Internal(GeminiInternalBody {
                        model: compaction_model.to_string(),
                        project: pid.clone(),
                        request: body,
                    }),
                )
            }
        };

        let resp = self
            .client
            .post(&url)
            .headers(req_headers)
            .json(&send_body)
            .timeout(std::time::Duration::from_secs(CONFIG.timeouts.read_seconds))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProxyError::Auth(format!(
                    "Gemini compaction unauthorized ({}): {}",
                    status, body
                )));
            }
            return Err(ProxyError::Provider(format!(
                "Compaction error ({}): {}",
                status, body
            )));
        }

        let mut final_text = String::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(ProxyError::Http)?;
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if !line.starts_with("data: ") {
                    continue;
                }
                let data_str = &line[6..];
                let chunk: GeminiChunk = match serde_json::from_str(data_str) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let resp_part = chunk.response();
                let cand = match resp_part.candidates.first() {
                    Some(c) => c,
                    None => continue,
                };
                let parts = match cand.content.as_ref() {
                    Some(c) => c.parts.as_slice(),
                    None => continue,
                };
                for p in parts {
                    if let Some(t) = p.text.as_deref() {
                        final_text.push_str(t);
                    }
                }
            }
        }

        Ok(axum::Json(CompactResponse {
            summary_text: final_text,
        })
        .into_response())
    }
}

impl Provider for GeminiProvider {
    fn handle_request(
        &self,
        _raw_request: ResponsesRequest,
        normalized_request: ChatRequest,
        _headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move { self.execute_stream(&normalized_request, &context).await })
    }

    fn handle_compact(
        &self,
        data: CompactRequest,
        _headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move { self.handle_compact_impl(&data, &context).await })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(GeminiProvider {
            client: self.client.clone(),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct CompactResponse {
    summary_text: String,
}

#[derive(Clone, Debug)]
enum GeminiBodyKind {
    Public,
    Internal,
}

#[derive(Clone, Debug, Serialize)]
struct GeminiThinkingConfig {
    #[serde(rename = "thinkingBudget")]
    thinking_budget: u64,
    #[serde(rename = "includeThoughts")]
    include_thoughts: bool,
}

#[derive(Clone, Debug, Serialize)]
struct GeminiGenerationConfig {
    temperature: f64,
    #[serde(rename = "topP")]
    top_p: f64,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Clone, Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: JsonValue,
}

#[derive(Clone, Debug, Serialize)]
struct GeminiTool {
    #[serde(
        rename = "functionDeclarations",
        skip_serializing_if = "Option::is_none"
    )]
    function_declarations: Option<Vec<GeminiFunctionDeclaration>>,
    #[serde(rename = "googleSearch", skip_serializing_if = "Option::is_none")]
    google_search: Option<GeminiEmptyObject>,
}

#[derive(Clone, Debug, Serialize)]
struct GeminiEmptyObject {}

#[derive(Clone, Debug, Serialize)]
struct GeminiFunctionCallingConfig {
    mode: String,
}

#[derive(Clone, Debug, Serialize)]
struct GeminiToolConfig {
    #[serde(rename = "functionCallingConfig")]
    function_calling_config: GeminiFunctionCallingConfig,
}

#[derive(Clone, Debug, Serialize)]
struct GeminiRequestBody {
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolConfig>,
}

#[derive(Clone, Debug, Serialize)]
struct GeminiInternalBody {
    model: String,
    project: String,
    request: GeminiRequestBody,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
enum GeminiSendBody {
    Public(GeminiRequestBody),
    Internal(GeminiInternalBody),
}

fn apply_tools(req_data: &ChatRequest) -> (Option<Vec<GeminiTool>>, Option<GeminiToolConfig>) {
    let mut tool_decls: Vec<GeminiFunctionDeclaration> = Vec::new();
    for t in &req_data.tools {
        if t.tool_type != "function" {
            continue;
        }
        let func = match &t.function {
            Some(f) => f,
            None => continue,
        };
        let params = func
            .parameters
            .as_ref()
            .cloned()
            .unwrap_or_else(|| JsonValue::Object(Default::default()));
        tool_decls.push(GeminiFunctionDeclaration {
            name: func.name.clone(),
            description: func.description.clone(),
            parameters: sanitize_params(&params),
        });
    }

    let mut tools: Vec<GeminiTool> = Vec::new();
    if !tool_decls.is_empty() {
        tools.push(GeminiTool {
            function_declarations: Some(tool_decls),
            google_search: None,
        });
    }

    if req_data.include.iter().any(|v| v == "search") {
        tools.push(GeminiTool {
            function_declarations: None,
            google_search: Some(GeminiEmptyObject {}),
        });
    }

    let tools_opt = if tools.is_empty() { None } else { Some(tools) };

    let tc = req_data.tool_choice.as_deref().unwrap_or("auto");
    let mode = match tc {
        "none" => "NONE",
        _ => "ANY",
    };
    let tool_config = if tools_opt.is_some() {
        Some(GeminiToolConfig {
            function_calling_config: GeminiFunctionCallingConfig {
                mode: mode.to_string(),
            },
        })
    } else {
        None
    };

    (tools_opt, tool_config)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum GeminiChunk {
    Wrapped { response: GeminiResponse },
    Direct(GeminiResponse),
}

impl GeminiChunk {
    fn response(&self) -> &GeminiResponse {
        match self {
            GeminiChunk::Wrapped { response } => response,
            GeminiChunk::Direct(r) => r,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiCandidateContent>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiCandidateContent {
    #[serde(default)]
    parts: Vec<GeminiPartResp>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeminiPartResp {
    #[serde(default)]
    text: Option<String>,
}

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
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
