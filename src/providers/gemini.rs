use axum::body::Body;
use axum::http::{HeaderMap, header};
use axum::response::Response;
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

use super::gemini_utils::{GeminiContent, GeminiSystemInstruction, map_messages, sanitize_params};
use crate::auth::AuthType;
use crate::config::with_config;
use crate::error::ProxyError;
use crate::providers::base::{Provider, ProviderExecutionContext};
use crate::schema::common::from_chat_request;
use crate::schema::openai::{ChatRequest, ResponsesRequest};
use serde_json::Value;

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
        let auth_ctx = context
            .gemini_auth
            .get_auth_context(&context.account, false)
            .await?;
        let model = context.upstream_model();

        let common = from_chat_request(req_data);
        let (contents, system_instruction) = map_messages(&req_data.messages, &common, model);
        let temperature = req_data.temperature.unwrap_or(1.0);
        let top_p = req_data.top_p.unwrap_or(1.0);
        let max_tokens = req_data.max_tokens;
        let reasoning_cfg = context.reasoning();

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

        let provider_config = with_config(&context.config, |cfg| {
            cfg.gemini_provider_config(context.provider())
        })
        .map_err(ProxyError::Config)?;
        let (url, req_headers, body_field) = match &auth_ctx.auth_type {
            AuthType::Public => {
                let key = auth_ctx.api_key.as_ref().unwrap();
                (
                    format!(
                        "{}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={key}",
                        provider_config.api_public
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
                        provider_config.api_internal
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
                    .timeout(std::time::Duration::from_secs(with_config(
                        &context.config,
                        |cfg| cfg.timeouts.read_seconds,
                    )))
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
                    .timeout(std::time::Duration::from_secs(with_config(
                        &context.config,
                        |cfg| cfg.timeouts.read_seconds,
                    )))
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
        let idle_timeout_seconds = with_config(&context.config, |cfg| cfg.timeouts.read_seconds);

        let byte_stream = resp.bytes_stream();
        let sse_stream = crate::providers::gemini_stream::stream_responses_sse(
            byte_stream,
            &resp_id,
            &model_owned,
            created_ts,
            req_data,
            idle_timeout_seconds,
        );
        let body = Body::from_stream(sse_stream);
        Ok(Response::builder()
            .status(200)
            .header("Content-Type", "text/event-stream; charset=utf-8")
            .header("Connection", "keep-alive")
            .body(body)
            .unwrap())
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

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(GeminiProvider {
            client: self.client.clone(),
        })
    }
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
    parameters: Value,
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
            .unwrap_or_else(|| Value::Object(Default::default()));
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

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
