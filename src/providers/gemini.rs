use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::auth::AuthType;
use crate::config::with_config;
use crate::error::{ProviderError, ProxyError};
use crate::providers::base::{Provider, ProviderExecutionContext};
use crate::schema::openai::{ChatRequest, ResponsesRequest};
use fp_agent::providers::gemini::{GeminiInternalBody, build_gemini_request};

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
        let thinking_budget = context.reasoning().map(|rc| rc.budget);
        let request_body = build_gemini_request(req_data, thinking_budget);

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
            return Err(ProxyError::Provider(ProviderError::new(
                Some(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY)),
                format!("Gemini API error ({}): {}", status, body),
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

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
