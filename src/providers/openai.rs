use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;

use crate::account_pool::AccountAuth;
use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::providers::base::{Provider, ProviderExecutionContext};
use crate::schema::openai::{ChatRequest, CompactRequest, ResponsesRequest};

pub struct OpenAiProvider {
    client: reqwest::Client,
}

impl Default for OpenAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn forward_json<T: serde::Serialize>(
        &self,
        payload: &T,
        mut headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let api_key = match &context.account.auth {
            AccountAuth::ApiKey { api_key } => api_key,
            _ => {
                return Err(ProxyError::Auth(
                    "OpenAI provider requires account auth.type=api_key".into(),
                ));
            }
        };

        headers.remove(axum::http::header::HOST);
        headers.remove(axum::http::header::CONTENT_LENGTH);
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {api_key}").parse().map_err(|e| {
                ProxyError::Internal(format!("Invalid OpenAI authorization header: {e}"))
            })?,
        );

        let response = self
            .client
            .post(&CONFIG.providers.openai.responses_url)
            .headers(headers)
            .json(payload)
            .timeout(std::time::Duration::from_secs(CONFIG.timeouts.read_seconds))
            .send()
            .await?;

        let status = response.status();
        let response_headers = response.headers().clone();
        let bytes = response.bytes().await?;

        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(ProxyError::Auth(format!(
                "OpenAI request unauthorized ({}): {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }
        if !status.is_success() {
            return Err(ProxyError::Provider(format!(
                "OpenAI request failed ({}): {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }

        let mut builder = Response::builder().status(status);
        for (name, value) in &response_headers {
            if name.as_str().eq_ignore_ascii_case("content-length") {
                continue;
            }
            builder = builder.header(name, value);
        }
        builder
            .body(Body::from(bytes))
            .map_err(|e| ProxyError::Internal(format!("Failed to build OpenAI response: {e}")))
    }
}

impl Provider for OpenAiProvider {
    fn handle_request(
        &self,
        raw_request: ResponsesRequest,
        _normalized_request: ChatRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let mut request = raw_request;
        request.model = context.route.upstream_model.clone();
        Box::pin(async move { self.forward_json(&request, headers, &context).await })
    }

    fn handle_compact(
        &self,
        data: CompactRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let request = ResponsesRequest {
            model: context.route.upstream_model.clone(),
            input: Some(data.input),
            instructions: Some(data.instructions),
            previous_response_id: None,
            store: Some(false),
            metadata: None,
            tools: None,
            tool_choice: None,
            temperature: Some(CONFIG.compaction.temperature),
            top_p: None,
            max_tokens: Some(4096),
            stream: Some(false),
            include: None,
        };
        Box::pin(async move { self.forward_json(&request, headers, &context).await })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(OpenAiProvider {
            client: self.client.clone(),
        })
    }
}
