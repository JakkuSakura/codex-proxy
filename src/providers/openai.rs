use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::{Value, json};

use crate::account_pool::AccountAuth;
use crate::config::{EffectiveReasoningConfig, with_config};
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

    fn resolve_endpoint_url(
        &self,
        context: &ProviderExecutionContext,
    ) -> Result<String, ProxyError> {
        with_config(&context.config, |cfg| {
            cfg.endpoint_url(context.provider(), context.endpoint_name())
        })
        .map_err(ProxyError::Config)
    }

    fn resolve_compact_endpoint_url(
        &self,
        context: &ProviderExecutionContext,
    ) -> Result<String, ProxyError> {
        let url = self.resolve_endpoint_url(context)?;
        let trimmed = url.trim_end_matches('/');
        if trimmed.ends_with("/compact") {
            return Ok(trimmed.to_string());
        }
        if trimmed.ends_with("/responses") || trimmed.ends_with("/v1/responses") {
            return Ok(format!("{trimmed}/compact"));
        }
        Ok(url)
    }

    fn resolve_models_url(&self, context: &ProviderExecutionContext) -> Result<String, ProxyError> {
        with_config(&context.config, |cfg| {
            cfg.provider_models_url(context.provider())
        })
        .ok_or_else(|| {
            ProxyError::Provider(format!(
                "Provider '{}' does not have a models_url configured",
                context.provider()
            ))
        })
    }

    async fn get_json_from(
        &self,
        endpoint_url: &str,
        mut headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<reqwest::Response, ProxyError> {
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

        self.client
            .get(endpoint_url)
            .headers(headers)
            .timeout(std::time::Duration::from_secs(with_config(
                &context.config,
                |cfg| cfg.timeouts.read_seconds,
            )))
            .send()
            .await
            .map_err(ProxyError::Http)
    }

    async fn send_json_to(
        &self,
        endpoint_url: &str,
        payload: Value,
        mut headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<reqwest::Response, ProxyError> {
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

        self.client
            .post(endpoint_url)
            .headers(headers)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(with_config(
                &context.config,
                |cfg| cfg.timeouts.read_seconds,
            )))
            .send()
            .await
            .map_err(ProxyError::Http)
    }

    async fn send_json(
        &self,
        payload: Value,
        headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<reqwest::Response, ProxyError> {
        let endpoint_url = self.resolve_endpoint_url(context)?;
        self.send_json_to(&endpoint_url, payload, headers, context)
            .await
    }

    async fn forward_json_to(
        &self,
        endpoint_url: &str,
        payload: Value,
        headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let response = self
            .send_json_to(endpoint_url, payload, headers, context)
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

    async fn forward_json(
        &self,
        payload: Value,
        headers: HeaderMap,
        context: &ProviderExecutionContext,
    ) -> Result<Response<Body>, ProxyError> {
        let response = self.send_json(payload, headers, context).await?;

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
        let mut payload =
            serde_json::to_value(raw_request).unwrap_or_else(|_| Value::Object(Default::default()));
        apply_openai_route_overrides(&mut payload, &context);
        Box::pin(async move { self.forward_json(payload, headers, &context).await })
    }

    fn handle_compact(
        &self,
        data: CompactRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let mut payload = json!({
            "model": context.upstream_model(),
            "input": data.input,
            "instructions": data.instructions,
            "store": false,
            "temperature": with_config(&context.config, |cfg| cfg.compaction.temperature),
            "max_tokens": 4096,
            "stream": false
        });
        apply_openai_route_overrides(&mut payload, &context);
        Box::pin(async move {
            let endpoint_url = self.resolve_compact_endpoint_url(&context)?;
            self.forward_json_to(&endpoint_url, payload, headers, &context)
                .await
        })
    }

    fn list_models(
        &self,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<String>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move {
            let models_url = self.resolve_models_url(&context)?;
            let resp = self
                .get_json_from(&models_url, HeaderMap::new(), &context)
                .await?;

            let status = resp.status();
            let body = resp.text().await.map_err(ProxyError::Http)?;
            if !status.is_success() {
                return Err(ProxyError::Provider(format!(
                    "Upstream models endpoint returned status {} body={}",
                    status, body
                )));
            }

            let parsed: Value =
                serde_json::from_str(&body).map_err(|e| ProxyError::Provider(e.to_string()))?;
            let data = parsed
                .get("data")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ProxyError::Provider("Upstream /models missing data[]".into()))?;

            let mut out = Vec::new();
            for item in data {
                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                    out.push(id.to_string());
                }
            }
            Ok(out)
        })
    }

    fn probe_account(
        &self,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ProxyError>> + Send + '_>>
    {
        let mut payload = json!({
            "model": context.upstream_model(),
            "input": "health check",
            "store": false,
            "max_tokens": 1,
            "stream": false
        });
        apply_openai_route_overrides(&mut payload, &context);
        Box::pin(async move {
            let response = self.send_json(payload, HeaderMap::new(), &context).await?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProxyError::Auth(format!(
                    "OpenAI recovery probe unauthorized ({}): {}",
                    status, body
                )));
            }
            if !status.is_success() {
                return Err(ProxyError::Provider(format!(
                    "OpenAI recovery probe failed ({}): {}",
                    status, body
                )));
            }
            Ok(())
        })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(OpenAiProvider {
            client: self.client.clone(),
        })
    }
}

fn apply_openai_route_overrides(payload: &mut Value, context: &ProviderExecutionContext) {
    let Some(object) = payload.as_object_mut() else {
        return;
    };
    object.insert(
        "model".into(),
        Value::String(context.upstream_model().to_string()),
    );
    if let Some(reasoning) = context.reasoning() {
        object.insert(
            "reasoning".into(),
            json!({
                "effort": reasoning_effort(reasoning)
            }),
        );
    }
}

fn reasoning_effort(reasoning: &EffectiveReasoningConfig) -> &'static str {
    if let Some(preset) = reasoning.preset.as_deref() {
        return match preset {
            "none" => "none",
            "minimal" => "minimal",
            "low" => "low",
            "medium" => "medium",
            "high" => "high",
            "xhigh" => "xhigh",
            _ => budget_to_effort(reasoning.budget),
        };
    }
    budget_to_effort(reasoning.budget)
}

fn budget_to_effort(budget: u64) -> &'static str {
    match budget {
        0 => "none",
        1..=2048 => "minimal",
        2049..=4096 => "low",
        4097..=16384 => "medium",
        16385..=32768 => "high",
        _ => "xhigh",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account_pool::{Account, AccountAuth, ResolvedRoute};
    use crate::config::{
        AccessControlConfig, CompactionConfig, ModelsConfig, ProviderConfig, ProvidersConfig,
        ReasoningConfig, RoutingConfig, RoutingHealthConfig, ServerConfig, StickyRoutingConfig,
        TimeoutsConfig,
    };
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn context(
        reasoning: Option<EffectiveReasoningConfig>,
        endpoint: Option<&str>,
    ) -> ProviderExecutionContext {
        let providers: ProvidersConfig = HashMap::from([(
            "openai".into(),
            ProviderConfig::OpenAi {
                api_url: "https://example.com/v1/responses".into(),
                models_url: None,
                endpoints: HashMap::new(),
                models: Vec::new(),
            },
        )]);
        let cfg = crate::config::Config {
            config_path: PathBuf::from("/tmp/test-config.json"),
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 8765,
                log_level: "INFO".into(),
                debug_mode: false,
            },
            providers,
            models: ModelsConfig {
                served: Vec::new(),
                fallback_models: HashMap::new(),
            },
            model_discovery: crate::config::ModelDiscoveryConfig::default(),
            model_metadata: crate::config::ProviderModelMetadataConfig::new(),
            models_endpoint: crate::config::ModelsEndpointConfig::default(),
            routing: RoutingConfig {
                model_overrides: HashMap::new(),
                preferred_models: HashMap::new(),
                sticky_routing: StickyRoutingConfig::default(),
                health: RoutingHealthConfig::default(),
            },
            accounts: Vec::new(),
            access: AccessControlConfig::default(),
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 30,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: Vec::new(),
            },
        };
        let config = Arc::new(RwLock::new(cfg));
        let gemini_auth = Arc::new(crate::auth::GeminiAuthManager::new(config.clone()));

        ProviderExecutionContext {
            route: ResolvedRoute {
                requested_model: "claude-sonnet-4-6".into(),
                logical_model: "claude-sonnet-4-6".into(),
                upstream_model: "gpt-5".into(),
                endpoint: endpoint.map(str::to_string),
                provider: "openai".into(),
                account_index: 0,
                account_id: "openai-a".into(),
                cache_hit: false,
                cache_key: 0,
                preferred_target_index: 0,
                reasoning,
            },
            account: Account {
                id: "openai-a".into(),
                provider: "openai".into(),
                auth: AccountAuth::ApiKey {
                    api_key: "test-key".into(),
                },
                enabled: true,
                weight: 1,
                models: None,
            },
            config,
            gemini_auth,
        }
    }

    #[test]
    fn injects_openai_reasoning_effort() {
        let mut payload = json!({"model": "placeholder", "input": "hi"});
        apply_openai_route_overrides(
            &mut payload,
            &context(
                Some(EffectiveReasoningConfig {
                    budget: 16384,
                    level: "MEDIUM".into(),
                    preset: Some("medium".into()),
                }),
                None,
            ),
        );
        assert_eq!(payload["model"], "gpt-5");
        assert_eq!(payload["reasoning"]["effort"], "medium");
    }

    #[test]
    fn maps_inline_budget_to_effort() {
        let mut payload = json!({"model": "placeholder", "input": "hi"});
        apply_openai_route_overrides(
            &mut payload,
            &context(
                Some(EffectiveReasoningConfig {
                    budget: 50000,
                    level: "HIGH".into(),
                    preset: None,
                }),
                None,
            ),
        );
        assert_eq!(payload["reasoning"]["effort"], "xhigh");
    }

    #[test]
    fn maps_named_reasoning_presets_to_openai_effort() {
        assert_eq!(
            reasoning_effort(&EffectiveReasoningConfig {
                budget: 0,
                level: "LOW".into(),
                preset: Some("none".into()),
            }),
            "none"
        );
        assert_eq!(
            reasoning_effort(&EffectiveReasoningConfig {
                budget: 2048,
                level: "LOW".into(),
                preset: Some("minimal".into()),
            }),
            "minimal"
        );
    }
}
