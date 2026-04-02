use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::{Value, json};

use crate::error::ProxyError;
use crate::providers::base::{Provider, ProviderExecutionContext};
use crate::schema::openai::{ChatRequest, CompactRequest, ResponsesRequest};

use super::openai::{
    OpenAiProvider, apply_openai_route_overrides, clamp_max_tokens, ensure_max_output_tokens,
    strip_null_object_fields,
};

pub struct OpenRouterProvider {
    inner: OpenAiProvider,
}

impl Default for OpenRouterProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterProvider {
    pub fn new() -> Self {
        Self {
            inner: OpenAiProvider::new(),
        }
    }
}

impl Provider for OpenRouterProvider {
    fn handle_request(
        &self,
        raw_request: ResponsesRequest,
        _normalized_request: ChatRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let inner = self.inner.clone();
        let mut payload =
            serde_json::to_value(raw_request).unwrap_or_else(|_| Value::Object(Default::default()));

        // OpenRouter is OpenAI-compatible but stricter about nulls, and it expects
        // Responses API token limit via max_output_tokens.
        strip_null_object_fields(&mut payload);
        ensure_max_output_tokens(&mut payload);
        ensure_openrouter_default_max_output_tokens(&mut payload, &context);
        clamp_max_tokens(&mut payload, &context);
        apply_openai_route_overrides(&mut payload, &context);

        Box::pin(async move { inner.forward_json(payload, headers, &context).await })
    }

    fn handle_compact(
        &self,
        data: CompactRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        let inner = self.inner.clone();
        let mut payload = json!({
            "model": context.upstream_model(),
            "input": data.input,
            "instructions": data.instructions,
            "store": false,
            "max_output_tokens": 4096,
            "stream": false
        });
        clamp_max_tokens(&mut payload, &context);
        apply_openai_route_overrides(&mut payload, &context);
        Box::pin(async move { inner.forward_json(payload, headers, &context).await })
    }

    fn list_models(
        &self,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<String>, ProxyError>> + Send + '_>,
    > {
        self.inner.list_models(context)
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(OpenRouterProvider::new())
    }
}

fn ensure_openrouter_default_max_output_tokens(
    payload: &mut Value,
    context: &ProviderExecutionContext,
) {
    let Some(object) = payload.as_object_mut() else {
        return;
    };
    if object.contains_key("max_output_tokens") {
        return;
    }

    let default_max_output_tokens = crate::config::with_config(&context.config, |cfg| {
        cfg.model_metadata(context.provider(), context.upstream_model())
            .and_then(|metadata| metadata.max_output_tokens)
            .unwrap_or(4096)
    });

    object.insert(
        "max_output_tokens".to_string(),
        Value::Number(default_max_output_tokens.into()),
    );
    object
        .entry("max_tokens".to_string())
        .or_insert_with(|| Value::Number(default_max_output_tokens.into()));
}
