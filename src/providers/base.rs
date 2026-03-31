use crate::account_pool::{Account, ResolvedRoute};
use crate::auth::GeminiAuthManager;
use crate::config::{ConfigHandle, EffectiveReasoningConfig};
use crate::error::ProxyError;
use crate::schema::openai::{ChatRequest, CompactRequest, ResponsesRequest};
use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;
use std::sync::Arc;

#[derive(Clone)]
pub struct ProviderExecutionContext {
    pub route: ResolvedRoute,
    pub account: Account,
    pub config: ConfigHandle,
    pub gemini_auth: Arc<GeminiAuthManager>,
}

impl ProviderExecutionContext {
    pub fn provider(&self) -> &str {
        &self.route.provider
    }

    pub fn upstream_model(&self) -> &str {
        &self.route.upstream_model
    }

    pub fn reasoning(&self) -> Option<&EffectiveReasoningConfig> {
        self.route.reasoning.as_ref()
    }

    pub fn preferred_target_index(&self) -> usize {
        self.route.preferred_target_index
    }

    pub fn endpoint_name(&self) -> Option<&str> {
        self.route.endpoint.as_deref()
    }
}

pub trait Provider: Send + Sync {
    fn handle_request(
        &self,
        raw_request: ResponsesRequest,
        normalized_request: ChatRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    >;

    fn handle_compact(
        &self,
        data: CompactRequest,
        headers: HeaderMap,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move {
            let _ = (data, headers, context);
            Err(ProxyError::NotImplemented(
                "Compaction not implemented for this provider".into(),
            ))
        })
    }

    fn list_models(
        &self,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<String>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move {
            let _ = context;
            Err(ProxyError::NotImplemented(
                "Model discovery not implemented for this provider".into(),
            ))
        })
    }

    fn probe_account(
        &self,
        context: ProviderExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ProxyError>> + Send + '_>>
    {
        Box::pin(async move {
            let _ = context;
            Err(ProxyError::Provider(
                "Recovery probe not implemented for this provider".into(),
            ))
        })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync>;
}
