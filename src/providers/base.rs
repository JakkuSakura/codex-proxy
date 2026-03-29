use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::Value;
use crate::error::ProxyError;

pub trait Provider: Send + Sync {
    fn handle_request(&self, data: Value, headers: HeaderMap)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>>;

    fn handle_compact(&self, _data: Value, _headers: HeaderMap)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>> {
        Box::pin(async move { Err(ProxyError::Provider("Compaction not implemented for this provider".into())) })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync>;
}
