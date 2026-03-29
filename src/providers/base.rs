use crate::error::ProxyError;
use crate::schema::openai::{ChatRequest, CompactRequest};
use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;

pub trait Provider: Send + Sync {
    fn handle_request(
        &self,
        data: ChatRequest,
        headers: HeaderMap,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    >;

    fn handle_compact(
        &self,
        _data: CompactRequest,
        _headers: HeaderMap,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Response<Body>, ProxyError>> + Send + '_>,
    > {
        Box::pin(async move {
            Err(ProxyError::Provider(
                "Compaction not implemented for this provider".into(),
            ))
        })
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync>;
}
