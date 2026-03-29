use axum::body::Body;
use axum::http::header;
use axum::response::Response;
use serde_json::{Value, json};

use crate::config::CONFIG;

const HTML: &str = include_str!("ui/index.html");

pub fn get_html() -> Response<Body> {
    Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(Body::from(HTML))
        .unwrap()
}

pub fn get_current_config() -> Value {
    json!({
        "port": CONFIG.port,
        "log_level": &CONFIG.log_level,
        "debug_mode": CONFIG.debug_mode,
        "z_ai_api_key": &CONFIG.z_ai_api_key,
        "gemini_api_key": &CONFIG.gemini_api_key,
        "models": CONFIG.models.clone(),
        "compaction_model": CONFIG.compaction_model.as_deref().unwrap_or(""),
        "request_timeout_connect": CONFIG.request_timeout_connect,
        "request_timeout_read": CONFIG.request_timeout_read,
        "reasoning_effort": &CONFIG.reasoning.default_effort,
    })
}

pub fn apply_and_save(_data: &Value) -> Result<Value, crate::error::ProxyError> {
    Ok(get_current_config())
}
