use axum::body::Body;
use axum::http::header;
use axum::response::Response;
use serde::{Deserialize, Serialize};

use crate::config::CONFIG;

const HTML: &str = include_str!("ui/index.html");

pub fn get_html() -> Response<Body> {
    Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(Body::from(HTML))
        .unwrap()
}

#[derive(Clone, Debug, Serialize)]
pub struct UiConfig {
    pub port: u16,
    pub log_level: String,
    pub debug_mode: bool,
    pub z_ai_api_key: String,
    pub gemini_api_key: String,
    pub models: Vec<String>,
    pub compaction_model: String,
    pub request_timeout_connect: u64,
    pub request_timeout_read: u64,
    pub reasoning_effort: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UiConfigUpdate {
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub log_level: Option<String>,
    #[serde(default)]
    pub debug_mode: Option<bool>,
    #[serde(default)]
    pub z_ai_api_key: Option<String>,
    #[serde(default)]
    pub gemini_api_key: Option<String>,
    #[serde(default)]
    pub models: Option<Vec<String>>,
    #[serde(default)]
    pub compaction_model: Option<String>,
    #[serde(default)]
    pub request_timeout_connect: Option<u64>,
    #[serde(default)]
    pub request_timeout_read: Option<u64>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

pub fn get_current_config() -> UiConfig {
    UiConfig {
        port: CONFIG.port,
        log_level: CONFIG.log_level.clone(),
        debug_mode: CONFIG.debug_mode,
        z_ai_api_key: CONFIG.z_ai_api_key.clone(),
        gemini_api_key: CONFIG.gemini_api_key.clone(),
        models: CONFIG.models.clone(),
        compaction_model: CONFIG.compaction_model.clone().unwrap_or_default(),
        request_timeout_connect: CONFIG.request_timeout_connect,
        request_timeout_read: CONFIG.request_timeout_read,
        reasoning_effort: CONFIG.reasoning.default_effort.clone(),
    }
}

pub fn apply_and_save(_data: &UiConfigUpdate) -> Result<UiConfig, crate::error::ProxyError> {
    Ok(get_current_config())
}
