use axum::body::Body;
use axum::http::{HeaderMap, header, StatusCode, Method};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tower_http::cors::{CorsLayer, Any};
use tracing::info;

use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::normalizer;
use crate::providers;
use crate::ui;
use crate::validator;

pub fn build_router() -> Router {
    providers::initialize_registry();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    Router::new()
        .route("/", get(ui_handler))
        .route("/ui", get(ui_handler))
        .route("/config", get(ui_handler).post(config_post_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/responses", post(responses_handler))
        .route("/v1/responses/compact", post(compact_handler))
        .route("/responses/compact", post(compact_handler))
        .layer(cors)
}

async fn ui_handler() -> Response<Body> {
    ui::get_html()
}

async fn config_post_handler(Json(_data): Json<Value>) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    // Config save is a stub for now — returns current config
    Ok(Json(ui::get_current_config()))
}

async fn responses_handler(
    headers: HeaderMap,
    Json(mut data): Json<Value>,
) -> Result<Response<Body>, ProxyError> {
    validator::validate_request(&data, "/v1/responses")?;

    data["_headers"] = json!({
        "session_id": headers.get("session_id").and_then(|v| v.to_str().ok()).unwrap_or(""),
        "x-openai-subagent": headers.get("x-openai-subagent").and_then(|v| v.to_str().ok()).unwrap_or(""),
        "x-codex-turn-state": headers.get("x-codex-turn-state").and_then(|v| v.to_str().ok()).unwrap_or(""),
        "x-codex-personality": headers.get("x-codex-personality").and_then(|v| v.to_str().ok()).unwrap_or(""),
    });

    normalizer::normalize(&mut data);
    data["_is_responses_api"] = json!(true);

    let model = data.get("model").and_then(|m| m.as_str()).unwrap_or("");
    let provider = providers::get_provider(model);
    provider.handle_request(data, headers).await
}

async fn compact_handler(
    headers: HeaderMap,
    Json(data): Json<Value>,
) -> Result<Response<Body>, ProxyError> {
    validator::validate_request(&data, "/v1/responses/compact")?;

    let compaction_model = CONFIG.compaction_model.as_deref()
        .unwrap_or_else(|| CONFIG.models.first().map(|s| s.as_str()).unwrap_or("gemini-2.5-flash-lite"));

    let provider = providers::get_provider(compaction_model);
    provider.handle_compact(data, headers).await
}

pub fn print_startup_info() {
    info!("Listening on {}:{}", CONFIG.host, CONFIG.port);
    info!("Config file: {}", CONFIG.config_path.display());
    info!("Log level: {}", CONFIG.log_level);
    info!("Debug mode: {}", CONFIG.debug_mode);
    info!("Config UI: http://{}:{}/config", CONFIG.host, CONFIG.port);
}
