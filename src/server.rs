use axum::body::Body;
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::normalizer;
use crate::providers;
use crate::schema::openai::{CompactRequest, ResponsesRequest};
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

async fn config_post_handler(
    Json(_data): Json<ui::UiConfigUpdate>,
) -> Result<Json<ui::UiConfig>, (StatusCode, Json<ui::UiConfig>)> {
    // Config save is a stub for now — returns current config
    Ok(Json(ui::get_current_config()))
}

async fn responses_handler(
    headers: HeaderMap,
    Json(data): Json<ResponsesRequest>,
) -> Result<Response<Body>, ProxyError> {
    validator::validate_responses_request(&data)?;

    let chat_req = normalizer::normalize(data);
    let provider = providers::get_provider(&chat_req.model);
    provider.handle_request(chat_req, headers).await
}

async fn compact_handler(
    headers: HeaderMap,
    Json(data): Json<CompactRequest>,
) -> Result<Response<Body>, ProxyError> {
    validator::validate_compact_request(&data)?;

    let compaction_model = CONFIG.compaction_model.as_deref().unwrap_or_else(|| {
        CONFIG
            .models
            .first()
            .map(|s| s.as_str())
            .unwrap_or("gemini-2.5-flash-lite")
    });

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
