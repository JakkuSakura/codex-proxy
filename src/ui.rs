use axum::body::Body;
use axum::http::header;
use axum::response::Response;
use serde::{Deserialize, Serialize};

use crate::account_pool::AccountStatus;
use crate::config::with_config;
use crate::state::AppState;

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
    pub server: UiServerConfig,
    pub providers: crate::config::ProvidersConfig,
    pub models: UiModelsConfig,
    pub routing: UiRoutingConfig,
    pub health: crate::config::RoutingHealthConfig,
    pub accounts: Vec<AccountStatus>,
    pub reasoning: crate::config::ReasoningConfig,
    pub timeouts: crate::config::TimeoutsConfig,
    pub compaction: crate::config::CompactionConfig,
    pub stats: UiStats,
}

#[derive(Clone, Debug, Serialize)]
pub struct UiServerConfig {
    pub host: String,
    pub port: u16,
    pub log_level: String,
    pub debug_mode: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct UiModelsConfig {
    pub served: Vec<String>,
    pub fallback_models: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UiRoutingConfig {
    pub model_overrides: std::collections::HashMap<String, String>,
    pub model_provider_priority:
        std::collections::HashMap<String, Vec<crate::config::RouteTargetConfig>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UiStats {
    pub account_count: usize,
    pub sticky_binding_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UiConfigUpdate {
    #[serde(default)]
    pub server: Option<serde_json::Value>,
}

pub fn get_current_config(state: &AppState) -> UiConfig {
    with_config(state.config(), |cfg| UiConfig {
        server: UiServerConfig {
            host: cfg.server.host.clone(),
            port: cfg.server.port,
            log_level: cfg.server.log_level.clone(),
            debug_mode: cfg.server.debug_mode,
        },
        providers: cfg.providers.clone(),
        models: UiModelsConfig {
            served: cfg.models.served.clone(),
            fallback_models: cfg.models.fallback_models.clone(),
        },
        routing: UiRoutingConfig {
            model_overrides: cfg.routing.model_overrides.clone(),
            model_provider_priority: cfg.routing.model_provider_priority.clone(),
        },
        health: cfg.health.clone(),
        accounts: state.accounts().all_accounts_snapshot(),
        reasoning: cfg.reasoning.clone(),
        timeouts: cfg.timeouts.clone(),
        compaction: cfg.compaction.clone(),
        stats: UiStats {
            account_count: state.accounts().account_count(),
            sticky_binding_count: state.routing().snapshot_size(),
        },
    })
}

pub fn apply_and_save(
    _data: &UiConfigUpdate,
    state: &AppState,
) -> Result<UiConfig, crate::error::ProxyError> {
    Ok(get_current_config(state))
}
