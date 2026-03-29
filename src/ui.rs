use axum::body::Body;
use axum::http::header;
use axum::response::Response;
use serde::{Deserialize, Serialize};

use crate::account_pool::{AccountPool, AccountStatus, RoutingState};
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
    pub server: UiServerConfig,
    pub providers: crate::config::ProvidersConfig,
    pub models: UiModelsConfig,
    pub routing: UiRoutingConfig,
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
    pub compaction_model: String,
    pub fallback_models: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UiRoutingConfig {
    pub model_overrides: std::collections::HashMap<String, String>,
    pub sticky_routing: crate::config::StickyRoutingConfig,
    pub health: crate::config::RoutingHealthConfig,
    pub provider_prefixes: std::collections::HashMap<String, crate::account_pool::AccountProvider>,
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

pub fn get_current_config(account_pool: &AccountPool, routing_state: &RoutingState) -> UiConfig {
    UiConfig {
        server: UiServerConfig {
            host: CONFIG.server.host.clone(),
            port: CONFIG.server.port,
            log_level: CONFIG.server.log_level.clone(),
            debug_mode: CONFIG.server.debug_mode,
        },
        providers: CONFIG.providers.clone(),
        models: UiModelsConfig {
            served: CONFIG.models.served.clone(),
            compaction_model: CONFIG.compaction_model().unwrap_or_default().to_string(),
            fallback_models: CONFIG.models.fallback_models.clone(),
        },
        routing: UiRoutingConfig {
            model_overrides: CONFIG.routing.model_overrides.clone(),
            sticky_routing: CONFIG.routing.sticky_routing.clone(),
            health: CONFIG.routing.health.clone(),
            provider_prefixes: CONFIG.routing.provider_prefixes.clone(),
        },
        accounts: account_pool.all_accounts_snapshot(),
        reasoning: CONFIG.reasoning.clone(),
        timeouts: CONFIG.timeouts.clone(),
        compaction: CONFIG.compaction.clone(),
        stats: UiStats {
            account_count: account_pool.account_count(),
            sticky_binding_count: routing_state.snapshot_size(),
        },
    }
}

pub fn apply_and_save(
    _data: &UiConfigUpdate,
    account_pool: &AccountPool,
    routing_state: &RoutingState,
) -> Result<UiConfig, crate::error::ProxyError> {
    Ok(get_current_config(account_pool, routing_state))
}
