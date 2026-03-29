use axum::body::Body;
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use once_cell::sync::Lazy;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::account_pool::{AccountPool, Router as AccountRouter, RoutingState};
use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::normalizer;
use crate::providers;
use crate::providers::base::ProviderExecutionContext;
use crate::schema::openai::{CompactRequest, ResponsesRequest};
use crate::ui;
use crate::validator;

static ACCOUNT_POOL: Lazy<AccountPool> = Lazy::new(AccountPool::new);
static ROUTING_STATE: Lazy<RoutingState> = Lazy::new(RoutingState::new);

pub fn build_router() -> Router {
    providers::initialize_registry();
    initialize_runtime_state();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    Router::new()
        .route("/", get(ui_handler))
        .route("/ui", get(ui_handler))
        .route("/config", get(config_get_handler).post(config_post_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/responses", post(responses_handler))
        .route("/v1/responses/compact", post(compact_handler))
        .route("/responses/compact", post(compact_handler))
        .layer(cors)
}

fn initialize_runtime_state() {
    ACCOUNT_POOL.configure_health(CONFIG.routing.health.clone());
    ACCOUNT_POOL.load_model_overrides(CONFIG.routing.model_overrides.clone());
    ACCOUNT_POOL.load_accounts(
        CONFIG
            .accounts
            .clone()
            .into_iter()
            .map(Into::into)
            .collect(),
    );
}

async fn ui_handler() -> Response<Body> {
    ui::get_html()
}

async fn config_get_handler() -> Json<ui::UiConfig> {
    Json(ui::get_current_config(&ACCOUNT_POOL, &ROUTING_STATE))
}

async fn config_post_handler(
    Json(_data): Json<ui::UiConfigUpdate>,
) -> Result<Json<ui::UiConfig>, (StatusCode, Json<ui::UiConfig>)> {
    Ok(Json(
        ui::apply_and_save(&_data, &ACCOUNT_POOL, &ROUTING_STATE).map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ui::get_current_config(&ACCOUNT_POOL, &ROUTING_STATE)),
            )
        })?,
    ))
}

async fn responses_handler(
    headers: HeaderMap,
    Json(data): Json<ResponsesRequest>,
) -> Result<Response<Body>, ProxyError> {
    validator::validate_responses_request(&data)?;
    if !CONFIG.is_served_model_allowed(&data.model) {
        return Err(ProxyError::Validation(format!(
            "Requested model '{}' is not in models.served",
            data.model
        )));
    }

    let normalized = normalizer::normalize(data.clone());
    let route = resolve_route(&data.model, &normalized.messages)?;
    let provider = providers::get_provider(route.provider);
    let (account, _) = ACCOUNT_POOL
        .get_account(route.account_index)
        .ok_or_else(|| ProxyError::Internal("Resolved account missing from pool".into()))?;
    let context = ProviderExecutionContext {
        route: route.clone(),
        account,
    };

    let result = provider
        .handle_request(data, normalized, headers, context.clone())
        .await;
    apply_routing_result(&context, result)
}

async fn compact_handler(
    headers: HeaderMap,
    Json(data): Json<CompactRequest>,
) -> Result<Response<Body>, ProxyError> {
    validator::validate_compact_request(&data)?;

    let compaction_model = CONFIG.compaction_model().ok_or_else(|| {
        ProxyError::Config(crate::error::ConfigError::InvalidValue(
            "models.compaction_model is not configured and no served model fallback exists".into(),
        ))
    })?;
    let normalized = normalizer::normalize(ResponsesRequest {
        model: compaction_model.to_string(),
        input: Some(data.input.clone()),
        instructions: None,
        previous_response_id: None,
        store: None,
        metadata: None,
        tools: None,
        tool_choice: None,
        temperature: None,
        top_p: None,
        max_tokens: None,
        stream: Some(false),
        include: None,
    });
    let route = resolve_route(compaction_model, &normalized.messages)?;
    let provider = providers::get_provider(route.provider);
    let (account, _) = ACCOUNT_POOL
        .get_account(route.account_index)
        .ok_or_else(|| ProxyError::Internal("Resolved account missing from pool".into()))?;
    let context = ProviderExecutionContext {
        route: route.clone(),
        account,
    };

    let result = provider
        .handle_compact(data, headers, context.clone())
        .await;
    apply_routing_result(&context, result)
}

fn resolve_route(
    requested_model: &str,
    messages: &[crate::schema::openai::ChatMessage],
) -> Result<crate::account_pool::ResolvedRoute, ProxyError> {
    let upstream_model = ACCOUNT_POOL.resolve_model(requested_model);
    let provider = CONFIG.provider_for_model(&upstream_model).ok_or_else(|| {
        ProxyError::Validation(format!(
            "Could not resolve provider for model '{}'",
            upstream_model
        ))
    })?;
    AccountRouter::resolve_route(
        &ACCOUNT_POOL,
        &ROUTING_STATE,
        requested_model,
        upstream_model,
        provider,
        messages,
    )
}

fn apply_routing_result(
    context: &ProviderExecutionContext,
    result: Result<Response<Body>, ProxyError>,
) -> Result<Response<Body>, ProxyError> {
    match result {
        Ok(response) => {
            ACCOUNT_POOL.mark_success(context.route.account_index);
            if CONFIG.routing.sticky_routing.enabled {
                ROUTING_STATE.bind_on_success(context.route.cache_key, context.route.account_index);
            }
            Ok(response)
        }
        Err(error) => {
            let is_auth_error = is_auth_failure(&error);
            ACCOUNT_POOL.mark_failure(context.route.account_index, is_auth_error);
            Err(error)
        }
    }
}

fn is_auth_failure(error: &ProxyError) -> bool {
    match error {
        ProxyError::Auth(_) => true,
        ProxyError::Http(err) => err
            .status()
            .map(|status| status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN)
            .unwrap_or(false),
        ProxyError::Provider(message) => {
            message.contains("401") || message.contains("403") || message.contains("unauthorized")
        }
        _ => false,
    }
}

pub fn print_startup_info() {
    info!("Listening on {}:{}", CONFIG.server.host, CONFIG.server.port);
    info!("Config file: {}", CONFIG.config_path.display());
    info!("Log level: {}", CONFIG.server.log_level);
    info!("Debug mode: {}", CONFIG.server.debug_mode);
    info!("Accounts loaded: {}", ACCOUNT_POOL.account_count());
    info!(
        "Config UI: http://{}:{}/config",
        CONFIG.server.host, CONFIG.server.port
    );
}
