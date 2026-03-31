use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::Response;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::json;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tower_http::cors::{Any, CorsLayer};
use tracing::warn;

use crate::access::{AuthenticatedKey, require_admin};
use crate::config::{PersistedConfig, with_config, with_config_mut};
use crate::error::ProxyError;
use crate::normalizer;
use crate::providers::base::ProviderExecutionContext;
use crate::schema::openai::{ChatMessage, CompactRequest, ResponsesRequest};
use crate::state::AppState;
use crate::ui;
use crate::validator;

pub fn build_router(state: AppState) -> Router {
    initialize_runtime_state(&state);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            header::HeaderName::from_static("x-api-key"),
            header::HeaderName::from_static("x-codex-proxy-key"),
        ]);

    Router::new()
        .route("/health", get(health_handler))
        .route("/favicon.ico", get(favicon_handler))
        .route("/", get(ui_handler))
        .route("/ui", get(ui_handler))
        .route("/v1/models", get(models_handler))
        .route("/models", get(models_handler))
        .route("/api/config", get(api_config_get).post(api_config_put))
        .route("/api/accounts", get(api_accounts_get))
        .route(
            "/api/access-keys",
            get(api_access_keys_get).post(api_access_keys_create),
        )
        .route("/api/access-keys/{id}", delete(api_access_keys_delete))
        .route("/api/usage/keys", get(api_usage_keys_get))
        .route("/api/usage/accounts", get(api_usage_accounts_get))
        .route("/api/models", get(api_models_get))
        .route("/v1/responses", post(responses_handler))
        .route("/responses", post(responses_handler))
        .route("/v1/responses/compact", post(compact_handler))
        .route("/responses/compact", post(compact_handler))
        .layer(cors)
        .with_state(state)
}

fn initialize_runtime_state(state: &AppState) {
    let (health, accounts) = with_config(state.config(), |cfg| {
        (
            cfg.routing.health.clone(),
            cfg.accounts.clone().into_iter().map(Into::into).collect(),
        )
    });
    state.accounts().configure_health(health);
    state.accounts().load_accounts(accounts);
    start_recovery_probe_loop(state);
    start_model_discovery_loop(state);
}

fn start_recovery_probe_loop(state: &AppState) {
    if state.recovery_started_flag().swap(true, Ordering::AcqRel) {
        return;
    }

    let state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            run_recovery_probe_pass(&state).await;
        }
    });
}

async fn run_recovery_probe_pass(state: &AppState) {
    for index in state.accounts().recovery_candidates() {
        if !state.accounts().begin_recovery_probe(index) {
            continue;
        }

        let Some((account, snapshot)) = state.accounts().get_account(index) else {
            state.accounts().finish_recovery_probe(index, false);
            continue;
        };

        let provider = crate::providers::get_provider(state, &account.provider);
        let route = crate::account_pool::ResolvedRoute {
            requested_model: "__recovery_probe__".into(),
            logical_model: "__recovery_probe__".into(),
            upstream_model: account
                .models
                .as_ref()
                .and_then(|models| models.first().cloned())
                .or_else(|| {
                    with_config(state.config(), |cfg| {
                        cfg.provider_catalog(&account.provider)
                            .and_then(|models| models.first().cloned())
                    })
                })
                .unwrap_or_else(|| "__probe__".into()),
            endpoint: None,
            provider: account.provider.clone(),
            account_index: index,
            account_id: account.id.clone(),
            cache_hit: false,
            cache_key: 0,
            preferred_target_index: usize::MAX,
            reasoning: Some(crate::config::EffectiveReasoningConfig {
                budget: 0,
                level: "LOW".into(),
                preset: Some("none".into()),
            }),
        };
        let context = ProviderExecutionContext {
            route,
            account,
            config: state.config().clone(),
            gemini_auth: state.gemini_auth(),
        };
        let success = provider.probe_account(context).await.is_ok();
        state.accounts().finish_recovery_probe(index, success);

        if !success && !snapshot.alive {
            warn!("Recovery probe failed for account index {}", index);
        }
    }
}

fn start_model_discovery_loop(state: &AppState) {
    if !with_config(state.config(), |cfg| cfg.model_discovery.enabled) {
        return;
    }
    if state
        .model_discovery_started_flag()
        .swap(true, Ordering::AcqRel)
    {
        return;
    }

    let interval_seconds = with_config(state.config(), |cfg| cfg.model_discovery.interval_seconds);
    let state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_seconds.max(5)));
        loop {
            interval.tick().await;
            run_model_discovery_pass(&state).await;
        }
    });
}

async fn run_model_discovery_pass(state: &AppState) {
    let provider_names = with_config(state.config(), |cfg| {
        let mut names: Vec<String> = cfg.providers.keys().cloned().collect();
        names.sort();
        names
    });

    for provider in provider_names {
        discover_models_for_provider(state, &provider).await;
    }
}

async fn discover_models_for_provider(state: &AppState, provider: &str) {
    let Some((account_index, account)) = state.accounts().first_account_for_provider(provider)
    else {
        state
            .model_catalog()
            .update_error(provider, "No configured account for provider".into());
        return;
    };

    let route = crate::account_pool::ResolvedRoute {
        requested_model: "__models__".into(),
        logical_model: "__models__".into(),
        upstream_model: "__models__".into(),
        endpoint: None,
        provider: provider.to_string(),
        account_index,
        account_id: account.id.clone(),
        cache_hit: false,
        cache_key: 0,
        preferred_target_index: usize::MAX,
        reasoning: None,
    };

    let context = ProviderExecutionContext {
        route,
        account,
        config: state.config().clone(),
        gemini_auth: state.gemini_auth(),
    };

    let provider_impl = crate::providers::get_provider(state, provider);
    match provider_impl.list_models(context).await {
        Ok(models) => state.model_catalog().update_success(provider, models),
        Err(ProxyError::NotImplemented(_)) => {
            let models = with_config(state.config(), |cfg| {
                cfg.provider_catalog(provider)
                    .cloned()
                    .unwrap_or_else(Vec::new)
            });
            if models.is_empty() {
                state.model_catalog().update_error(
                    provider,
                    "Provider does not support model discovery and no provider.models are configured"
                        .into(),
                );
            } else {
                state.model_catalog().update_success(provider, models);
            }
        }
        Err(err) => state
            .model_catalog()
            .update_error(provider, err.to_string()),
    }
}

async fn ui_handler() -> Response<Body> {
    ui::get_html()
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn favicon_handler() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn models_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    let _ = crate::access::authenticate_request(state.config(), &headers)?;
    let (served, targets, metadata, source) = with_config(state.config(), |cfg| {
        (
            cfg.models.served.clone(),
            cfg.routing.preferred_models.clone(),
            cfg.model_metadata.clone(),
            cfg.models_endpoint.source,
        )
    });

    use std::collections::{BTreeMap, BTreeSet};

    #[derive(Default)]
    struct DiscoveredModel {
        providers: BTreeSet<String>,
    }

    let mut out: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    if source == crate::config::ModelsEndpointSource::Discovered
        || source == crate::config::ModelsEndpointSource::Both
    {
        let mut discovered: BTreeMap<String, DiscoveredModel> = BTreeMap::new();
        for snapshot in state.model_catalog().snapshot() {
            for model_id in snapshot.models {
                discovered
                    .entry(model_id)
                    .or_default()
                    .providers
                    .insert(snapshot.provider.clone());
            }
        }

        for (model_id, item) in discovered {
            let providers: Vec<String> = item.providers.into_iter().collect();
            let mut provider_metadata = Vec::new();
            for provider in &providers {
                let meta = metadata
                    .get(provider)
                    .and_then(|models| models.get(&model_id));
                provider_metadata.push(json!({
                    "provider": provider,
                    "metadata": meta,
                }));
            }

            out.insert(
                model_id.clone(),
                json!({
                    "id": model_id,
                    "object": "model",
                    "owned_by": "codex-proxy",
                    "metadata": {
                        "providers": providers,
                        "provider_metadata": provider_metadata,
                        "served": false,
                    }
                }),
            );
        }
    }

    if source == crate::config::ModelsEndpointSource::Served
        || source == crate::config::ModelsEndpointSource::Both
    {
        for logical_model in served {
            let routing_targets = targets.get(&logical_model).cloned().unwrap_or_default();
            let default_target = routing_targets.first().cloned();
            let default_target_metadata = default_target.as_ref().and_then(|target| {
                metadata
                    .get(&target.provider)
                    .and_then(|models| models.get(&target.model))
            });

            out.insert(
                logical_model.clone(),
                json!({
                    "id": logical_model,
                    "object": "model",
                    "owned_by": "codex-proxy",
                    "metadata": {
                        "routing_targets": routing_targets,
                        "default_target": default_target,
                        "default_target_metadata": default_target_metadata,
                        "served": true,
                    }
                }),
            );
        }
    }

    Ok(Json(json!({
        "object": "list",
        "data": out.into_values().collect::<Vec<_>>(),
    })))
}

async fn api_models_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    Ok(Json(json!({
        "providers": state.model_catalog().snapshot(),
    })))
}

async fn responses_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(data): Json<ResponsesRequest>,
) -> Result<Response<Body>, ProxyError> {
    let access_key = crate::access::authenticate_request(state.config(), &headers)?;
    validator::validate_responses_request(&data)?;
    if !with_config(state.config(), |cfg| {
        cfg.is_served_model_allowed(&data.model)
    }) {
        return Err(ProxyError::Validation(format!(
            "Requested model '{}' is not in models.served",
            data.model
        )));
    }

    let normalized = normalizer::normalize(data.clone());
    let route = resolve_response_route(&state, &data.model, &normalized.messages)?;
    let provider = crate::providers::get_provider(&state, &route.provider);
    let (account, _) = state
        .accounts()
        .get_account(route.account_index)
        .ok_or_else(|| ProxyError::Internal("Resolved account missing from pool".into()))?;
    let context = ProviderExecutionContext {
        route: route.clone(),
        account,
        config: state.config().clone(),
        gemini_auth: state.gemini_auth(),
    };

    let result = provider
        .handle_request(data, normalized, headers, context.clone())
        .await;
    record_and_apply_result(&state, &access_key, &context, result)
}

async fn compact_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(data): Json<CompactRequest>,
) -> Result<Response<Body>, ProxyError> {
    let access_key = crate::access::authenticate_request(state.config(), &headers)?;
    validator::validate_compact_request(&data)?;

    let normalized = normalizer::normalize(ResponsesRequest {
        model: "__compaction__".to_string(),
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
    let route = resolve_compaction_route(&state, &normalized.messages)?;
    let provider_type = with_config(state.config(), |cfg| cfg.provider_type(&route.provider))
        .map_err(ProxyError::Config)?;
    if provider_type != crate::config::ProviderType::OpenAi {
        return Err(ProxyError::NotImplemented(format!(
            "Compaction is only supported for OpenAI-compatible providers; resolved provider '{}' does not support native compaction",
            route.provider
        )));
    }
    let provider = crate::providers::get_provider(&state, &route.provider);
    let (account, _) = state
        .accounts()
        .get_account(route.account_index)
        .ok_or_else(|| ProxyError::Internal("Resolved account missing from pool".into()))?;
    let context = ProviderExecutionContext {
        route: route.clone(),
        account,
        config: state.config().clone(),
        gemini_auth: state.gemini_auth(),
    };

    let result = provider
        .handle_compact(data, headers, context.clone())
        .await;
    record_and_apply_result(&state, &access_key, &context, result)
}

fn resolve_response_route(
    state: &AppState,
    requested_model: &str,
    messages: &[ChatMessage],
) -> Result<crate::account_pool::ResolvedRoute, ProxyError> {
    let logical_model = with_config(state.config(), |cfg| {
        cfg.resolve_logical_model(requested_model)
    });
    let targets = with_config(state.config(), |cfg| {
        cfg.preferred_targets_for_model(requested_model)
            .map(|targets| targets.to_vec())
    })
    .ok_or_else(|| {
        ProxyError::Validation(format!(
            "No preferred route targets configured for logical model '{}'",
            logical_model
        ))
    })?;
    let candidates = crate::account_pool::Router::build_candidates(
        requested_model,
        &logical_model,
        &targets,
        |target| {
            with_config(state.config(), |cfg| {
                cfg.resolve_reasoning(target.reasoning.as_ref())
            })
            .map_err(ProxyError::Config)
        },
    )?;
    crate::account_pool::Router::resolve_route(
        state.accounts(),
        state.routing(),
        &candidates,
        messages,
    )
}

fn resolve_compaction_route(
    state: &AppState,
    messages: &[ChatMessage],
) -> Result<crate::account_pool::ResolvedRoute, ProxyError> {
    let targets = with_config(state.config(), |cfg| cfg.compaction_targets().to_vec());
    if targets.is_empty() {
        return Err(ProxyError::Validation(
            "No compaction route targets configured".into(),
        ));
    }
    let candidates: Vec<crate::account_pool::RouteCandidate> =
        crate::account_pool::Router::build_candidates(
            "__compaction__",
            "__compaction__",
            &targets,
            |target| {
                with_config(state.config(), |cfg| {
                    cfg.resolve_reasoning(target.reasoning.as_ref())
                })
                .map_err(ProxyError::Config)
            },
        )?;
    crate::account_pool::Router::resolve_route(
        state.accounts(),
        state.routing(),
        &candidates,
        messages,
    )
}

fn record_and_apply_result(
    state: &AppState,
    access_key: &Option<AuthenticatedKey>,
    context: &ProviderExecutionContext,
    result: Result<Response<Body>, ProxyError>,
) -> Result<Response<Body>, ProxyError> {
    let status = match &result {
        Ok(resp) => resp.status(),
        Err(err) => err.status_code(),
    };
    state.usage().record_request_result(
        access_key.as_ref().map(|k| k.id.as_str()),
        &context.route.provider,
        &context.route.account_id,
        &context.route.upstream_model,
        status,
        context.route.cache_hit,
    );

    match result {
        Ok(response) => {
            state.accounts().mark_success(context.route.account_index);
            if with_config(state.config(), |cfg| cfg.routing.sticky_routing.enabled) {
                state.routing().bind_on_success(&context.route);
            }
            Ok(response)
        }
        Err(error) => {
            let is_auth_error = is_auth_failure(&error);
            state
                .accounts()
                .mark_failure(context.route.account_index, is_auth_error);
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

fn authenticate_admin(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<Option<AuthenticatedKey>, ProxyError> {
    let key = crate::access::authenticate_request(state.config(), headers)?;
    require_admin(key.as_ref())?;
    Ok(key)
}

async fn api_config_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let (config_path, persisted) = with_config(state.config(), |cfg| {
        (cfg.config_path.display().to_string(), cfg.to_persisted())
    });
    let mut value = serde_json::to_value(persisted).unwrap_or_else(|_| json!({}));
    if let Some(accounts) = value.get_mut("accounts").and_then(|v| v.as_array_mut()) {
        for account in accounts {
            if let Some(auth) = account.get_mut("auth") {
                if let Some(obj) = auth.as_object_mut()
                    && obj.get("type").and_then(|v| v.as_str()) == Some("api_key")
                {
                    if let Some(api_key) = obj.get("api_key").and_then(|v| v.as_str()) {
                        let masked = if api_key.len() <= 8 {
                            "***".to_string()
                        } else {
                            format!("{}...{}", &api_key[..4], &api_key[api_key.len() - 4..])
                        };
                        obj.insert("api_key".to_string(), serde_json::Value::String(masked));
                    }
                }
            }
        }
    }
    Ok(Json(json!({ "config_path": config_path, "config": value })))
}

async fn api_config_put(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(data): Json<PersistedConfig>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let config_path = with_config(state.config(), |cfg| cfg.config_path.clone());
    let next = data.into_runtime(config_path.clone());
    next.validate().map_err(ProxyError::Config)?;
    next.save_to_path(&config_path)
        .map_err(ProxyError::Config)?;

    with_config_mut(state.config(), |cfg| {
        *cfg = next.clone();
    });
    state
        .accounts()
        .configure_health(next.routing.health.clone());
    state
        .accounts()
        .load_accounts(next.accounts.into_iter().map(Into::into).collect());
    state.routing().clear();

    api_config_get(State(state), headers).await
}

async fn api_accounts_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    Ok(Json(json!({
        "accounts": state.accounts().all_accounts_snapshot(),
        "sticky_bindings": state.routing().snapshot_size(),
    })))
}

async fn api_usage_keys_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    Ok(Json(json!({ "keys": state.usage().snapshot_keys() })))
}

async fn api_usage_accounts_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    Ok(Json(
        json!({ "accounts": state.usage().snapshot_accounts() }),
    ))
}

#[derive(serde::Deserialize)]
struct AccessKeyCreateRequest {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub role: Option<crate::config::AccessKeyRole>,
}

async fn api_access_keys_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let keys = with_config(state.config(), |cfg| cfg.access.keys.clone());
    Ok(Json(json!({ "keys": keys })))
}

async fn api_access_keys_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AccessKeyCreateRequest>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let plaintext = crate::access::generate_access_key();
    let sha = crate::access::sha256_hex(&plaintext);

    let key_id = req
        .id
        .unwrap_or_else(|| format!("key-{}", sha[..12].to_string()));
    let enabled = req.enabled.unwrap_or(true);
    let role = req.role.unwrap_or(crate::config::AccessKeyRole::Api);
    let name = req.name;

    let config_path = with_config(state.config(), |cfg| cfg.config_path.clone());
    let next = with_config_mut(state.config(), |cfg| {
        if cfg.access.keys.iter().any(|k| k.id == key_id) {
            return Err(ProxyError::Validation(format!(
                "access key id '{}' already exists",
                key_id
            )));
        }
        cfg.access.keys.push(crate::config::AccessKeyConfig {
            id: key_id.clone(),
            key_sha256: sha.clone(),
            name,
            enabled,
            role: Some(role),
            is_admin: false,
        });
        cfg.validate().map_err(ProxyError::Config)?;
        Ok(cfg.clone())
    })?;
    next.save_to_path(&config_path)
        .map_err(ProxyError::Config)?;
    Ok(Json(json!({
        "id": key_id,
        "plaintext": plaintext,
        "key_sha256": sha,
        "role": role,
    })))
}

async fn api_access_keys_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;
    let config_path = with_config(state.config(), |cfg| cfg.config_path.clone());
    let next = with_config_mut(state.config(), |cfg| {
        let before = cfg.access.keys.len();
        cfg.access.keys.retain(|k| k.id != id);
        if cfg.access.keys.len() == before {
            return Err(ProxyError::Validation(format!(
                "access key '{}' not found",
                id
            )));
        }
        cfg.validate().map_err(ProxyError::Config)?;
        Ok(cfg.clone())
    })?;
    next.save_to_path(&config_path)
        .map_err(ProxyError::Config)?;
    Ok(Json(json!({ "ok": true })))
}
