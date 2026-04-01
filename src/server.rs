use axum::body::Body;
use axum::body::to_bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::Response;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;
use std::io;
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
            header::HeaderName::from_static("x-codex-proxy-session"),
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
        .route("/api/usage/series", get(api_usage_series_get))
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
            cfg.health.clone(),
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
            cfg.routing.model_provider_priority.clone(),
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
        for served_model in served {
            let logical_model = with_config(state.config(), |cfg| cfg.resolve_logical_model(&served_model));
            let routing_targets = targets.get(&logical_model).cloned().unwrap_or_default();
            let default_target = routing_targets.first().cloned();
            let default_target_metadata = default_target.as_ref().and_then(|target| {
                metadata
                    .get(&target.provider)
                    .and_then(|models| models.get(&target.model))
            });

            out.insert(
                served_model.clone(),
                json!({
                    "id": served_model,
                    "object": "model",
                    "owned_by": "codex-proxy",
                    "metadata": {
                        "logical_model": logical_model,
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

    let request_bytes = serde_json::to_vec(&data)
        .map(|v| v.len() as u64)
        .unwrap_or(0);

    let session_key = resolve_session_key(&state, &headers, &data)?;
    let cache_key_override = Some(session_key.cache_key_override());
    let normalized = normalizer::normalize(data.clone());
    let route = resolve_response_route(
        &state,
        &data.model,
        &normalized.messages,
        cache_key_override,
    )?;
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

    let auto_cfg = with_config(state.config(), |cfg| cfg.auto_compaction.clone());
    let provider_type = with_config(state.config(), |cfg| cfg.provider_type(&route.provider))
        .map_err(ProxyError::Config)?;

    let mut attempt = 0u32;
    let mut current_request = data.clone();
    let mut current_normalized = normalized;
    let mut auto_compacted = false;
    let final_result = loop {
        let result = provider
            .handle_request(
                current_request.clone(),
                current_normalized.clone(),
                headers.clone(),
                context.clone(),
            )
            .await;

        match result {
            Ok(resp) => break Ok(resp),
            Err(err) => {
                if !auto_cfg.enabled
                    || attempt >= auto_cfg.max_attempts_per_request
                    || !is_context_length_error(&err)
                {
                    break Err(err);
                }

                attempt += 1;
                let compacted = auto_compact_request(
                    &state,
                    &current_request,
                    &auto_cfg,
                    provider_type,
                    cache_key_override,
                    &current_normalized.messages,
                )
                .await?;
                current_request = compacted;
                current_normalized = normalizer::normalize(current_request.clone());
                auto_compacted = true;
            }
        }
    };

    let final_result = match final_result {
        Ok(resp) => {
            let mut resp = finalize_response(&state, resp, &session_key).await;
            if auto_compacted {
                resp.headers_mut().insert(
                    header::HeaderName::from_static("x-codex-proxy-auto-compacted"),
                    header::HeaderValue::from_static("true"),
                );
            }
            Ok(resp)
        }
        Err(err) => Err(err),
    };

    record_and_apply_result(&state, &access_key, &context, request_bytes, final_result)
}

async fn compact_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(data): Json<CompactRequest>,
) -> Result<Response<Body>, ProxyError> {
    let access_key = crate::access::authenticate_request(state.config(), &headers)?;
    validator::validate_compact_request(&data)?;
    let request_bytes = serde_json::to_vec(&data)
        .map(|v| v.len() as u64)
        .unwrap_or(0);

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
    let route = resolve_compaction_route(&state, &normalized.messages, None)?;
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
    record_and_apply_result(&state, &access_key, &context, request_bytes, result)
}

fn resolve_response_route(
    state: &AppState,
    requested_model: &str,
    messages: &[ChatMessage],
    cache_key_override: Option<u64>,
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
        cache_key_override,
    )
}

fn resolve_compaction_route(
    state: &AppState,
    messages: &[ChatMessage],
    cache_key_override: Option<u64>,
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
        cache_key_override,
    )
}

fn resolve_session_key(
    state: &AppState,
    headers: &HeaderMap,
    request: &ResponsesRequest,
) -> Result<crate::session::SessionKey, ProxyError> {
    let (header_name, metadata_key) = with_config(state.config(), |cfg| {
        (
            cfg.session.header_name.clone(),
            cfg.session.metadata_key.clone(),
        )
    });

    if let Some(key) = crate::session::extract_session_key_from_headers(headers, &header_name) {
        return Ok(key);
    }

    let metadata_json = request
        .metadata
        .as_ref()
        .and_then(|m| serde_json::to_value(m).ok());
    if let Some(key) =
        crate::session::extract_session_key_from_metadata(metadata_json.as_ref(), &metadata_key)
    {
        return Ok(key);
    }

    if let Some(prev) = request.previous_response_id.as_deref()
        && let Some(key) = state.sessions().get_by_previous_response_id(prev)
    {
        return Ok(key);
    }

    Ok(crate::session::SessionKey::generate())
}

async fn finalize_response(
    state: &AppState,
    response: Response<Body>,
    session_key: &crate::session::SessionKey,
) -> Response<Body> {
    let header_name = with_config(state.config(), |cfg| cfg.session.header_name.clone());
    let mut response = response;
    crate::session::attach_session_header(response.headers_mut(), &header_name, session_key);

    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !content_type.starts_with("application/json") {
        return response;
    }

    if let Some(len) = response
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        && len > 4 * 1024 * 1024
    {
        return response;
    }

    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, 4 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return Response::from_parts(parts, Body::empty());
        }
    };

    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
        if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
            state
                .sessions()
                .record_response_id(id.to_string(), session_key.clone());
        }
    }

    Response::from_parts(parts, Body::from(bytes))
}

fn is_context_length_error(error: &ProxyError) -> bool {
    let lower = match error {
        ProxyError::Provider(msg) => msg.to_ascii_lowercase(),
        ProxyError::Validation(msg) => msg.to_ascii_lowercase(),
        ProxyError::Http(err) => err.to_string().to_ascii_lowercase(),
        ProxyError::Auth(_) => return false,
        ProxyError::Config(_) => return false,
        ProxyError::NotImplemented(_) => return false,
        ProxyError::Internal(_) => return false,
    };
    let has_status = lower.contains("(400)") || lower.contains("(413)") || lower.contains(" 413");
    let has_context = lower.contains("context")
        || lower.contains("token")
        || lower.contains("prompt")
        || lower.contains("maximum")
        || lower.contains("too long");
    let has_signal = lower.contains("context length")
        || lower.contains("maximum context")
        || lower.contains("prompt is too long")
        || lower.contains("too many tokens")
        || lower.contains("token limit")
        || lower.contains("exceeds")
        || lower.contains("too large");

    (has_status && has_context) || has_signal
}

async fn auto_compact_request(
    state: &AppState,
    request: &ResponsesRequest,
    cfg: &crate::config::AutoCompactionConfig,
    provider_type: crate::config::ProviderType,
    cache_key_override: Option<u64>,
    normalized_messages: &[ChatMessage],
) -> Result<ResponsesRequest, ProxyError> {
    use crate::schema::openai::{Content, InputItem, ResponsesInput};

    let input = request
        .input
        .clone()
        .ok_or_else(|| ProxyError::Validation("Auto-compaction requires request.input".into()))?;
    let ResponsesInput::Items(items) = input else {
        return Err(ProxyError::Validation(
            "Auto-compaction requires request.input to be an items[] array".into(),
        ));
    };

    let tail = cfg.tail_items_to_keep.max(1);
    if items.len() <= tail {
        return Err(ProxyError::Provider(
            "Auto-compaction skipped: not enough input items to compact".into(),
        ));
    }

    let prefix_items = items[..items.len() - tail].to_vec();
    let tail_items = items[items.len() - tail..].to_vec();

    let rewritten_items = match provider_type {
        crate::config::ProviderType::OpenAi => {
            let encrypted = run_native_compaction(
                state,
                prefix_items,
                crate::config::AUTO_COMPACTION_COMPACT_INSTRUCTIONS.to_string(),
                cache_key_override,
                normalized_messages,
            )
            .await?;

            let mut out = Vec::with_capacity(1 + tail_items.len());
            out.push(InputItem {
                item_type: "compaction".into(),
                id: None,
                call_id: None,
                role: None,
                name: None,
                content: None,
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: None,
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: Some(encrypted),
            });
            out.extend(tail_items);
            out
        }
        _ => {
            let summary = run_summary_compaction(
                state,
                prefix_items,
                crate::config::AUTO_COMPACTION_SUMMARY_INSTRUCTIONS.to_string(),
                cache_key_override,
                normalized_messages,
            )
            .await?;

            let mut out = Vec::with_capacity(1 + tail_items.len());
            out.push(InputItem {
                item_type: "message".into(),
                id: None,
                call_id: None,
                role: Some("system".into()),
                name: None,
                content: Some(Content::Text(format!(
                    "[codex-proxy auto-compaction summary]\n{}",
                    summary
                ))),
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: None,
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: None,
            });
            out.extend(tail_items);
            out
        }
    };

    let mut next = request.clone();
    next.input = Some(ResponsesInput::Items(rewritten_items));
    next.previous_response_id = None;
    Ok(next)
}

async fn run_native_compaction(
    state: &AppState,
    prefix_items: Vec<crate::schema::openai::InputItem>,
    instructions: String,
    cache_key_override: Option<u64>,
    normalized_messages: &[ChatMessage],
) -> Result<String, ProxyError> {
    use crate::schema::openai::{CompactRequest, Instructions, ResponsesInput};

    let route = resolve_compaction_route(state, normalized_messages, cache_key_override)?;
    let provider_type = with_config(state.config(), |cfg| cfg.provider_type(&route.provider))
        .map_err(ProxyError::Config)?;
    if provider_type != crate::config::ProviderType::OpenAi {
        return Err(ProxyError::NotImplemented(
            "Native compaction requires an OpenAI-compatible compaction provider".into(),
        ));
    }
    let provider = crate::providers::get_provider(state, &route.provider);
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

    let compact = CompactRequest {
        input: ResponsesInput::Items(prefix_items),
        instructions: Instructions::Text(instructions),
    };

    let resp = provider
        .handle_compact(compact, HeaderMap::new(), context)
        .await?;
    let bytes = to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .map_err(|e| ProxyError::Internal(format!("Failed to read compaction response: {e}")))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| ProxyError::Provider(e.to_string()))?;
    find_encrypted_content(&value).ok_or_else(|| {
        ProxyError::Provider(
            "Compaction response did not include an encrypted_content artifact".into(),
        )
    })
}

fn find_encrypted_content(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Object(map) => {
            if map
                .get("type")
                .and_then(|v| v.as_str())
                .is_some_and(|t| t == "compaction")
            {
                if let Some(enc) = map.get("encrypted_content").and_then(|v| v.as_str()) {
                    return Some(enc.to_string());
                }
            }
            for v in map.values() {
                if let Some(found) = find_encrypted_content(v) {
                    return Some(found);
                }
            }
            None
        }
        serde_json::Value::Array(arr) => arr.iter().find_map(find_encrypted_content),
        _ => None,
    }
}

async fn run_summary_compaction(
    state: &AppState,
    prefix_items: Vec<crate::schema::openai::InputItem>,
    instructions: String,
    cache_key_override: Option<u64>,
    normalized_messages: &[ChatMessage],
) -> Result<String, ProxyError> {
    let route = resolve_compaction_route(state, normalized_messages, cache_key_override)?;
    let provider_type = with_config(state.config(), |cfg| cfg.provider_type(&route.provider))
        .map_err(ProxyError::Config)?;
    if provider_type != crate::config::ProviderType::OpenAi {
        return Err(ProxyError::NotImplemented(
            "Summary compaction requires an OpenAI-compatible compaction provider".into(),
        ));
    }
    let provider = crate::providers::get_provider(state, &route.provider);
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

    let prompt = build_summary_prompt(&prefix_items, &instructions);
    let raw = ResponsesRequest {
        model: "__compaction__".into(),
        input: Some(crate::schema::openai::ResponsesInput::Text(prompt)),
        instructions: None,
        previous_response_id: None,
        store: Some(false),
        metadata: None,
        tools: None,
        tool_choice: None,
        temperature: Some(0.1),
        top_p: None,
        max_tokens: Some(4096),
        stream: Some(false),
        include: None,
    };
    let normalized = normalizer::normalize(raw.clone());
    let resp = provider
        .handle_request(raw, normalized, HeaderMap::new(), context)
        .await?;
    let bytes = to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .map_err(|e| ProxyError::Internal(format!("Failed to read summary response: {e}")))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| ProxyError::Provider(e.to_string()))?;
    extract_output_text(&value).ok_or_else(|| {
        ProxyError::Provider("Summary compaction response did not include output text".into())
    })
}

fn build_summary_prompt(
    prefix_items: &[crate::schema::openai::InputItem],
    instructions: &str,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(&mut out, "{instructions}");
    let _ = writeln!(&mut out, "");
    let _ = writeln!(&mut out, "Conversation:");
    for item in prefix_items {
        let role = item.role.as_deref().unwrap_or("unknown");
        let mut line = String::new();
        if let Some(content) = &item.content {
            match content {
                crate::schema::openai::Content::Text(text) => {
                    line.push_str(text);
                }
                crate::schema::openai::Content::Parts(parts) => {
                    for part in parts {
                        if let Some(text) = &part.text {
                            line.push_str(text);
                        }
                    }
                }
                _ => {}
            }
        }
        if line.is_empty() {
            continue;
        }
        let _ = writeln!(&mut out, "- {role}: {line}");
    }
    out
}

fn extract_output_text(value: &serde_json::Value) -> Option<String> {
    if let Some(text) = value.get("output_text").and_then(|v| v.as_str()) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    let output = value.get("output")?.as_array()?;
    for item in output {
        if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
            let mut parts = String::new();
            for part in content {
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    parts.push_str(text);
                }
            }
            let trimmed = parts.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod auto_compaction_tests {
    use super::*;

    #[test]
    fn detects_context_length_provider_error() {
        let err = ProxyError::Provider(
            "OpenAI request failed (400): This model's maximum context length is 128000 tokens"
                .into(),
        );
        assert!(is_context_length_error(&err));
    }

    #[test]
    fn finds_encrypted_content_in_nested_shape() {
        let value = serde_json::json!({
            "compaction": {
                "output": [
                    {"type":"compaction","encrypted_content":"ENC"}
                ]
            }
        });
        assert_eq!(find_encrypted_content(&value).as_deref(), Some("ENC"));
    }

    #[test]
    fn extracts_output_text_from_output_text_field() {
        let value = serde_json::json!({
            "output_text": " hello "
        });
        assert_eq!(extract_output_text(&value).as_deref(), Some("hello"));
    }

    #[test]
    fn extracts_output_text_from_output_array() {
        let value = serde_json::json!({
            "output": [
                {
                    "content": [{"type":"output_text","text":"abc"}]
                }
            ]
        });
        assert_eq!(extract_output_text(&value).as_deref(), Some("abc"));
    }

    #[test]
    fn builds_summary_prompt_from_message_items() {
        use crate::schema::openai::{Content, InputItem};
        let items = vec![
            InputItem {
                item_type: "message".into(),
                id: None,
                call_id: None,
                role: Some("user".into()),
                name: None,
                content: Some(Content::Text("hi".into())),
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: None,
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: None,
            },
            InputItem {
                item_type: "message".into(),
                id: None,
                call_id: None,
                role: Some("assistant".into()),
                name: None,
                content: Some(Content::Text("hello".into())),
                reasoning_content: None,
                thought_signature: None,
                thought: None,
                arguments: None,
                input: None,
                action: None,
                command: None,
                cwd: None,
                working_directory: None,
                changes: None,
                output: None,
                stdout: None,
                stderr: None,
                encrypted_content: None,
            },
        ];
        let prompt = build_summary_prompt(&items, "do a summary");
        assert!(prompt.contains("do a summary"));
        assert!(prompt.contains("- user: hi"));
        assert!(prompt.contains("- assistant: hello"));
    }
}

fn record_and_apply_result(
    state: &AppState,
    access_key: &Option<AuthenticatedKey>,
    context: &ProviderExecutionContext,
    request_bytes: u64,
    result: Result<Response<Body>, ProxyError>,
) -> Result<Response<Body>, ProxyError> {
    let status = match &result {
        Ok(resp) => resp.status(),
        Err(err) => err.status_code(),
    };
    let usage_handle = state.usage().record_request_start_handle(
        access_key.as_ref().map(|k| k.id.as_str()),
        &context.route.provider,
        &context.route.account_id,
        &context.route.upstream_model,
        status,
        context.route.cache_hit,
        request_bytes,
    );

    match result {
        Ok(response) => {
            let response = wrap_response_for_usage(state.clone(), usage_handle, response);
            state.accounts().mark_success(context.route.account_index);
            state.routing().bind_on_success(&context.route);
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

struct UsageResponseTracker {
    state: AppState,
    handle: crate::usage::UsageHandle,
    response_bytes: u64,
}

impl Drop for UsageResponseTracker {
    fn drop(&mut self) {
        self.state
            .usage()
            .record_response_bytes(&self.handle, self.response_bytes);
    }
}

fn wrap_response_for_usage(
    state: AppState,
    handle: crate::usage::UsageHandle,
    response: Response<Body>,
) -> Response<Body> {
    let (parts, body) = response.into_parts();
    let mut tracker = UsageResponseTracker {
        state,
        handle,
        response_bytes: 0,
    };

    let mut data_stream = body.into_data_stream();
    let stream = async_stream::stream! {
        while let Some(chunk) = data_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    tracker.response_bytes += bytes.len() as u64;
                    yield Ok::<Bytes, io::Error>(bytes);
                }
                Err(err) => {
                    yield Err(io::Error::new(io::ErrorKind::Other, err.to_string()));
                    break;
                }
            }
        }
        drop(tracker);
    };

    Response::from_parts(parts, Body::from_stream(stream))
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
    let mut next = data.into_runtime(config_path.clone());
    next.validate().map_err(ProxyError::Config)?;
    next.save_to_path(&config_path)
        .map_err(ProxyError::Config)?;

    with_config_mut(state.config(), |cfg| {
        *cfg = next.clone();
    });
    state
        .sessions()
        .set_ttl(Duration::from_secs(next.session.response_id_ttl_seconds));
    state.accounts().configure_health(next.health.clone());
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
struct UsageSeriesQuery {
    #[serde(default)]
    bucket_seconds: Option<u64>,
    #[serde(default)]
    window_seconds: Option<u64>,
}

async fn api_usage_series_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<UsageSeriesQuery>,
) -> Result<Json<serde_json::Value>, ProxyError> {
    authenticate_admin(&state, &headers)?;

    let bucket_seconds = query.bucket_seconds.unwrap_or(300);
    if bucket_seconds == 0 || bucket_seconds % 60 != 0 {
        return Err(ProxyError::Validation(
            "bucket_seconds must be a positive multiple of 60".into(),
        ));
    }
    let window_seconds = query.window_seconds.unwrap_or(24 * 60 * 60);
    if window_seconds == 0 {
        return Err(ProxyError::Validation(
            "window_seconds must be positive".into(),
        ));
    }

    let buckets = state
        .usage()
        .snapshot_series(bucket_seconds, window_seconds);
    Ok(Json(json!({
        "bucket_seconds": bucket_seconds,
        "window_seconds": window_seconds,
        "buckets": buckets,
    })))
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
            plaintext: None,
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
