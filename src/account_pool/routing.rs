use super::pool::AccountPool;
use crate::config::{EffectiveReasoningConfig, RouteTargetConfig};
use crate::error::{ProviderError, ProxyError};
use crate::schema::openai::{ChatContent, ChatMessage};
use parking_lot::RwLock;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use tracing::debug;

#[derive(Debug, Clone)]
pub struct RouteCandidate {
    pub requested_model: String,
    pub logical_model: String,
    pub provider: String,
    pub upstream_model: String,
    pub endpoint: Option<String>,
    pub reasoning: Option<EffectiveReasoningConfig>,
    pub priority_index: usize,
}

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub account_index: usize,
    pub cache_hit: bool,
    pub cache_key: u64,
    pub preferred_target_index: usize,
}

#[derive(Debug, Clone)]
pub struct ResolvedRoute {
    pub requested_model: String,
    pub logical_model: String,
    pub upstream_model: String,
    pub endpoint: Option<String>,
    pub provider: String,
    pub account_index: usize,
    pub account_id: String,
    pub cache_hit: bool,
    pub cache_key: u64,
    pub preferred_target_index: usize,
    pub reasoning: Option<EffectiveReasoningConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StickyRouteBinding {
    provider: String,
    model: String,
    endpoint: Option<String>,
    account_index: usize,
    preferred_target_index: usize,
}

pub struct RoutingState {
    sticky_bindings: RwLock<HashMap<u64, StickyRouteBinding>>,
}

impl RoutingState {
    pub fn new() -> Self {
        Self {
            sticky_bindings: RwLock::new(HashMap::new()),
        }
    }

    pub fn bind_on_success(&self, route: &ResolvedRoute) {
        self.sticky_bindings.write().insert(
            route.cache_key,
            StickyRouteBinding {
                provider: route.provider.clone(),
                model: route.upstream_model.clone(),
                endpoint: route.endpoint.clone(),
                account_index: route.account_index,
                preferred_target_index: route.preferred_target_index,
            },
        );
    }

    pub fn snapshot_size(&self) -> usize {
        self.sticky_bindings.read().len()
    }

    pub fn clear(&self) {
        self.sticky_bindings.write().clear();
    }
}

fn compute_cache_key(messages_prefix: &[(String, String)]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    for (role, content) in messages_prefix {
        role.hash(&mut hasher);
        hasher.write_u8(0);
        content.hash(&mut hasher);
        hasher.write_u8(0);
    }
    hasher.finish()
}

fn message_signature(messages: &[ChatMessage]) -> Vec<(String, String)> {
    messages
        .iter()
        .map(|message| {
            let mut content = match message.content.as_ref() {
                Some(ChatContent::Text(text)) => text.clone(),
                Some(ChatContent::Parts(parts)) => parts
                    .iter()
                    .filter_map(|part| part.text.clone())
                    .collect::<Vec<_>>()
                    .join(""),
                None => String::new(),
            };
            if let Some(reasoning) = &message.reasoning_content {
                content.push_str(reasoning);
            }
            for tool_call in &message.tool_calls {
                content.push_str(&tool_call.function.name);
                content.push_str(&tool_call.function.arguments);
            }
            (message.role.clone(), content)
        })
        .collect()
}

pub struct Router;

impl Router {
    pub fn build_candidates(
        requested_model: &str,
        logical_model: &str,
        targets: &[RouteTargetConfig],
        mut reasoning_for: impl FnMut(
            &RouteTargetConfig,
        ) -> Result<Option<EffectiveReasoningConfig>, ProxyError>,
    ) -> Result<Vec<RouteCandidate>, ProxyError> {
        targets
            .iter()
            .enumerate()
            .map(|(priority_index, target)| {
                Ok(RouteCandidate {
                    requested_model: requested_model.to_string(),
                    logical_model: logical_model.to_string(),
                    provider: target.provider.clone(),
                    upstream_model: target.model.clone(),
                    endpoint: target.endpoint.clone(),
                    reasoning: reasoning_for(target)?,
                    priority_index,
                })
            })
            .collect()
    }

    pub fn route(
        pool: &AccountPool,
        state: &RoutingState,
        candidates: &[RouteCandidate],
        messages: &[ChatMessage],
        cache_key_override: Option<u64>,
    ) -> Option<RoutingDecision> {
        if candidates.is_empty() {
            return None;
        }

        let cache_key =
            cache_key_override.unwrap_or_else(|| compute_cache_key(&message_signature(messages)));

        if let Some(binding) = state.sticky_bindings.read().get(&cache_key).cloned() {
            for candidate in candidates {
                if candidate.provider != binding.provider
                    || candidate.upstream_model != binding.model
                    || candidate.endpoint != binding.endpoint
                    || candidate.priority_index != binding.preferred_target_index
                {
                    continue;
                }
                if let Some((account, snapshot)) = pool.get_account(binding.account_index)
                    && account.provider == candidate.provider
                    && account.supports_model(&candidate.upstream_model)
                    && snapshot.alive
                {
                    pool.increment_cache_hits(binding.account_index);
                    debug!(
                        "KV-cache hit: key={} -> account {} ({}) model {}",
                        cache_key, account.id, account.provider, candidate.upstream_model
                    );
                    return Some(RoutingDecision {
                        account_index: binding.account_index,
                        cache_hit: true,
                        cache_key,
                        preferred_target_index: candidate.priority_index,
                    });
                }
            }
        }

        for candidate in candidates {
            let candidate_indices =
                pool.healthy_compatible_accounts_for_target(&RouteTargetConfig {
                    provider: candidate.provider.clone(),
                    model: candidate.upstream_model.clone(),
                    endpoint: candidate.endpoint.clone(),
                    reasoning: None,
                });
            if candidate_indices.is_empty() {
                continue;
            }

            let mut sorted_indices = candidate_indices;
            sorted_indices.sort_by_key(|idx| {
                let (weight, failures, hits) = pool
                    .get_account(*idx)
                    .map(|(account, snapshot)| {
                        (
                            account.weight,
                            snapshot.consecutive_failures,
                            snapshot.cache_key_hits,
                        )
                    })
                    .unwrap_or((u32::MAX, u32::MAX, u64::MAX));
                (weight, failures, Reverse(hits), *idx)
            });

            if let Some(account_index) = sorted_indices.first().copied() {
                return Some(RoutingDecision {
                    account_index,
                    cache_hit: false,
                    cache_key,
                    preferred_target_index: candidate.priority_index,
                });
            }
        }

        None
    }

    pub fn resolve_route(
        pool: &AccountPool,
        state: &RoutingState,
        candidates: &[RouteCandidate],
        messages: &[ChatMessage],
        cache_key_override: Option<u64>,
    ) -> Result<ResolvedRoute, ProxyError> {
        let decision = Self::route(pool, state, candidates, messages, cache_key_override)
            .ok_or_else(|| {
                ProxyError::Provider(ProviderError::new(
                    None,
                    "No compatible healthy accounts available for any preferred target",
                ))
            })?;
        let candidate = candidates
            .iter()
            .find(|candidate| candidate.priority_index == decision.preferred_target_index)
            .ok_or_else(|| {
                ProxyError::Internal(format!(
                    "Selected preferred target index {} is missing",
                    decision.preferred_target_index
                ))
            })?;
        let (account, _) = pool.get_account(decision.account_index).ok_or_else(|| {
            ProxyError::Internal(format!(
                "Selected account index {} is missing",
                decision.account_index
            ))
        })?;
        Ok(ResolvedRoute {
            requested_model: candidate.requested_model.clone(),
            logical_model: candidate.logical_model.clone(),
            upstream_model: candidate.upstream_model.clone(),
            endpoint: candidate.endpoint.clone(),
            provider: candidate.provider.clone(),
            account_index: decision.account_index,
            account_id: account.id,
            cache_hit: decision.cache_hit,
            cache_key: decision.cache_key,
            preferred_target_index: decision.preferred_target_index,
            reasoning: candidate.reasoning.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account_pool::{Account, AccountAuth};
    use crate::schema::openai::ChatMessage;

    fn account(id: &str, provider: &str, models: Option<Vec<&str>>, weight: u32) -> Account {
        Account {
            id: id.into(),
            provider: provider.into(),
            auth: AccountAuth::ApiKey {
                api_key: "test-key".into(),
            },
            enabled: true,
            weight,
            models: models.map(|values| values.into_iter().map(str::to_string).collect()),
        }
    }

    fn messages() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".into(),
            content: Some(ChatContent::Text("hello".into())),
            reasoning_content: None,
            thought_signature: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }]
    }

    fn messages_with_text(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".into(),
            content: Some(ChatContent::Text(text.into())),
            reasoning_content: None,
            thought_signature: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }]
    }

    fn candidate(priority_index: usize, provider: &str, model: &str) -> RouteCandidate {
        RouteCandidate {
            requested_model: "claude-sonnet-4-6".into(),
            logical_model: "claude-sonnet-4-6".into(),
            provider: provider.into(),
            upstream_model: model.into(),
            endpoint: None,
            reasoning: None,
            priority_index,
        }
    }

    #[test]
    fn prefers_first_healthy_candidate_in_order() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![
            account("zai", "zai", Some(vec!["glm-4.6"]), 1),
            account("openai", "openai", Some(vec!["gpt-4.1"]), 1),
        ]);
        pool.mark_success(0);
        pool.mark_success(1);
        let state = RoutingState::new();
        let route = Router::resolve_route(
            &pool,
            &state,
            &[
                candidate(0, "openai", "gpt-4.1"),
                candidate(1, "zai", "glm-4.6"),
            ],
            &messages(),
            None,
        )
        .unwrap();
        assert_eq!(route.provider, "openai");
        assert_eq!(route.upstream_model, "gpt-4.1");
    }

    #[test]
    fn reuses_sticky_binding_when_still_healthy() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![account("openai", "openai", Some(vec!["gpt-4.1"]), 1)]);
        pool.mark_success(0);
        let state = RoutingState::new();
        let first = Router::resolve_route(
            &pool,
            &state,
            &[candidate(0, "openai", "gpt-4.1")],
            &messages(),
            None,
        )
        .unwrap();
        state.bind_on_success(&first);

        let second = Router::resolve_route(
            &pool,
            &state,
            &[candidate(0, "openai", "gpt-4.1")],
            &messages(),
            None,
        )
        .unwrap();
        assert!(second.cache_hit);
        assert_eq!(second.account_index, first.account_index);
    }

    #[test]
    fn preserves_sticky_binding_with_cache_key_override() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![account("openai", "openai", Some(vec!["gpt-4.1"]), 1)]);
        pool.mark_success(0);
        let state = RoutingState::new();

        let cache_key_override = Some(42);
        let first = Router::resolve_route(
            &pool,
            &state,
            &[candidate(0, "openai", "gpt-4.1")],
            &messages_with_text("hello"),
            cache_key_override,
        )
        .unwrap();
        state.bind_on_success(&first);

        let second = Router::resolve_route(
            &pool,
            &state,
            &[candidate(0, "openai", "gpt-4.1")],
            &messages_with_text("different"),
            cache_key_override,
        )
        .unwrap();

        assert!(second.cache_hit);
        assert_eq!(second.cache_key, 42);
        assert_eq!(second.account_index, first.account_index);
    }
}
