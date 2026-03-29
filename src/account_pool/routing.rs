use super::pool::{AccountPool, AccountProvider};
use crate::error::ProxyError;
use crate::schema::openai::{ChatContent, ChatMessage};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use tracing::debug;

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub account_index: usize,
    pub cache_hit: bool,
    pub cache_key: u64,
}

#[derive(Debug, Clone)]
pub struct ResolvedRoute {
    pub requested_model: String,
    pub upstream_model: String,
    pub provider: AccountProvider,
    pub account_index: usize,
    pub account_id: String,
    pub cache_hit: bool,
    pub cache_key: u64,
}

pub struct RoutingState {
    sticky_bindings: RwLock<HashMap<u64, usize>>,
}

impl RoutingState {
    pub fn new() -> Self {
        Self {
            sticky_bindings: RwLock::new(HashMap::new()),
        }
    }

    pub fn bind_on_success(&self, cache_key: u64, account_index: usize) {
        self.sticky_bindings
            .write()
            .insert(cache_key, account_index);
    }

    pub fn snapshot_size(&self) -> usize {
        self.sticky_bindings.read().len()
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
    pub fn route(
        pool: &AccountPool,
        state: &RoutingState,
        provider: AccountProvider,
        candidate_indices: &[usize],
        messages: &[ChatMessage],
    ) -> Option<RoutingDecision> {
        if candidate_indices.is_empty() {
            return None;
        }

        let cache_key = compute_cache_key(&message_signature(messages));

        if let Some(&bound_idx) = state.sticky_bindings.read().get(&cache_key) {
            if let Some((account, snapshot)) = pool.get_account(bound_idx)
                && account.provider == provider
                && snapshot.alive
                && candidate_indices.contains(&bound_idx)
            {
                pool.increment_cache_hits(bound_idx);
                debug!(
                    "KV-cache hit: key={} -> account {} ({})",
                    cache_key, account.id, account.provider
                );
                return Some(RoutingDecision {
                    account_index: bound_idx,
                    cache_hit: true,
                    cache_key,
                });
            }
        }

        let mut best_idx = None;
        let mut best_score = u32::MAX;
        let mut best_hits = u64::MAX;

        for &idx in candidate_indices {
            if let Some((_account, snapshot)) = pool.get_account(idx) {
                if !snapshot.alive {
                    continue;
                }
                if snapshot.consecutive_failures < best_score
                    || (snapshot.consecutive_failures == best_score
                        && snapshot.cache_key_hits < best_hits)
                {
                    best_score = snapshot.consecutive_failures;
                    best_hits = snapshot.cache_key_hits;
                    best_idx = Some(idx);
                }
            }
        }

        best_idx.map(|idx| RoutingDecision {
            account_index: idx,
            cache_hit: false,
            cache_key,
        })
    }

    pub fn resolve_route(
        pool: &AccountPool,
        state: &RoutingState,
        requested_model: &str,
        upstream_model: String,
        provider: AccountProvider,
        messages: &[ChatMessage],
    ) -> Result<ResolvedRoute, ProxyError> {
        let candidates = pool.accounts_for_provider(provider);
        let decision =
            Self::route(pool, state, provider, &candidates, messages).ok_or_else(|| {
                ProxyError::Provider(format!(
                    "No healthy accounts available for provider {}",
                    provider
                ))
            })?;
        let (account, _) = pool.get_account(decision.account_index).ok_or_else(|| {
            ProxyError::Internal(format!(
                "Selected account index {} is missing",
                decision.account_index
            ))
        })?;
        Ok(ResolvedRoute {
            requested_model: requested_model.to_string(),
            upstream_model,
            provider,
            account_index: decision.account_index,
            account_id: account.id,
            cache_hit: decision.cache_hit,
            cache_key: decision.cache_key,
        })
    }
}
