use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use crate::config::{AccountConfig, RouteTargetConfig, RoutingHealthConfig};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AccountAuth {
    ApiKey {
        api_key: String,
    },
    GeminiOAuth {
        #[serde(default)]
        creds_path: Option<PathBuf>,
        #[serde(default)]
        client_id: Option<String>,
        #[serde(default)]
        client_secret: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: String,
    pub provider: String,
    pub auth: AccountAuth,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing)]
    pub weight: u32,
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

impl Account {
    pub fn supports_model(&self, model: &str) -> bool {
        match &self.models {
            Some(models) => models.iter().any(|allowed| allowed == model),
            None => true,
        }
    }
}

impl From<AccountConfig> for Account {
    fn from(value: AccountConfig) -> Self {
        Self {
            id: value.id,
            provider: value.provider,
            auth: value.auth,
            enabled: value.enabled,
            weight: value.weight,
            models: value.models,
        }
    }
}

struct AccountState {
    alive: bool,
    consecutive_failures: u32,
    cache_key_hits: AtomicU64,
    last_failure_at: Option<SystemTime>,
    unhealthy_until: Option<SystemTime>,
    recovery_probe_due: bool,
    probe_in_progress: AtomicBool,
}

impl AccountState {
    fn new() -> Self {
        Self {
            alive: true,
            consecutive_failures: 0,
            cache_key_hits: AtomicU64::new(0),
            last_failure_at: None,
            unhealthy_until: None,
            recovery_probe_due: false,
            probe_in_progress: AtomicBool::new(false),
        }
    }
}

pub struct AccountPool {
    accounts: RwLock<Vec<(Account, RwLock<AccountState>)>>,
    health: RwLock<RoutingHealthConfig>,
}

impl AccountPool {
    pub fn new() -> Self {
        Self {
            accounts: RwLock::new(Vec::new()),
            health: RwLock::new(RoutingHealthConfig::default()),
        }
    }

    pub fn configure_health(&self, config: RoutingHealthConfig) {
        *self.health.write() = config;
    }

    pub fn load_accounts(&self, accounts: Vec<Account>) {
        let mut next = Vec::new();
        {
            let guard = self.accounts.read();
            for account in accounts {
                if !account.enabled {
                    continue;
                }
                let existing_state = guard
                    .iter()
                    .find(|(existing, _)| existing.id == account.id)
                    .map(|(_, state)| {
                        let snapshot = state.read();
                        RwLock::new(AccountState {
                            alive: snapshot.alive,
                            consecutive_failures: snapshot.consecutive_failures,
                            cache_key_hits: AtomicU64::new(
                                snapshot.cache_key_hits.load(Ordering::Relaxed),
                            ),
                            last_failure_at: snapshot.last_failure_at,
                            unhealthy_until: snapshot.unhealthy_until,
                            recovery_probe_due: snapshot.recovery_probe_due,
                            probe_in_progress: AtomicBool::new(
                                snapshot.probe_in_progress.load(Ordering::Relaxed),
                            ),
                        })
                    })
                    .unwrap_or_else(|| RwLock::new(AccountState::new()));
                next.push((account, existing_state));
            }
        }
        *self.accounts.write() = next;
        info!(
            "Account pool loaded: {} accounts",
            self.accounts.read().len()
        );
    }

    pub fn healthy_compatible_accounts_for_target(&self, target: &RouteTargetConfig) -> Vec<usize> {
        let guard = self.accounts.read();
        guard
            .iter()
            .enumerate()
            .filter_map(|(i, (account, _))| {
                (account.provider == target.provider
                    && account.supports_model(&target.model)
                    && self.get_account(i).map(|(_, s)| s.alive).unwrap_or(false))
                .then_some(i)
            })
            .collect()
    }

    pub fn recovery_candidates(&self) -> Vec<usize> {
        let guard = self.accounts.read();
        let now = SystemTime::now();
        guard
            .iter()
            .enumerate()
            .filter_map(|(i, (_, state))| {
                let state = state.read();
                let due = state.recovery_probe_due
                    && state
                        .unhealthy_until
                        .map(|until| until <= now)
                        .unwrap_or(false)
                    && !state.probe_in_progress.load(Ordering::Relaxed);
                due.then_some(i)
            })
            .collect()
    }

    pub fn begin_recovery_probe(&self, index: usize) -> bool {
        let guard = self.accounts.read();
        let Some((_, state)) = guard.get(index) else {
            return false;
        };
        let state = state.read();
        !state.probe_in_progress.swap(true, Ordering::AcqRel)
    }

    pub fn finish_recovery_probe(&self, index: usize, success: bool) {
        let guard = self.accounts.read();
        if let Some((account, state)) = guard.get(index) {
            let mut state = state.write();
            state.probe_in_progress.store(false, Ordering::Release);
            if success {
                state.alive = true;
                state.consecutive_failures = 0;
                state.unhealthy_until = None;
                state.recovery_probe_due = false;
                info!(
                    "Recovery probe succeeded for account {} ({})",
                    account.id, account.provider
                );
            } else {
                let health = self.health.read().clone();
                let now = SystemTime::now();
                state.consecutive_failures += 1;
                let backoff_factor = state
                    .consecutive_failures
                    .saturating_sub(1)
                    .min(health.failure_threshold.saturating_sub(1));
                let cooldown_multiplier = 1u64 << backoff_factor;
                let cooldown_seconds = health.cooldown_seconds.saturating_mul(cooldown_multiplier);
                state.unhealthy_until = Some(now + Duration::from_secs(cooldown_seconds));
                state.recovery_probe_due = true;
                warn!(
                    "Recovery probe failed for account {} ({}) next backoff={}s failures={}",
                    account.id, account.provider, cooldown_seconds, state.consecutive_failures
                );
            }
        }
    }

    pub fn get_account(&self, index: usize) -> Option<(Account, AccountSnapshot)> {
        let guard = self.accounts.read();
        guard.get(index).map(|(account, state)| {
            let state = state.read();
            (
                account.clone(),
                AccountSnapshot {
                    alive: state.alive,
                    consecutive_failures: state.consecutive_failures,
                    cache_key_hits: state.cache_key_hits.load(Ordering::Relaxed),
                    last_failure_at: state.last_failure_at,
                    unhealthy_until: state.unhealthy_until,
                    recovery_probe_due: state.recovery_probe_due,
                    probe_in_progress: state.probe_in_progress.load(Ordering::Relaxed),
                },
            )
        })
    }

    pub fn mark_success(&self, index: usize) {
        let guard = self.accounts.read();
        if let Some((account, state)) = guard.get(index) {
            let mut state = state.write();
            state.alive = true;
            state.consecutive_failures = 0;
            state.unhealthy_until = None;
            state.recovery_probe_due = false;
            state.probe_in_progress.store(false, Ordering::Release);
            debug!(
                "Account {} ({}) marked healthy",
                account.id, account.provider
            );
        }
    }

    pub fn mark_failure(&self, index: usize, is_auth_error: bool) {
        let guard = self.accounts.read();
        if let Some((account, state)) = guard.get(index) {
            let mut state = state.write();
            let health = self.health.read().clone();
            let now = SystemTime::now();
            state.last_failure_at = Some(now);
            state.consecutive_failures += 1;

            let backoff_factor = state
                .consecutive_failures
                .saturating_sub(1)
                .min(health.failure_threshold.saturating_sub(1));
            let cooldown_multiplier = 1u64 << backoff_factor;
            let cooldown_seconds = health.cooldown_seconds.saturating_mul(cooldown_multiplier);

            let should_mark_unhealthy = (is_auth_error && health.auth_failure_immediate_unhealthy)
                || state.consecutive_failures >= 1;
            if should_mark_unhealthy {
                state.alive = false;
                state.recovery_probe_due = true;
                state.unhealthy_until = Some(now + Duration::from_secs(cooldown_seconds));
                state.probe_in_progress.store(false, Ordering::Release);
                warn!(
                    "Account {} ({}) marked unhealthy (failures={}, auth_error={}, backoff={}s)",
                    account.id,
                    account.provider,
                    state.consecutive_failures,
                    is_auth_error,
                    cooldown_seconds
                );
            }
        }
    }

    pub fn increment_cache_hits(&self, index: usize) {
        let guard = self.accounts.read();
        if let Some((_, state)) = guard.get(index) {
            state.read().cache_key_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn all_accounts_snapshot(&self) -> Vec<AccountStatus> {
        let guard = self.accounts.read();
        guard
            .iter()
            .map(|(account, state)| {
                let state = state.read();
                AccountStatus {
                    id: account.id.clone(),
                    provider: account.provider.clone(),
                    models: account.models.clone(),
                    weight: account.weight,
                    auth: mask_auth(&account.auth),
                    alive: state.alive,
                    consecutive_failures: state.consecutive_failures,
                    cache_key_hits: state.cache_key_hits.load(Ordering::Relaxed),
                    last_failure_at: state.last_failure_at,
                    unhealthy_until: state.unhealthy_until,
                    recovery_probe_due: state.recovery_probe_due,
                    probe_in_progress: state.probe_in_progress.load(Ordering::Relaxed),
                }
            })
            .collect()
    }

    pub fn account_count(&self) -> usize {
        self.accounts.read().len()
    }
}

pub struct AccountSnapshot {
    pub alive: bool,
    pub consecutive_failures: u32,
    pub cache_key_hits: u64,
    pub last_failure_at: Option<SystemTime>,
    pub unhealthy_until: Option<SystemTime>,
    pub recovery_probe_due: bool,
    pub probe_in_progress: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaskedAccountAuth {
    pub auth_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_masked: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creds_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id_masked: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret_masked: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccountStatus {
    pub id: String,
    pub provider: String,
    pub models: Option<Vec<String>>,
    pub weight: u32,
    pub auth: MaskedAccountAuth,
    pub alive: bool,
    pub consecutive_failures: u32,
    pub cache_key_hits: u64,
    pub last_failure_at: Option<SystemTime>,
    pub unhealthy_until: Option<SystemTime>,
    pub recovery_probe_due: bool,
    pub probe_in_progress: bool,
}

fn mask_auth(auth: &AccountAuth) -> MaskedAccountAuth {
    match auth {
        AccountAuth::ApiKey { api_key } => MaskedAccountAuth {
            auth_type: "api_key".into(),
            api_key_masked: Some(mask_key(api_key)),
            creds_path: None,
            client_id_masked: None,
            client_secret_masked: None,
        },
        AccountAuth::GeminiOAuth {
            creds_path,
            client_id,
            client_secret,
        } => MaskedAccountAuth {
            auth_type: "gemini_oauth".into(),
            api_key_masked: None,
            creds_path: creds_path.as_ref().map(|p| p.display().to_string()),
            client_id_masked: client_id.as_deref().map(mask_key),
            client_secret_masked: client_secret.as_deref().map(mask_key),
        },
    }
}

fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        return "***".into();
    }
    format!("{}...{}", &key[..4], &key[key.len() - 4..])
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn target(provider: &str, model: &str) -> RouteTargetConfig {
        RouteTargetConfig {
            provider: provider.into(),
            model: model.into(),
            endpoint: None,
            reasoning: None,
        }
    }

    #[test]
    fn filters_accounts_by_model_capability() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![
            account("a", "openai", Some(vec!["gpt-4.1"]), 1),
            account("b", "openai", Some(vec!["gpt-4.1-mini"]), 1),
        ]);

        let compatible = pool.healthy_compatible_accounts_for_target(&target("openai", "gpt-4.1"));
        assert_eq!(compatible, vec![0]);
    }

    #[test]
    fn marks_account_unhealthy_on_first_failure() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![account("a", "openai", None, 1)]);
        pool.mark_failure(0, false);
        let (_, snapshot) = pool.get_account(0).unwrap();
        assert!(!snapshot.alive);
        assert!(snapshot.recovery_probe_due);
        assert!(snapshot.unhealthy_until.is_some());
    }
}
