use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use crate::config::{AccountConfig, RoutingHealthConfig};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountProvider {
    Gemini,
    Zai,
    OpenAi,
}

impl AccountProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            AccountProvider::Gemini => "gemini",
            AccountProvider::Zai => "zai",
            AccountProvider::OpenAi => "openai",
        }
    }
}

impl std::fmt::Display for AccountProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for AccountProvider {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gemini" => Ok(AccountProvider::Gemini),
            "zai" => Ok(AccountProvider::Zai),
            "openai" => Ok(AccountProvider::OpenAi),
            _ => Err(format!("unknown provider: {s}")),
        }
    }
}

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
    pub provider: AccountProvider,
    pub auth: AccountAuth,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing)]
    pub weight: u32,
}

impl From<AccountConfig> for Account {
    fn from(value: AccountConfig) -> Self {
        Self {
            id: value.id,
            provider: value.provider,
            auth: value.auth,
            enabled: value.enabled,
            weight: value.weight,
        }
    }
}

struct AccountState {
    alive: bool,
    consecutive_failures: u32,
    cache_key_hits: AtomicU64,
    last_failure_at: Option<SystemTime>,
    unhealthy_until: Option<SystemTime>,
}

impl AccountState {
    fn new() -> Self {
        Self {
            alive: true,
            consecutive_failures: 0,
            cache_key_hits: AtomicU64::new(0),
            last_failure_at: None,
            unhealthy_until: None,
        }
    }
}

pub struct AccountPool {
    accounts: RwLock<Vec<(Account, RwLock<AccountState>)>>,
    model_overrides: RwLock<HashMap<String, String>>,
    health: RwLock<RoutingHealthConfig>,
}

impl AccountPool {
    pub fn new() -> Self {
        Self {
            accounts: RwLock::new(Vec::new()),
            model_overrides: RwLock::new(HashMap::new()),
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

    pub fn load_model_overrides(&self, overrides: HashMap<String, String>) {
        *self.model_overrides.write() = overrides;
    }

    pub fn resolve_model(&self, model: &str) -> String {
        self.model_overrides
            .read()
            .get(model)
            .cloned()
            .unwrap_or_else(|| model.to_string())
    }

    pub fn accounts_for_provider(&self, provider: AccountProvider) -> Vec<usize> {
        let guard = self.accounts.read();
        let now = SystemTime::now();
        guard
            .iter()
            .enumerate()
            .filter_map(|(i, (account, state))| {
                if account.provider != provider {
                    return None;
                }
                let mut state = state.write();
                if let Some(until) = state.unhealthy_until
                    && until <= now
                {
                    state.alive = true;
                    state.unhealthy_until = None;
                    state.consecutive_failures = 0;
                }
                Some(i)
            })
            .collect()
    }

    pub fn get_account(&self, index: usize) -> Option<(Account, AccountSnapshot)> {
        let guard = self.accounts.read();
        guard.get(index).map(|(account, state)| {
            let mut state = state.write();
            let now = SystemTime::now();
            if let Some(until) = state.unhealthy_until
                && until <= now
            {
                state.alive = true;
                state.unhealthy_until = None;
                state.consecutive_failures = 0;
            }
            (
                account.clone(),
                AccountSnapshot {
                    alive: state.alive,
                    consecutive_failures: state.consecutive_failures,
                    cache_key_hits: state.cache_key_hits.load(Ordering::Relaxed),
                    last_failure_at: state.last_failure_at,
                    unhealthy_until: state.unhealthy_until,
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
            let should_mark_unhealthy = (is_auth_error && health.auth_failure_immediate_unhealthy)
                || state.consecutive_failures >= health.failure_threshold;
            if should_mark_unhealthy {
                state.alive = false;
                state.unhealthy_until = Some(now + Duration::from_secs(health.cooldown_seconds));
                warn!(
                    "Account {} ({}) marked unhealthy (failures={}, auth_error={})",
                    account.id, account.provider, state.consecutive_failures, is_auth_error
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
                    provider: account.provider,
                    auth: mask_auth(&account.auth),
                    alive: state.alive,
                    consecutive_failures: state.consecutive_failures,
                    cache_key_hits: state.cache_key_hits.load(Ordering::Relaxed),
                    last_failure_at: state.last_failure_at,
                    unhealthy_until: state.unhealthy_until,
                }
            })
            .collect()
    }

    pub fn model_overrides_snapshot(&self) -> HashMap<String, String> {
        self.model_overrides.read().clone()
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
    pub provider: AccountProvider,
    pub auth: MaskedAccountAuth,
    pub alive: bool,
    pub consecutive_failures: u32,
    pub cache_key_hits: u64,
    pub last_failure_at: Option<SystemTime>,
    pub unhealthy_until: Option<SystemTime>,
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
