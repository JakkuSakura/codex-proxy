use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use tracing::{info, warn};

use crate::account_pool::{AccountAuth, AccountProvider};
use crate::error::ConfigError;

pub const GEMINI_CLI_CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";
pub const GEMINI_CLI_CLIENT_SECRET: &str = "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";

fn validate_port(port_str: &str) -> Result<u16, ConfigError> {
    let port: u16 = port_str
        .parse()
        .map_err(|_| ConfigError::InvalidPort(port_str.into()))?;
    if port == 0 {
        return Err(ConfigError::InvalidPort("port must be 1-65535".into()));
    }
    Ok(port)
}

fn validate_url(url: &str, name: &str) -> Result<String, ConfigError> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(ConfigError::InvalidUrl(format!(
            "{name} must start with http:// or https://"
        )));
    }
    Ok(url.to_string())
}

fn validate_model_prefix(prefix: &str) -> Result<String, ConfigError> {
    if prefix.is_empty()
        || !prefix
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
    {
        return Err(ConfigError::InvalidPrefix(format!(
            "model prefix must be lowercase alphanumeric, '-' or '.': {prefix}"
        )));
    }
    Ok(prefix.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub effort_levels: HashMap<String, EffortLevel>,
    pub default_effort: String,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        let mut effort_levels = HashMap::new();
        effort_levels.insert(
            "none".into(),
            EffortLevel {
                budget: 0,
                level: "LOW".into(),
            },
        );
        effort_levels.insert(
            "minimal".into(),
            EffortLevel {
                budget: 2048,
                level: "LOW".into(),
            },
        );
        effort_levels.insert(
            "low".into(),
            EffortLevel {
                budget: 4096,
                level: "LOW".into(),
            },
        );
        effort_levels.insert(
            "medium".into(),
            EffortLevel {
                budget: 16384,
                level: "MEDIUM".into(),
            },
        );
        effort_levels.insert(
            "high".into(),
            EffortLevel {
                budget: 32768,
                level: "HIGH".into(),
            },
        );
        effort_levels.insert(
            "xhigh".into(),
            EffortLevel {
                budget: 65536,
                level: "HIGH".into(),
            },
        );
        Self {
            effort_levels,
            default_effort: "medium".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffortLevel {
    pub budget: u64,
    pub level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub log_level: String,
    pub debug_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiProviderConfig {
    pub api_internal: String,
    pub api_public: String,
    pub default_client_id: String,
    pub default_client_secret: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiProviderConfig {
    pub api_url: String,
    #[serde(default)]
    pub allow_authorization_passthrough: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiProviderConfig {
    pub responses_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvidersConfig {
    pub gemini: GeminiProviderConfig,
    pub zai: ZaiProviderConfig,
    pub openai: OpenAiProviderConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    #[serde(default)]
    pub served: Vec<String>,
    #[serde(default)]
    pub compaction_model: Option<String>,
    #[serde(default)]
    pub fallback_models: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StickyRoutingConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingHealthConfig {
    #[serde(default = "default_auth_failure_immediate_unhealthy")]
    pub auth_failure_immediate_unhealthy: bool,
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    #[serde(default)]
    pub model_overrides: HashMap<String, String>,
    #[serde(default = "default_provider_prefixes")]
    pub provider_prefixes: HashMap<String, AccountProvider>,
    #[serde(default)]
    pub sticky_routing: StickyRoutingConfig,
    #[serde(default)]
    pub health: RoutingHealthConfig,
}

impl Default for StickyRoutingConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl Default for RoutingHealthConfig {
    fn default() -> Self {
        Self {
            auth_failure_immediate_unhealthy: true,
            failure_threshold: default_failure_threshold(),
            cooldown_seconds: default_cooldown_seconds(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutsConfig {
    pub connect_seconds: u64,
    pub read_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountConfig {
    pub id: String,
    pub provider: AccountProvider,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_weight")]
    pub weight: u32,
    pub auth: AccountAuth,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub config_path: PathBuf,
    pub server: ServerConfig,
    pub providers: ProvidersConfig,
    pub models: ModelsConfig,
    pub routing: RoutingConfig,
    pub accounts: Vec<AccountConfig>,
    pub reasoning: ReasoningConfig,
    pub timeouts: TimeoutsConfig,
    pub compaction: CompactionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FileConfigV2 {
    #[serde(default)]
    pub server: Option<ServerConfig>,
    #[serde(default)]
    pub providers: Option<ProvidersConfig>,
    #[serde(default)]
    pub models: Option<ModelsConfig>,
    #[serde(default)]
    pub routing: Option<RoutingConfig>,
    #[serde(default)]
    pub accounts: Option<Vec<AccountConfig>>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub timeouts: Option<TimeoutsConfig>,
    #[serde(default)]
    pub compaction: Option<CompactionConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyFileConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub log_level: Option<String>,
    pub debug_mode: Option<bool>,
    pub z_ai_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub gemini_creds_path: Option<String>,
    pub models: Option<Vec<String>>,
    pub compaction_model: Option<String>,
    pub fallback_models: Option<HashMap<String, String>>,
    pub model_prefixes: Option<HashMap<String, String>>,
    pub model_overrides: Option<HashMap<String, String>>,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<ReasoningConfig>,
    pub z_ai_url: Option<String>,
    pub gemini_api_internal: Option<String>,
    pub gemini_api_public: Option<String>,
    pub openai_responses_url: Option<String>,
    pub request_timeout_connect: Option<u64>,
    pub request_timeout_read: Option<u64>,
    pub compaction_temperature: Option<f64>,
}

pub static CONFIG: Lazy<Config> = Lazy::new(Config::new);

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    pub fn new() -> Self {
        let mut cfg = Self::defaults();
        cfg.load_from_file();
        cfg.validate().expect("invalid configuration");
        cfg
    }

    fn defaults() -> Self {
        let home = dirs_home();
        let host = env::var("CODEX_PROXY_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let port = env::var("CODEX_PROXY_PORT")
            .map(|p| validate_port(&p).unwrap_or(8765))
            .unwrap_or(8765);
        let log_level = env::var("CODEX_PROXY_LOG_LEVEL")
            .unwrap_or_else(|_| "DEBUG".into())
            .to_uppercase();
        let debug_mode = env::var("CODEX_PROXY_DEBUG")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(true);

        let gemini_api_internal = env::var("CODEX_PROXY_GEMINI_API_INTERNAL")
            .unwrap_or_else(|_| "https://cloudcode-pa.googleapis.com".into());
        let gemini_api_public = env::var("CODEX_PROXY_GEMINI_API_PUBLIC")
            .unwrap_or_else(|_| "https://generativelanguage.googleapis.com".into());
        let z_ai_url = env::var("CODEX_PROXY_ZAI_URL")
            .unwrap_or_else(|_| "https://api.z.ai/api/coding/paas/v4/chat/completions".into());
        let openai_responses_url = env::var("CODEX_PROXY_OPENAI_RESPONSES_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1/responses".into());

        let served_models: Vec<String> = env::var("CODEX_PROXY_MODELS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let gemini_client_id = env::var("CODEX_PROXY_GEMINI_CLIENT_ID")
            .unwrap_or_else(|_| GEMINI_CLI_CLIENT_ID.into());
        let gemini_client_secret = env::var("CODEX_PROXY_GEMINI_CLIENT_SECRET")
            .unwrap_or_else(|_| GEMINI_CLI_CLIENT_SECRET.into());

        let mut accounts = Vec::new();
        if let Ok(key) = env::var("CODEX_PROXY_GEMINI_API_KEY")
            && !key.is_empty()
        {
            accounts.push(AccountConfig {
                id: "gemini-default".into(),
                provider: AccountProvider::Gemini,
                enabled: true,
                weight: 1,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }
        if let Ok(key) = env::var("CODEX_PROXY_ZAI_API_KEY")
            && !key.is_empty()
        {
            accounts.push(AccountConfig {
                id: "zai-default".into(),
                provider: AccountProvider::Zai,
                enabled: true,
                weight: 1,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }
        if let Ok(key) = env::var("CODEX_PROXY_OPENAI_API_KEY")
            && !key.is_empty()
        {
            accounts.push(AccountConfig {
                id: "openai-default".into(),
                provider: AccountProvider::OpenAi,
                enabled: true,
                weight: 1,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }
        if accounts.is_empty() {
            let gemini_creds_path = env::var("CODEX_PROXY_GEMINI_CREDS_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| home.join(".gemini/oauth_creds.json"));
            accounts.push(AccountConfig {
                id: "gemini-oauth".into(),
                provider: AccountProvider::Gemini,
                enabled: true,
                weight: 1,
                auth: AccountAuth::GeminiOAuth {
                    creds_path: Some(gemini_creds_path),
                    client_id: Some(gemini_client_id.clone()),
                    client_secret: Some(gemini_client_secret.clone()),
                },
            });
        }

        Self {
            config_path: home.join(".config/codex-proxy/config.json"),
            server: ServerConfig {
                host,
                port,
                log_level,
                debug_mode,
            },
            providers: ProvidersConfig {
                gemini: GeminiProviderConfig {
                    api_internal: validate_url(&gemini_api_internal, "Gemini internal").unwrap(),
                    api_public: validate_url(&gemini_api_public, "Gemini public").unwrap(),
                    default_client_id: gemini_client_id,
                    default_client_secret: gemini_client_secret,
                },
                zai: ZaiProviderConfig {
                    api_url: validate_url(&z_ai_url, "Z.AI URL").unwrap(),
                    allow_authorization_passthrough: false,
                },
                openai: OpenAiProviderConfig {
                    responses_url: validate_url(&openai_responses_url, "OpenAI responses URL")
                        .unwrap(),
                },
            },
            models: ModelsConfig {
                served: served_models,
                compaction_model: None,
                fallback_models: HashMap::new(),
            },
            routing: RoutingConfig {
                model_overrides: HashMap::new(),
                provider_prefixes: default_provider_prefixes(),
                sticky_routing: StickyRoutingConfig::default(),
                health: RoutingHealthConfig::default(),
            },
            accounts,
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 600,
            },
            compaction: CompactionConfig { temperature: 0.1 },
        }
    }

    fn load_from_file(&mut self) {
        if !self.config_path.exists() {
            return;
        }
        let content = match fs::read_to_string(&self.config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!(
                    "Failed to read config {}: {}",
                    self.config_path.display(),
                    e
                );
                return;
            }
        };
        let raw: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    "Failed to parse config {}: {}",
                    self.config_path.display(),
                    e
                );
                return;
            }
        };

        if looks_like_v2_config(&raw) {
            match serde_json::from_value::<FileConfigV2>(raw) {
                Ok(v2) => self.apply_v2(v2),
                Err(e) => warn!(
                    "Failed to parse v2 config {}: {}",
                    self.config_path.display(),
                    e
                ),
            }
        } else {
            match serde_json::from_str::<LegacyFileConfig>(&content) {
                Ok(legacy) => {
                    warn!(
                        "Loaded legacy flat config from {}. Migrate to the v2 structured schema.",
                        self.config_path.display()
                    );
                    self.apply_legacy(legacy);
                }
                Err(e) => warn!(
                    "Failed to parse legacy config {}: {}",
                    self.config_path.display(),
                    e
                ),
            }
        }

        info!("Loaded config from {}", self.config_path.display());
    }

    fn apply_v2(&mut self, file_cfg: FileConfigV2) {
        if let Some(server) = file_cfg.server {
            self.server = server;
        }
        if let Some(providers) = file_cfg.providers {
            self.providers = providers;
        }
        if let Some(models) = file_cfg.models {
            self.models = models;
        }
        if let Some(routing) = file_cfg.routing {
            self.routing = routing;
        }
        if let Some(accounts) = file_cfg.accounts {
            self.accounts = accounts;
        }
        if let Some(reasoning) = file_cfg.reasoning {
            self.reasoning = reasoning;
        }
        if let Some(timeouts) = file_cfg.timeouts {
            self.timeouts = timeouts;
        }
        if let Some(compaction) = file_cfg.compaction {
            self.compaction = compaction;
        }
    }

    fn apply_legacy(&mut self, legacy: LegacyFileConfig) {
        if let Some(host) = legacy.host {
            self.server.host = host;
        }
        if let Some(port) = legacy.port {
            self.server.port = port;
        }
        if let Some(log_level) = legacy.log_level {
            self.server.log_level = log_level.to_uppercase();
        }
        if let Some(debug_mode) = legacy.debug_mode {
            self.server.debug_mode = debug_mode;
        }
        if let Some(models) = legacy.models {
            self.models.served = models;
        }
        if let Some(compaction_model) = legacy.compaction_model
            && !compaction_model.is_empty()
        {
            self.models.compaction_model = Some(compaction_model);
        }
        if let Some(fallback_models) = legacy.fallback_models {
            self.models.fallback_models = fallback_models;
        }
        if let Some(model_prefixes) = legacy.model_prefixes {
            self.routing.provider_prefixes = model_prefixes
                .into_iter()
                .filter_map(|(prefix, provider)| {
                    let prefix = validate_model_prefix(&prefix).ok()?;
                    let provider = provider.parse::<AccountProvider>().ok()?;
                    Some((prefix, provider))
                })
                .collect();
        }
        if let Some(model_overrides) = legacy.model_overrides {
            self.routing.model_overrides = model_overrides;
        }
        if let Some(reasoning_effort) = legacy.reasoning_effort {
            self.reasoning.default_effort = reasoning_effort;
        }
        if let Some(reasoning) = legacy.reasoning {
            self.reasoning = reasoning;
        }
        if let Some(request_timeout_connect) = legacy.request_timeout_connect {
            self.timeouts.connect_seconds = request_timeout_connect;
        }
        if let Some(request_timeout_read) = legacy.request_timeout_read {
            self.timeouts.read_seconds = request_timeout_read;
        }
        if let Some(compaction_temperature) = legacy.compaction_temperature {
            self.compaction.temperature = compaction_temperature;
        }
        if let Some(url) = legacy.z_ai_url
            && let Ok(valid) = validate_url(&url, "Z.AI URL")
        {
            self.providers.zai.api_url = valid;
        }
        if let Some(url) = legacy.gemini_api_internal
            && let Ok(valid) = validate_url(&url, "Gemini internal")
        {
            self.providers.gemini.api_internal = valid;
        }
        if let Some(url) = legacy.gemini_api_public
            && let Ok(valid) = validate_url(&url, "Gemini public")
        {
            self.providers.gemini.api_public = valid;
        }
        if let Some(url) = legacy.openai_responses_url
            && let Ok(valid) = validate_url(&url, "OpenAI responses URL")
        {
            self.providers.openai.responses_url = valid;
        }
        if let Some(client_id) = legacy.client_id {
            self.providers.gemini.default_client_id = client_id;
        }
        if let Some(client_secret) = legacy.client_secret {
            self.providers.gemini.default_client_secret = client_secret;
        }

        let mut migrated_accounts = Vec::new();
        if let Some(key) = legacy.gemini_api_key.filter(|k| !k.is_empty()) {
            migrated_accounts.push(AccountConfig {
                id: "gemini-default".into(),
                provider: AccountProvider::Gemini,
                enabled: true,
                weight: 1,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        } else {
            migrated_accounts.push(AccountConfig {
                id: "gemini-oauth".into(),
                provider: AccountProvider::Gemini,
                enabled: true,
                weight: 1,
                auth: AccountAuth::GeminiOAuth {
                    creds_path: legacy.gemini_creds_path.map(PathBuf::from),
                    client_id: Some(self.providers.gemini.default_client_id.clone()),
                    client_secret: Some(self.providers.gemini.default_client_secret.clone()),
                },
            });
        }
        if let Some(key) = legacy.z_ai_api_key.filter(|k| !k.is_empty()) {
            migrated_accounts.push(AccountConfig {
                id: "zai-default".into(),
                provider: AccountProvider::Zai,
                enabled: true,
                weight: 1,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }
        if let Some(key) = legacy.openai_api_key.filter(|k| !k.is_empty()) {
            migrated_accounts.push(AccountConfig {
                id: "openai-default".into(),
                provider: AccountProvider::OpenAi,
                enabled: true,
                weight: 1,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }
        self.accounts = migrated_accounts;
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_url(&self.providers.gemini.api_internal, "Gemini internal")?;
        validate_url(&self.providers.gemini.api_public, "Gemini public")?;
        validate_url(&self.providers.zai.api_url, "Z.AI URL")?;
        validate_url(&self.providers.openai.responses_url, "OpenAI responses URL")?;

        if self.server.port == 0 {
            return Err(ConfigError::InvalidPort("port must be 1-65535".into()));
        }
        if !self
            .reasoning
            .effort_levels
            .contains_key(&self.reasoning.default_effort)
        {
            return Err(ConfigError::InvalidValue(format!(
                "reasoning.default_effort '{}' is not defined in reasoning.effort_levels",
                self.reasoning.default_effort
            )));
        }

        let mut seen_ids = HashSet::new();
        let enabled_accounts: Vec<&AccountConfig> =
            self.accounts.iter().filter(|a| a.enabled).collect();
        if enabled_accounts.is_empty() {
            return Err(ConfigError::InvalidValue(
                "accounts must contain at least one enabled account".into(),
            ));
        }
        for account in &self.accounts {
            if !seen_ids.insert(account.id.clone()) {
                return Err(ConfigError::InvalidValue(format!(
                    "duplicate account id: {}",
                    account.id
                )));
            }
            match (&account.provider, &account.auth) {
                (AccountProvider::Gemini, AccountAuth::ApiKey { api_key }) => {
                    if api_key.is_empty() {
                        return Err(ConfigError::InvalidValue(format!(
                            "account '{}' has empty api_key auth",
                            account.id
                        )));
                    }
                }
                (
                    AccountProvider::Gemini,
                    AccountAuth::GeminiOAuth {
                        creds_path,
                        client_id,
                        client_secret,
                    },
                ) => {
                    if creds_path.is_none() && client_id.is_none() && client_secret.is_none() {
                        return Err(ConfigError::InvalidValue(format!(
                            "account '{}' needs Gemini OAuth credentials or defaults",
                            account.id
                        )));
                    }
                }
                (AccountProvider::Zai, AccountAuth::ApiKey { api_key })
                | (AccountProvider::OpenAi, AccountAuth::ApiKey { api_key }) => {
                    if api_key.is_empty() {
                        return Err(ConfigError::InvalidValue(format!(
                            "account '{}' has empty api_key auth",
                            account.id
                        )));
                    }
                }
                (provider, auth) => {
                    return Err(ConfigError::InvalidValue(format!(
                        "account '{}' has invalid auth {:?} for provider {}",
                        account.id, auth, provider
                    )));
                }
            }
        }

        for prefix in self.routing.provider_prefixes.keys() {
            validate_model_prefix(prefix)?;
        }
        for target in self.routing.model_overrides.values() {
            if self.provider_for_model(target).is_none() {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_overrides target '{}' does not resolve to a configured provider prefix",
                    target
                )));
            }
        }
        if let Some(model) = self.models.compaction_model.as_deref()
            && self.provider_for_model(model).is_none()
        {
            return Err(ConfigError::InvalidValue(format!(
                "models.compaction_model '{}' does not resolve to a configured provider prefix",
                model
            )));
        }
        Ok(())
    }

    pub fn resolve_model(&self, requested_model: &str) -> String {
        self.routing
            .model_overrides
            .get(requested_model)
            .cloned()
            .unwrap_or_else(|| requested_model.to_string())
    }

    pub fn provider_for_model(&self, model: &str) -> Option<AccountProvider> {
        self.routing
            .provider_prefixes
            .iter()
            .filter(|(prefix, _)| model.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, provider)| *provider)
    }

    pub fn compaction_model(&self) -> Option<&str> {
        self.models
            .compaction_model
            .as_deref()
            .or_else(|| self.models.served.first().map(|model| model.as_str()))
    }

    pub fn is_served_model_allowed(&self, model: &str) -> bool {
        self.models.served.is_empty() || self.models.served.iter().any(|m| m == model)
    }
}

fn looks_like_v2_config(value: &serde_json::Value) -> bool {
    value
        .as_object()
        .map(|obj| {
            obj.contains_key("server")
                || obj.contains_key("providers")
                || obj.contains_key("models")
                || obj.contains_key("routing")
                || obj.contains_key("accounts")
                || obj.contains_key("timeouts")
                || obj.contains_key("compaction")
        })
        .unwrap_or(false)
}

fn default_true() -> bool {
    true
}

fn default_weight() -> u32 {
    1
}

fn default_failure_threshold() -> u32 {
    3
}

fn default_cooldown_seconds() -> u64 {
    300
}

fn default_auth_failure_immediate_unhealthy() -> bool {
    true
}

fn default_provider_prefixes() -> HashMap<String, AccountProvider> {
    HashMap::from([
        ("gemini".into(), AccountProvider::Gemini),
        ("glm".into(), AccountProvider::Zai),
        ("zai".into(), AccountProvider::Zai),
        ("gpt".into(), AccountProvider::OpenAi),
        ("o".into(), AccountProvider::OpenAi),
        ("openai".into(), AccountProvider::OpenAi),
    ])
}

fn dirs_home() -> PathBuf {
    resolve_home_dir(
        env::var_os("HOME").map(PathBuf::from),
        env::var_os("USERPROFILE").map(PathBuf::from),
        env::var_os("HOMEDRIVE"),
        env::var_os("HOMEPATH"),
        cfg!(windows),
    )
}

fn resolve_home_dir(
    home: Option<PathBuf>,
    userprofile: Option<PathBuf>,
    homedrive: Option<OsString>,
    homepath: Option<OsString>,
    prefer_windows_env: bool,
) -> PathBuf {
    let home = home.filter(|p| !p.as_os_str().is_empty());
    let userprofile = userprofile.filter(|p| !p.as_os_str().is_empty());
    let windows_home = join_windows_home(homedrive, homepath);

    if prefer_windows_env {
        if let Some(path) = userprofile.as_ref() {
            return path.clone();
        }
        if let Some(path) = windows_home.as_ref() {
            return path.clone();
        }
    }

    if let Some(path) = home {
        return path;
    }

    if !prefer_windows_env {
        if let Some(path) = userprofile {
            return path;
        }
        if let Some(path) = windows_home {
            return path;
        }
    }

    PathBuf::from("/root")
}

fn join_windows_home(homedrive: Option<OsString>, homepath: Option<OsString>) -> Option<PathBuf> {
    let mut homedrive = homedrive.filter(|value| !value.is_empty())?;
    let homepath = homepath.filter(|value| !value.is_empty())?;
    homedrive.push(homepath);
    Some(PathBuf::from(homedrive))
}

#[cfg(test)]
mod tests {
    use super::resolve_home_dir;
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn resolve_home_dir_prefers_userprofile_on_windows() {
        let home = resolve_home_dir(
            Some(PathBuf::from("/root")),
            Some(PathBuf::from(r"C:\Users\woodrow")),
            None,
            None,
            true,
        );

        assert_eq!(home, PathBuf::from(r"C:\Users\woodrow"));
    }

    #[test]
    fn resolve_home_dir_falls_back_to_home_when_windows_vars_missing() {
        let home = resolve_home_dir(Some(PathBuf::from("/custom/home")), None, None, None, true);

        assert_eq!(home, PathBuf::from("/custom/home"));
    }

    #[test]
    fn resolve_home_dir_combines_home_drive_and_path_on_windows() {
        let home = resolve_home_dir(
            None,
            None,
            Some(OsString::from("C:")),
            Some(OsString::from(r"\Users\woodrow")),
            true,
        );

        assert_eq!(home, PathBuf::from(r"C:\Users\woodrow"));
    }

    #[test]
    fn resolve_home_dir_prefers_home_on_non_windows() {
        let home = resolve_home_dir(
            Some(PathBuf::from("/custom/home")),
            Some(PathBuf::from(r"C:\Users\woodrow")),
            None,
            None,
            false,
        );

        assert_eq!(home, PathBuf::from("/custom/home"));
    }
}
