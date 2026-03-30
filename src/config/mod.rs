use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use crate::account_pool::AccountAuth;
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EffortLevel {
    pub budget: u64,
    pub level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningConfig {
    pub effort_levels: HashMap<String, EffortLevel>,
    #[serde(default)]
    pub default_effort: Option<String>,
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
            default_effort: Some("medium".into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EffectiveReasoningConfig {
    pub budget: u64,
    pub level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RouteReasoningConfig {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub budget: Option<u64>,
    #[serde(default)]
    pub level: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub log_level: String,
    pub debug_mode: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    Gemini,
    Zai,
    OpenAi,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ProviderType::Gemini => "gemini",
            ProviderType::Zai => "zai",
            ProviderType::OpenAi => "openai",
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiProviderConfig {
    pub api_internal: String,
    pub api_public: String,
    pub default_client_id: String,
    pub default_client_secret: String,
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiProviderConfig {
    pub api_url: String,
    #[serde(default)]
    pub endpoints: HashMap<String, String>,
    #[serde(default)]
    pub allow_authorization_passthrough: bool,
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiProviderConfig {
    pub api_url: String,
    #[serde(default)]
    pub endpoints: HashMap<String, String>,
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderConfig {
    Gemini {
        api_internal: String,
        api_public: String,
        default_client_id: String,
        default_client_secret: String,
        #[serde(default)]
        models: Vec<String>,
    },
    Zai {
        api_url: String,
        #[serde(default)]
        endpoints: HashMap<String, String>,
        #[serde(default)]
        allow_authorization_passthrough: bool,
        #[serde(default)]
        models: Vec<String>,
    },
    OpenAi {
        api_url: String,
        #[serde(default)]
        endpoints: HashMap<String, String>,
        #[serde(default)]
        models: Vec<String>,
    },
}

impl ProviderConfig {
    pub fn provider_type(&self) -> ProviderType {
        match self {
            ProviderConfig::Gemini { .. } => ProviderType::Gemini,
            ProviderConfig::Zai { .. } => ProviderType::Zai,
            ProviderConfig::OpenAi { .. } => ProviderType::OpenAi,
        }
    }

    pub fn models(&self) -> &Vec<String> {
        match self {
            ProviderConfig::Gemini { models, .. }
            | ProviderConfig::Zai { models, .. }
            | ProviderConfig::OpenAi { models, .. } => models,
        }
    }

    pub fn as_gemini(&self) -> Option<GeminiProviderConfig> {
        match self {
            ProviderConfig::Gemini {
                api_internal,
                api_public,
                default_client_id,
                default_client_secret,
                models,
            } => Some(GeminiProviderConfig {
                api_internal: api_internal.clone(),
                api_public: api_public.clone(),
                default_client_id: default_client_id.clone(),
                default_client_secret: default_client_secret.clone(),
                models: models.clone(),
            }),
            _ => None,
        }
    }

    pub fn as_zai(&self) -> Option<ZaiProviderConfig> {
        match self {
            ProviderConfig::Zai {
                api_url,
                endpoints,
                allow_authorization_passthrough,
                models,
            } => Some(ZaiProviderConfig {
                api_url: api_url.clone(),
                endpoints: endpoints.clone(),
                allow_authorization_passthrough: *allow_authorization_passthrough,
                models: models.clone(),
            }),
            _ => None,
        }
    }

    pub fn endpoint_url(&self, provider_name: &str, endpoint: Option<&str>) -> Result<String, ConfigError> {
        match self {
            ProviderConfig::Gemini { api_public, .. } => match endpoint {
                Some(name) => Err(ConfigError::InvalidValue(format!(
                    "Provider '{}' of type gemini does not support named endpoints in route targets: {}",
                    provider_name, name
                ))),
                None => Ok(api_public.clone()),
            },
            ProviderConfig::OpenAi {
                api_url,
                endpoints,
                ..
            }
            | ProviderConfig::Zai {
                api_url,
                endpoints,
                ..
            } => match endpoint {
                Some(name) => endpoints.get(name).cloned().ok_or_else(|| {
                    ConfigError::InvalidValue(format!(
                        "Unknown endpoint '{}' referenced by provider '{}'",
                        name, provider_name
                    ))
                }),
                None => Ok(api_url.clone()),
            },
        }
    }

    pub fn validate(&self, provider_name: &str) -> Result<(), ConfigError> {
        match self {
            ProviderConfig::Gemini {
                api_internal,
                api_public,
                ..
            } => {
                validate_url(api_internal, &format!("Provider '{}' gemini api_internal", provider_name))?;
                validate_url(api_public, &format!("Provider '{}' gemini api_public", provider_name))?;
            }
            ProviderConfig::Zai {
                api_url,
                endpoints,
                ..
            }
            | ProviderConfig::OpenAi {
                api_url,
                endpoints,
                ..
            } => {
                validate_url(api_url, &format!("Provider '{}' api_url", provider_name))?;
                for (name, url) in endpoints {
                    validate_url(url, &format!("Provider '{}' endpoint '{}'", provider_name, name))?;
                }
            }
        }
        Ok(())
    }
}

pub type ProvidersConfig = HashMap<String, ProviderConfig>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    #[serde(default)]
    pub served: Vec<String>,
    #[serde(default)]
    pub fallback_models: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteTargetConfig {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub reasoning: Option<RouteReasoningConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct StickyRoutingConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    #[serde(default)]
    pub preferred_models: HashMap<String, Vec<RouteTargetConfig>>,
    #[serde(default)]
    pub sticky_routing: StickyRoutingConfig,
    #[serde(default)]
    pub health: RoutingHealthConfig,
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
    pub preferred_targets: Vec<RouteTargetConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountConfig {
    pub id: String,
    pub provider: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_weight")]
    pub weight: u32,
    #[serde(default)]
    pub models: Option<Vec<String>>,
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
struct FileConfig {
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
        let openai_api_url = env::var("CODEX_PROXY_OPENAI_API_URL")
            .or_else(|_| env::var("CODEX_PROXY_OPENAI_RESPONSES_URL"))
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
                provider: "gemini".into(),
                enabled: true,
                weight: 1,
                models: None,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }
        if let Ok(key) = env::var("CODEX_PROXY_ZAI_API_KEY")
            && !key.is_empty()
        {
            accounts.push(AccountConfig {
                id: "zai-default".into(),
                provider: "zai".into(),
                enabled: true,
                weight: 1,
                models: None,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }
        if let Ok(key) = env::var("CODEX_PROXY_OPENAI_API_KEY")
            && !key.is_empty()
        {
            accounts.push(AccountConfig {
                id: "openai-default".into(),
                provider: "openai".into(),
                enabled: true,
                weight: 1,
                models: None,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }
        if accounts.is_empty() {
            let gemini_creds_path = env::var("CODEX_PROXY_GEMINI_CREDS_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| home.join(".gemini/oauth_creds.json"));
            accounts.push(AccountConfig {
                id: "gemini-oauth".into(),
                provider: "gemini".into(),
                enabled: true,
                weight: 1,
                models: None,
                auth: AccountAuth::GeminiOAuth {
                    creds_path: Some(gemini_creds_path),
                    client_id: Some(gemini_client_id.clone()),
                    client_secret: Some(gemini_client_secret.clone()),
                },
            });
        }

        let mut providers = HashMap::new();
        providers.insert(
            "gemini".into(),
            ProviderConfig::Gemini {
                api_internal: validate_url(&gemini_api_internal, "Gemini internal").unwrap(),
                api_public: validate_url(&gemini_api_public, "Gemini public").unwrap(),
                default_client_id: gemini_client_id,
                default_client_secret: gemini_client_secret,
                models: Vec::new(),
            },
        );
        providers.insert(
            "zai".into(),
            ProviderConfig::Zai {
                api_url: validate_url(&z_ai_url, "Z.AI URL").unwrap(),
                endpoints: HashMap::new(),
                allow_authorization_passthrough: false,
                models: Vec::new(),
            },
        );
        providers.insert(
            "openai".into(),
            ProviderConfig::OpenAi {
                api_url: validate_url(&openai_api_url, "OpenAI API URL").unwrap(),
                endpoints: HashMap::new(),
                models: Vec::new(),
            },
        );

        Self {
            config_path: default_config_search_paths(&home)[0].clone(),
            server: ServerConfig {
                host,
                port,
                log_level,
                debug_mode,
            },
            providers,
            models: ModelsConfig {
                served: served_models,
                fallback_models: HashMap::new(),
            },
            routing: RoutingConfig {
                model_overrides: HashMap::new(),
                preferred_models: HashMap::new(),
                sticky_routing: StickyRoutingConfig::default(),
                health: RoutingHealthConfig::default(),
            },
            accounts,
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 600,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: Vec::new(),
            },
        }
    }

    fn load_from_file(&mut self) {
        for path in default_config_search_paths(&dirs_home()) {
            if !path.exists() {
                continue;
            }
            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    warn!("Failed to read config {}: {}", path.display(), e);
                    continue;
                }
            };
            let file_cfg: FileConfig = match serde_json::from_str(&content) {
                Ok(cfg) => cfg,
                Err(e) => {
                    warn!("Failed to parse config {}: {}", path.display(), e);
                    continue;
                }
            };

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

            self.config_path = path.clone();
            info!("Loaded config from {}", self.config_path.display());
            return;
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.providers.is_empty() {
            return Err(ConfigError::InvalidValue(
                "providers must contain at least one provider entry".into(),
            ));
        }
        for (provider_name, provider) in &self.providers {
            if provider_name.trim().is_empty() {
                return Err(ConfigError::InvalidValue(
                    "providers contains an empty provider name".into(),
                ));
            }
            provider.validate(provider_name)?;
        }

        if self.server.port == 0 {
            return Err(ConfigError::InvalidPort("port must be 1-65535".into()));
        }
        if let Some(default_effort) = &self.reasoning.default_effort
            && !self.reasoning.effort_levels.contains_key(default_effort)
        {
            return Err(ConfigError::InvalidValue(format!(
                "reasoning.default_effort '{}' is not defined in reasoning.effort_levels",
                default_effort
            )));
        }

        let mut seen_ids = HashSet::new();
        let enabled_accounts: Vec<&AccountConfig> = self.accounts.iter().filter(|a| a.enabled).collect();
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

            let provider_type = self.provider_type(&account.provider)?;
            match (provider_type, &account.auth) {
                (ProviderType::Gemini, AccountAuth::ApiKey { api_key }) => {
                    if api_key.is_empty() {
                        return Err(ConfigError::InvalidValue(format!(
                            "account '{}' has empty api_key auth",
                            account.id
                        )));
                    }
                }
                (
                    ProviderType::Gemini,
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
                (ProviderType::Zai | ProviderType::OpenAi, AccountAuth::ApiKey { api_key }) => {
                    if api_key.is_empty() {
                        return Err(ConfigError::InvalidValue(format!(
                            "account '{}' has empty api_key auth",
                            account.id
                        )));
                    }
                }
                (provider_type, auth) => {
                    return Err(ConfigError::InvalidValue(format!(
                        "account '{}' has invalid auth {:?} for provider '{}' (type {})",
                        account.id, auth, account.provider, provider_type
                    )));
                }
            }

            if let Some(models) = &account.models {
                if models.is_empty() {
                    return Err(ConfigError::InvalidValue(format!(
                        "account '{}' models restriction must not be empty when present",
                        account.id
                    )));
                }
                if let Some(provider_models) = self.provider_catalog(&account.provider) {
                    for model in models {
                        if !provider_models.contains(model) {
                            return Err(ConfigError::InvalidValue(format!(
                                "account '{}' references model '{}' not present in provider '{}' catalog",
                                account.id, model, account.provider
                            )));
                        }
                    }
                }
            }
        }

        for (logical_model, targets) in &self.routing.preferred_models {
            if targets.is_empty() {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.preferred_models['{}'] must not be empty",
                    logical_model
                )));
            }
            for target in targets {
                self.validate_route_target(
                    target,
                    &format!("routing.preferred_models['{}']", logical_model),
                )?;
            }
            if !self.has_compatible_enabled_account(targets) {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.preferred_models['{}'] has no compatible enabled account",
                    logical_model
                )));
            }
        }

        if self.compaction.preferred_targets.is_empty() {
            return Err(ConfigError::InvalidValue(
                "compaction.preferred_targets must not be empty".into(),
            ));
        }
        for target in &self.compaction.preferred_targets {
            self.validate_route_target(target, "compaction.preferred_targets")?;
        }
        if !self.has_compatible_enabled_account(&self.compaction.preferred_targets) {
            return Err(ConfigError::InvalidValue(
                "compaction.preferred_targets has no compatible enabled account".into(),
            ));
        }

        for mapped_model in self.routing.model_overrides.values() {
            if !self.routing.preferred_models.contains_key(mapped_model) {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_overrides target '{}' is not defined in routing.preferred_models",
                    mapped_model
                )));
            }
        }

        if !self.models.served.is_empty() {
            for served_model in &self.models.served {
                let logical_model = self.resolve_logical_model(served_model);
                if !self.routing.preferred_models.contains_key(&logical_model) {
                    return Err(ConfigError::InvalidValue(format!(
                        "served model '{}' resolves to logical model '{}' but no routing.preferred_models entry exists",
                        served_model, logical_model
                    )));
                }
            }
        }

        Ok(())
    }

    pub fn resolve_logical_model(&self, requested_model: &str) -> String {
        self.routing
            .model_overrides
            .get(requested_model)
            .cloned()
            .unwrap_or_else(|| requested_model.to_string())
    }

    pub fn preferred_targets_for_model(&self, requested_model: &str) -> Option<&[RouteTargetConfig]> {
        let logical_model = self.resolve_logical_model(requested_model);
        self.routing
            .preferred_models
            .get(&logical_model)
            .map(Vec::as_slice)
    }

    pub fn compaction_targets(&self) -> &[RouteTargetConfig] {
        &self.compaction.preferred_targets
    }

    pub fn is_served_model_allowed(&self, model: &str) -> bool {
        self.models.served.is_empty() || self.models.served.iter().any(|m| m == model)
    }

    pub fn resolve_reasoning(
        &self,
        reasoning: Option<&RouteReasoningConfig>,
    ) -> Result<Option<EffectiveReasoningConfig>, ConfigError> {
        let Some(reasoning) = reasoning else {
            return Ok(self.reasoning.default_effort.as_ref().map(|preset| {
                let cfg = self
                    .reasoning
                    .effort_levels
                    .get(preset)
                    .expect("validated default reasoning preset must exist");
                EffectiveReasoningConfig {
                    budget: cfg.budget,
                    level: cfg.level.clone(),
                    preset: Some(preset.clone()),
                }
            }));
        };

        if let Some(preset) = &reasoning.effort {
            let cfg = self.reasoning.effort_levels.get(preset).ok_or_else(|| {
                ConfigError::InvalidValue(format!(
                    "reasoning preset '{}' is not defined in reasoning.effort_levels",
                    preset
                ))
            })?;
            let budget = reasoning.budget.unwrap_or(cfg.budget);
            let level = reasoning.level.clone().unwrap_or_else(|| cfg.level.clone());
            return Ok(Some(EffectiveReasoningConfig {
                budget,
                level,
                preset: Some(preset.clone()),
            }));
        }

        if reasoning.budget.is_none() && reasoning.level.is_none() {
            return Ok(self.reasoning.default_effort.as_ref().map(|preset| {
                let cfg = self
                    .reasoning
                    .effort_levels
                    .get(preset)
                    .expect("validated default reasoning preset must exist");
                EffectiveReasoningConfig {
                    budget: cfg.budget,
                    level: cfg.level.clone(),
                    preset: Some(preset.clone()),
                }
            }));
        }

        Ok(Some(EffectiveReasoningConfig {
            budget: reasoning.budget.unwrap_or(0),
            level: reasoning.level.clone().unwrap_or_else(|| "LOW".into()),
            preset: None,
        }))
    }

    pub fn provider_type(&self, provider: &str) -> Result<ProviderType, ConfigError> {
        self.providers
            .get(provider)
            .map(ProviderConfig::provider_type)
            .ok_or_else(|| ConfigError::InvalidValue(format!("Unknown provider '{}'", provider)))
    }

    pub fn provider_catalog(&self, provider: &str) -> Option<&Vec<String>> {
        let models = self.providers.get(provider)?.models();
        (!models.is_empty()).then_some(models)
    }

    pub fn endpoint_url(&self, provider: &str, endpoint: Option<&str>) -> Result<String, ConfigError> {
        self.providers
            .get(provider)
            .ok_or_else(|| ConfigError::InvalidValue(format!("Unknown provider '{}'", provider)))?
            .endpoint_url(provider, endpoint)
    }

    pub fn gemini_provider_config(&self, provider: &str) -> Result<GeminiProviderConfig, ConfigError> {
        self.providers
            .get(provider)
            .and_then(ProviderConfig::as_gemini)
            .ok_or_else(|| {
                ConfigError::InvalidValue(format!(
                    "Provider '{}' is not configured as type gemini",
                    provider
                ))
            })
    }

    pub fn zai_provider_config(&self, provider: &str) -> Result<ZaiProviderConfig, ConfigError> {
        self.providers
            .get(provider)
            .and_then(ProviderConfig::as_zai)
            .ok_or_else(|| {
                ConfigError::InvalidValue(format!(
                    "Provider '{}' is not configured as type zai",
                    provider
                ))
            })
    }

    fn validate_route_target(&self, target: &RouteTargetConfig, scope: &str) -> Result<(), ConfigError> {
        if target.provider.trim().is_empty() {
            return Err(ConfigError::InvalidValue(format!(
                "{scope} contains an empty provider name"
            )));
        }
        if target.model.trim().is_empty() {
            return Err(ConfigError::InvalidValue(format!(
                "{scope} contains an empty target model"
            )));
        }
        self.provider_type(&target.provider)?;
        if let Some(provider_models) = self.provider_catalog(&target.provider)
            && !provider_models.contains(&target.model)
        {
            return Err(ConfigError::InvalidValue(format!(
                "{scope} target '{}:{}' is not present in provider catalog",
                target.provider, target.model
            )));
        }
        self.resolve_reasoning(target.reasoning.as_ref())?;
        self.endpoint_url(&target.provider, target.endpoint.as_deref())?;
        Ok(())
    }

    fn has_compatible_enabled_account(&self, targets: &[RouteTargetConfig]) -> bool {
        targets.iter().any(|target| {
            self.accounts.iter().any(|account| {
                if !account.enabled || account.provider != target.provider {
                    return false;
                }
                match &account.models {
                    Some(models) => models.contains(&target.model),
                    None => true,
                }
            })
        })
    }
}

fn default_config_search_paths(home: &Path) -> Vec<PathBuf> {
    vec![
        PathBuf::from("config/config.json.local"),
        home.join(".config/codex-proxy/config.json"),
        PathBuf::from("config/config.json"),
    ]
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

fn dirs_home() -> PathBuf {
    env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> Config {
        Config {
            config_path: PathBuf::from("/tmp/config.json"),
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 8765,
                log_level: "INFO".into(),
                debug_mode: false,
            },
            providers: HashMap::from([
                (
                    "gemini".into(),
                    ProviderConfig::Gemini {
                        api_internal: "https://internal.example.com".into(),
                        api_public: "https://public.example.com".into(),
                        default_client_id: "id".into(),
                        default_client_secret: "secret".into(),
                        models: vec!["gemini-2.5-pro".into()],
                    },
                ),
                (
                    "zai".into(),
                    ProviderConfig::Zai {
                        api_url: "https://z.ai/chat".into(),
                        endpoints: HashMap::from([("fast".into(), "https://z.ai/fast".into())]),
                        allow_authorization_passthrough: false,
                        models: vec!["glm-4.6".into()],
                    },
                ),
                (
                    "openai".into(),
                    ProviderConfig::OpenAi {
                        api_url: "https://api.openai.com/v1/responses".into(),
                        endpoints: HashMap::from([(
                            "priority".into(),
                            "https://priority.openai.com/v1/responses".into(),
                        )]),
                        models: vec!["gpt-4.1".into(), "gpt-4.1-mini".into()],
                    },
                ),
                (
                    "tabcode".into(),
                    ProviderConfig::OpenAi {
                        api_url: "https://tabcode.example/v1/responses".into(),
                        endpoints: HashMap::new(),
                        models: vec!["gpt-4.1".into()],
                    },
                ),
            ]),
            models: ModelsConfig {
                served: vec!["claude-sonnet-4-6".into()],
                fallback_models: HashMap::new(),
            },
            routing: RoutingConfig {
                model_overrides: HashMap::new(),
                preferred_models: HashMap::from([(
                    "claude-sonnet-4-6".into(),
                    vec![RouteTargetConfig {
                        provider: "tabcode".into(),
                        model: "gpt-4.1".into(),
                        endpoint: None,
                        reasoning: Some(RouteReasoningConfig {
                            effort: Some("medium".into()),
                            budget: None,
                            level: None,
                        }),
                    }],
                )]),
                sticky_routing: StickyRoutingConfig { enabled: true },
                health: RoutingHealthConfig::default(),
            },
            accounts: vec![AccountConfig {
                id: "tabcode-a".into(),
                provider: "tabcode".into(),
                enabled: true,
                weight: 1,
                models: Some(vec!["gpt-4.1".into()]),
                auth: AccountAuth::ApiKey {
                    api_key: "sk-test".into(),
                },
            }],
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 30,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: vec![RouteTargetConfig {
                    provider: "tabcode".into(),
                    model: "gpt-4.1".into(),
                    endpoint: None,
                    reasoning: Some(RouteReasoningConfig {
                        effort: Some("none".into()),
                        budget: None,
                        level: None,
                    }),
                }],
            },
        }
    }

    #[test]
    fn validates_capability_routing_config() {
        let cfg = base_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_account_model_outside_provider_catalog() {
        let mut cfg = base_config();
        cfg.accounts[0].models = Some(vec!["unknown-model".into()]);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("not present in provider 'tabcode' catalog"));
    }

    #[test]
    fn resolves_effective_reasoning_from_preset_override() {
        let cfg = base_config();
        let target = &cfg.routing.preferred_models["claude-sonnet-4-6"][0];
        let reasoning = cfg
            .resolve_reasoning(target.reasoning.as_ref())
            .unwrap()
            .unwrap();
        assert_eq!(reasoning.budget, 16384);
        assert_eq!(reasoning.level, "MEDIUM");
        assert_eq!(reasoning.preset.as_deref(), Some("medium"));
    }

    #[test]
    fn resolves_provider_type_from_dynamic_provider_name() {
        let cfg = base_config();
        assert_eq!(cfg.provider_type("tabcode").unwrap(), ProviderType::OpenAi);
    }
}
