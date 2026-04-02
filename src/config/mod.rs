use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    Gemini,
    Zai,
    #[serde(rename = "openrouter", alias = "open_router")]
    OpenRouter,
    #[serde(rename = "openai", alias = "open_ai")]
    OpenAi,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ProviderType::Gemini => "gemini",
            ProviderType::Zai => "zai",
            ProviderType::OpenRouter => "openrouter",
            ProviderType::OpenAi => "openai",
        })
    }
}

impl ProviderType {
    pub fn is_openai_compatible(self) -> bool {
        matches!(self, ProviderType::OpenAi | ProviderType::OpenRouter)
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
    pub models_url: Option<String>,
    #[serde(default)]
    pub endpoints: HashMap<String, String>,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub max_tokens_cap: Option<u64>,
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
    #[serde(rename = "openai", alias = "open_ai")]
    OpenAi {
        api_url: String,
        #[serde(default)]
        models_url: Option<String>,
        #[serde(default)]
        endpoints: HashMap<String, String>,
        #[serde(default)]
        models: Vec<String>,
        #[serde(default)]
        max_tokens_cap: Option<u64>,
    },
    #[serde(rename = "openrouter", alias = "open_router")]
    OpenRouter {
        api_url: String,
        #[serde(default)]
        models_url: Option<String>,
        #[serde(default)]
        endpoints: HashMap<String, String>,
        #[serde(default)]
        models: Vec<String>,
        #[serde(default)]
        max_tokens_cap: Option<u64>,
    },
}

impl ProviderConfig {
    pub fn provider_type(&self) -> ProviderType {
        match self {
            ProviderConfig::Gemini { .. } => ProviderType::Gemini,
            ProviderConfig::Zai { .. } => ProviderType::Zai,
            ProviderConfig::OpenAi { .. } => ProviderType::OpenAi,
            ProviderConfig::OpenRouter { .. } => ProviderType::OpenRouter,
        }
    }

    pub fn models(&self) -> &Vec<String> {
        match self {
            ProviderConfig::Gemini { models, .. }
            | ProviderConfig::Zai { models, .. }
            | ProviderConfig::OpenAi { models, .. }
            | ProviderConfig::OpenRouter { models, .. } => models,
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

    pub fn as_openai(&self) -> Option<OpenAiProviderConfig> {
        match self {
            ProviderConfig::OpenAi {
                api_url,
                models_url,
                endpoints,
                models,
                max_tokens_cap,
            } => Some(OpenAiProviderConfig {
                api_url: api_url.clone(),
                models_url: models_url.clone(),
                endpoints: endpoints.clone(),
                models: models.clone(),
                max_tokens_cap: *max_tokens_cap,
            }),
            ProviderConfig::OpenRouter {
                api_url,
                models_url,
                endpoints,
                models,
                max_tokens_cap,
            } => Some(OpenAiProviderConfig {
                api_url: api_url.clone(),
                models_url: models_url.clone(),
                endpoints: endpoints.clone(),
                models: models.clone(),
                max_tokens_cap: *max_tokens_cap,
            }),
            _ => None,
        }
    }

    pub fn endpoint_url(
        &self,
        provider_name: &str,
        endpoint: Option<&str>,
    ) -> Result<String, ConfigError> {
        match self {
            ProviderConfig::Gemini { api_public, .. } => match endpoint {
                Some(name) => Err(ConfigError::InvalidValue(format!(
                    "Provider '{}' of type gemini does not support named endpoints in route targets: {}",
                    provider_name, name
                ))),
                None => Ok(api_public.clone()),
            },
            ProviderConfig::OpenAi {
                api_url, endpoints, ..
            }
            | ProviderConfig::OpenRouter {
                api_url, endpoints, ..
            }
            | ProviderConfig::Zai {
                api_url, endpoints, ..
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

    pub fn models_url(&self, provider_name: &str) -> Option<String> {
        match self {
            ProviderConfig::OpenAi {
                api_url,
                models_url,
                ..
            } => models_url
                .clone()
                .or_else(|| infer_openai_models_url(api_url))
                .or_else(|| Some(format!("{}/v1/models", api_url.trim_end_matches('/')))),
            ProviderConfig::OpenRouter {
                api_url,
                models_url,
                ..
            } => models_url
                .clone()
                .or_else(|| infer_openai_models_url(api_url))
                .or_else(|| Some(format!("{}/v1/models", api_url.trim_end_matches('/')))),
            _ => {
                let _ = provider_name;
                None
            }
        }
    }

    pub fn validate(&self, provider_name: &str) -> Result<(), ConfigError> {
        match self {
            ProviderConfig::Gemini {
                api_internal,
                api_public,
                ..
            } => {
                validate_url(
                    api_internal,
                    &format!("Provider '{}' gemini api_internal", provider_name),
                )?;
                validate_url(
                    api_public,
                    &format!("Provider '{}' gemini api_public", provider_name),
                )?;
            }
            ProviderConfig::Zai {
                api_url, endpoints, ..
            } => {
                validate_url(api_url, &format!("Provider '{}' api_url", provider_name))?;
                for (name, url) in endpoints {
                    validate_url(
                        url,
                        &format!("Provider '{}' endpoint '{}'", provider_name, name),
                    )?;
                }
            }
            ProviderConfig::OpenAi {
                api_url,
                endpoints,
                models_url,
                ..
            } => {
                validate_url(api_url, &format!("Provider '{}' api_url", provider_name))?;
                for (name, url) in endpoints {
                    validate_url(
                        url,
                        &format!("Provider '{}' endpoint '{}'", provider_name, name),
                    )?;
                }
                if let Some(url) = models_url {
                    validate_url(url, &format!("Provider '{}' models_url", provider_name))?;
                }
            }
            ProviderConfig::OpenRouter {
                api_url,
                endpoints,
                models_url,
                ..
            } => {
                validate_url(api_url, &format!("Provider '{}' api_url", provider_name))?;
                for (name, url) in endpoints {
                    validate_url(
                        url,
                        &format!("Provider '{}' endpoint '{}'", provider_name, name),
                    )?;
                }
                if let Some(url) = models_url {
                    validate_url(url, &format!("Provider '{}' models_url", provider_name))?;
                }
            }
        }
        Ok(())
    }
}

fn infer_openai_models_url(api_url: &str) -> Option<String> {
    let trimmed = api_url.trim_end_matches('/');
    if let Some(prefix) = trimmed.strip_suffix("/v1/responses") {
        return Some(format!("{prefix}/v1/models"));
    }
    if let Some(prefix) = trimmed.strip_suffix("/responses") {
        return Some(format!("{prefix}/models"));
    }
    None
}

pub type ProvidersConfig = HashMap<String, ProviderConfig>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    #[serde(default)]
    pub served: Vec<String>,
    #[serde(default)]
    pub fallback_models: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelDiscoveryConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_model_discovery_interval_seconds")]
    pub interval_seconds: u64,
}

fn default_model_discovery_interval_seconds() -> u64 {
    300
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default = "default_session_header_name")]
    pub header_name: String,
    #[serde(default = "default_session_metadata_key")]
    pub metadata_key: String,
    #[serde(default = "default_session_response_id_ttl_seconds")]
    pub response_id_ttl_seconds: u64,
}

fn default_session_header_name() -> String {
    "x-codex-proxy-session".into()
}

fn default_session_metadata_key() -> String {
    "codex_proxy_session".into()
}

fn default_session_response_id_ttl_seconds() -> u64 {
    24 * 60 * 60
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            header_name: default_session_header_name(),
            metadata_key: default_session_metadata_key(),
            response_id_ttl_seconds: default_session_response_id_ttl_seconds(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoCompactionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_auto_compaction_max_attempts")]
    pub max_attempts_per_request: u32,
    #[serde(default = "default_auto_compaction_tail_items")]
    pub tail_items_to_keep: usize,
}

fn default_auto_compaction_max_attempts() -> u32 {
    1
}

fn default_auto_compaction_tail_items() -> usize {
    8
}

pub const AUTO_COMPACTION_COMPACT_INSTRUCTIONS: &str = "Compact the conversation history for continued use. Preserve all tool and file context needed to continue the session.";

pub const AUTO_COMPACTION_SUMMARY_INSTRUCTIONS: &str = "Summarize the conversation history so far for continued use. Preserve key decisions, constraints, open tasks, file paths, and relevant technical details. Be concise but complete.";

impl Default for AutoCompactionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_attempts_per_request: default_auto_compaction_max_attempts(),
            tail_items_to_keep: default_auto_compaction_tail_items(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricingConfig {
    #[serde(default)]
    pub input_per_mtoken: Option<f64>,
    #[serde(default)]
    pub output_per_mtoken: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetadataConfig {
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub pricing: Option<ModelPricingConfig>,
}

pub type ProviderModelMetadataConfig = HashMap<String, HashMap<String, ModelMetadataConfig>>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelsEndpointSource {
    Served,
    Discovered,
    Both,
}

fn default_models_endpoint_source() -> ModelsEndpointSource {
    ModelsEndpointSource::Served
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsEndpointConfig {
    #[serde(default = "default_models_endpoint_source")]
    pub source: ModelsEndpointSource,
}

impl Default for ModelsEndpointConfig {
    fn default() -> Self {
        Self {
            source: default_models_endpoint_source(),
        }
    }
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
#[serde(deny_unknown_fields)]
pub struct RoutingConfig {
    #[serde(default)]
    pub model_overrides: HashMap<String, String>,
    #[serde(default, alias = "preferred_models")]
    pub model_provider_priority: HashMap<String, Vec<RouteTargetConfig>>,
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
    #[serde(default, deserialize_with = "deserialize_compaction_preferred_targets")]
    pub preferred_targets: Vec<String>,
}

fn deserialize_compaction_preferred_targets<'de, D>(
    deserializer: D,
) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum PreferredTargetItem {
        String(String),
        Map(serde_json::Map<String, serde_json::Value>),
    }

    let raw = Vec::<PreferredTargetItem>::deserialize(deserializer)?;
    raw.into_iter()
        .map(|item| match item {
            PreferredTargetItem::String(s) => Ok(s),
            PreferredTargetItem::Map(map) => {
                if let Some(v) = map.get("logical_model").and_then(|v| v.as_str()) {
                    return Ok(v.to_string());
                }
                if let Some(v) = map.get("model").and_then(|v| v.as_str()) {
                    return Ok(v.to_string());
                }
                Err(D::Error::custom(
                    "compaction.preferred_targets entries must be a string or an object with 'logical_model' or 'model'",
                ))
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessKeyRole {
    Admin,
    Api,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessKeyConfig {
    pub id: String,
    #[serde(default)]
    pub key_sha256: String,
    #[serde(default, skip_serializing)]
    pub plaintext: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub role: Option<AccessKeyRole>,
    #[serde(default, skip_serializing)]
    pub is_admin: bool,
}

impl AccessKeyConfig {
    pub fn effective_role(&self) -> AccessKeyRole {
        self.role.unwrap_or_else(|| {
            if self.is_admin {
                AccessKeyRole::Admin
            } else {
                AccessKeyRole::Api
            }
        })
    }

    pub fn normalize_secret(&mut self) -> Result<(), ConfigError> {
        let has_hash = !self.key_sha256.trim().is_empty();
        let plaintext = self.plaintext.as_deref().unwrap_or("").trim();
        let has_plaintext = !plaintext.is_empty();

        match (has_hash, has_plaintext) {
            (true, true) => Err(ConfigError::InvalidValue(format!(
                "access.keys['{}'] must set only one of key_sha256 or plaintext",
                self.id
            ))),
            (false, false) => Err(ConfigError::InvalidValue(format!(
                "access.keys['{}'] must set key_sha256 or plaintext",
                self.id
            ))),
            (false, true) => {
                self.key_sha256 = crate::access::sha256_hex(plaintext);
                self.plaintext = None;
                Ok(())
            }
            (true, false) => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    #[serde(default)]
    pub require_key: bool,
    #[serde(default)]
    pub keys: Vec<AccessKeyConfig>,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            require_key: false,
            keys: Vec::new(),
        }
    }
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
    pub model_discovery: ModelDiscoveryConfig,
    pub model_metadata: ProviderModelMetadataConfig,
    pub models_endpoint: ModelsEndpointConfig,
    pub session: SessionConfig,
    pub auto_compaction: AutoCompactionConfig,
    pub routing: RoutingConfig,
    pub health: RoutingHealthConfig,
    pub accounts: Vec<AccountConfig>,
    pub access: AccessControlConfig,
    pub reasoning: ReasoningConfig,
    pub timeouts: TimeoutsConfig,
    pub compaction: CompactionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedConfig {
    pub server: ServerConfig,
    pub providers: ProvidersConfig,
    pub models: ModelsConfig,
    #[serde(default)]
    pub model_discovery: ModelDiscoveryConfig,
    #[serde(default)]
    pub model_metadata: ProviderModelMetadataConfig,
    #[serde(default)]
    pub models_endpoint: ModelsEndpointConfig,
    #[serde(default)]
    pub session: SessionConfig,
    #[serde(default)]
    pub auto_compaction: AutoCompactionConfig,
    pub routing: RoutingConfig,
    #[serde(default)]
    pub health: RoutingHealthConfig,
    pub accounts: Vec<AccountConfig>,
    #[serde(default)]
    pub access: AccessControlConfig,
    pub reasoning: ReasoningConfig,
    pub timeouts: TimeoutsConfig,
    pub compaction: CompactionConfig,
}

impl PersistedConfig {
    pub fn into_runtime(mut self, config_path: PathBuf) -> Config {
        for key in &mut self.access.keys {
            key.plaintext = None;
        }
        Config {
            config_path,
            server: self.server,
            providers: self.providers,
            models: self.models,
            model_discovery: self.model_discovery,
            model_metadata: self.model_metadata,
            models_endpoint: self.models_endpoint,
            session: self.session,
            auto_compaction: self.auto_compaction,
            routing: self.routing,
            health: self.health,
            accounts: self.accounts,
            access: self.access,
            reasoning: self.reasoning,
            timeouts: self.timeouts,
            compaction: self.compaction,
        }
    }
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
    pub model_discovery: Option<ModelDiscoveryConfig>,
    #[serde(default)]
    pub model_metadata: Option<ProviderModelMetadataConfig>,
    #[serde(default)]
    pub models_endpoint: Option<ModelsEndpointConfig>,
    #[serde(default)]
    pub session: Option<SessionConfig>,
    #[serde(default)]
    pub auto_compaction: Option<AutoCompactionConfig>,
    #[serde(default)]
    pub routing: Option<RoutingConfig>,
    #[serde(default)]
    pub health: Option<RoutingHealthConfig>,
    #[serde(default)]
    pub accounts: Option<Vec<AccountConfig>>,
    #[serde(default)]
    pub access: Option<AccessControlConfig>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub timeouts: Option<TimeoutsConfig>,
    #[serde(default)]
    pub compaction: Option<CompactionConfig>,
}

pub type ConfigHandle = Arc<RwLock<Config>>;

pub fn with_config<T>(handle: &ConfigHandle, f: impl FnOnce(&Config) -> T) -> T {
    let guard = handle.read();
    f(&guard)
}

pub fn with_config_mut<T>(handle: &ConfigHandle, f: impl FnOnce(&mut Config) -> T) -> T {
    let mut guard = handle.write();
    f(&mut guard)
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    pub fn new() -> Self {
        let mut cfg = Self::defaults();
        let loaded = cfg.load_from_file();
        if !loaded {
            let tried = default_config_search_paths(&dirs_home())
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            panic!("No config file found. Tried: {tried}");
        }
        cfg.validate().expect("invalid configuration");
        cfg
    }

    pub fn new_from_path(path: impl AsRef<Path>) -> Self {
        let mut cfg = Self::defaults();
        cfg.load_from_path(path.as_ref());
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
                models_url: None,
                endpoints: HashMap::new(),
                models: Vec::new(),
                max_tokens_cap: None,
            },
        );

        Self {
            config_path: default_config_search_paths(&home)[0].clone(),
            server: ServerConfig {
                host,
                port,
                log_level,
            },
            providers,
            models: ModelsConfig {
                served: served_models,
                fallback_models: HashMap::new(),
            },
            model_discovery: ModelDiscoveryConfig::default(),
            model_metadata: ProviderModelMetadataConfig::new(),
            models_endpoint: ModelsEndpointConfig::default(),
            session: SessionConfig::default(),
            auto_compaction: AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_overrides: HashMap::new(),
                model_provider_priority: HashMap::new(),
            },
            health: RoutingHealthConfig::default(),
            accounts,
            access: AccessControlConfig::default(),
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

    fn load_from_path(&mut self, path: &Path) {
        if !path.exists() {
            panic!(
                "Config {} does not exist. Refusing to start.",
                path.display()
            );
        }
        let content = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read config {}: {e}", path.display()));
        let file_cfg: FileConfig = parse_json_with_path(&content)
            .unwrap_or_else(|e| panic!("Failed to parse config {}: {e}", path.display()));
        if file_cfg.access.is_none() {
            panic!(
                "Config {} is missing required 'access' section. Refusing to start.",
                path.display()
            );
        }

        if let Some(server) = file_cfg.server {
            self.server = server;
        }
        if let Some(providers) = file_cfg.providers {
            self.providers = providers;
        }
        if let Some(models) = file_cfg.models {
            self.models = models;
        }
        if let Some(model_discovery) = file_cfg.model_discovery {
            self.model_discovery = model_discovery;
        }
        if let Some(model_metadata) = file_cfg.model_metadata {
            self.model_metadata = model_metadata;
        }
        if let Some(models_endpoint) = file_cfg.models_endpoint {
            self.models_endpoint = models_endpoint;
        }
        if let Some(session) = file_cfg.session {
            self.session = session;
        }
        if let Some(auto_compaction) = file_cfg.auto_compaction {
            self.auto_compaction = auto_compaction;
        }
        if let Some(routing) = file_cfg.routing {
            self.routing = routing;
        }
        if let Some(health) = file_cfg.health {
            self.health = health;
        }
        if let Some(accounts) = file_cfg.accounts {
            self.accounts = accounts;
        }
        if let Some(access) = file_cfg.access {
            self.access = access;
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

        self.config_path = path.to_path_buf();
        info!("Loaded config from {}", self.config_path.display());
    }

    fn load_from_file(&mut self) -> bool {
        for path in default_config_search_paths(&dirs_home()) {
            if !path.exists() {
                continue;
            }
            let content = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read config {}: {e}", path.display()));
            let file_cfg: FileConfig = parse_json_with_path(&content).unwrap_or_else(|e| {
                panic!("Failed to parse config {}: {e}", path.display());
            });
            if file_cfg.access.is_none() {
                panic!(
                    "Config {} is missing required 'access' section. Refusing to start.",
                    path.display()
                );
            }

            if let Some(server) = file_cfg.server {
                self.server = server;
            }
            if let Some(providers) = file_cfg.providers {
                self.providers = providers;
            }
            if let Some(models) = file_cfg.models {
                self.models = models;
            }
            if let Some(model_discovery) = file_cfg.model_discovery {
                self.model_discovery = model_discovery;
            }
            if let Some(model_metadata) = file_cfg.model_metadata {
                self.model_metadata = model_metadata;
            }
            if let Some(models_endpoint) = file_cfg.models_endpoint {
                self.models_endpoint = models_endpoint;
            }
            if let Some(session) = file_cfg.session {
                self.session = session;
            }
            if let Some(auto_compaction) = file_cfg.auto_compaction {
                self.auto_compaction = auto_compaction;
            }
            if let Some(routing) = file_cfg.routing {
                self.routing = routing;
            }
            if let Some(health) = file_cfg.health {
                self.health = health;
            }
            if let Some(accounts) = file_cfg.accounts {
                self.accounts = accounts;
            }
            if let Some(access) = file_cfg.access {
                self.access = access;
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
            return true;
        }
        false
    }

    pub fn to_persisted(&self) -> PersistedConfig {
        PersistedConfig {
            server: self.server.clone(),
            providers: self.providers.clone(),
            models: self.models.clone(),
            model_discovery: self.model_discovery.clone(),
            model_metadata: self.model_metadata.clone(),
            models_endpoint: self.models_endpoint.clone(),
            session: self.session.clone(),
            auto_compaction: self.auto_compaction.clone(),
            routing: self.routing.clone(),
            health: self.health.clone(),
            accounts: self.accounts.clone(),
            access: self.access.clone(),
            reasoning: self.reasoning.clone(),
            timeouts: self.timeouts.clone(),
            compaction: self.compaction.clone(),
        }
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), ConfigError> {
        let persisted = self.to_persisted();
        let json = serde_json::to_string_pretty(&persisted)?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let tmp_path = parent.join(format!(
            ".{}.tmp",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("config.json")
        ));
        fs::write(&tmp_path, json)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    pub fn validate(&mut self) -> Result<(), ConfigError> {
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

        for (provider, models) in &self.model_metadata {
            self.provider_type(provider)?;
            for (model, metadata) in models {
                if model.trim().is_empty() {
                    return Err(ConfigError::InvalidValue(format!(
                        "model_metadata['{}'] contains an empty model id",
                        provider
                    )));
                }
                if let Some(value) = metadata.context_window
                    && value == 0
                {
                    return Err(ConfigError::InvalidValue(format!(
                        "model_metadata['{}']['{}'].context_window must be > 0",
                        provider, model
                    )));
                }
                if let Some(value) = metadata.max_output_tokens
                    && value == 0
                {
                    return Err(ConfigError::InvalidValue(format!(
                        "model_metadata['{}']['{}'].max_output_tokens must be > 0",
                        provider, model
                    )));
                }
                if let Some(pricing) = &metadata.pricing {
                    if let Some(v) = pricing.input_per_mtoken
                        && v < 0.0
                    {
                        return Err(ConfigError::InvalidValue(format!(
                            "model_metadata['{}']['{}'].pricing.input_per_mtoken must be >= 0",
                            provider, model
                        )));
                    }
                    if let Some(v) = pricing.output_per_mtoken
                        && v < 0.0
                    {
                        return Err(ConfigError::InvalidValue(format!(
                            "model_metadata['{}']['{}'].pricing.output_per_mtoken must be >= 0",
                            provider, model
                        )));
                    }
                }
            }
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

        let mut seen_access_ids = HashSet::new();
        let mut enabled_access_key_count = 0usize;
        for key in &mut self.access.keys {
            if !seen_access_ids.insert(key.id.clone()) {
                return Err(ConfigError::InvalidValue(format!(
                    "duplicate access key id: {}",
                    key.id
                )));
            }
            if key.enabled {
                enabled_access_key_count += 1;
            }
            key.normalize_secret()?;
            let hash = key.key_sha256.trim();
            let is_hex_64 = hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit());
            if !is_hex_64 {
                return Err(ConfigError::InvalidValue(format!(
                    "access.keys['{}'] key_sha256 must be 64 hex chars",
                    key.id
                )));
            }
        }
        if self.access.require_key && enabled_access_key_count == 0 {
            return Err(ConfigError::InvalidValue(
                "access.require_key is true but no enabled access keys are configured".into(),
            ));
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
                (
                    ProviderType::Zai | ProviderType::OpenAi | ProviderType::OpenRouter,
                    AccountAuth::ApiKey { api_key },
                ) => {
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

        for (logical_model, targets) in &self.routing.model_provider_priority {
            if targets.is_empty() {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_provider_priority['{}'] must not be empty",
                    logical_model
                )));
            }
            for target in targets {
                self.validate_route_target(
                    target,
                    &format!("routing.model_provider_priority['{}']", logical_model),
                )?;
            }
            if !self.has_compatible_enabled_account(targets) {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_provider_priority['{}'] has no compatible enabled account",
                    logical_model
                )));
            }
        }

        for logical_model in &self.compaction.preferred_targets {
            if !self
                .routing
                .model_provider_priority
                .contains_key(logical_model.as_str())
            {
                return Err(ConfigError::InvalidValue(format!(
                    "compaction.preferred_targets references logical model '{}' which is not defined in routing.model_provider_priority",
                    logical_model
                )));
            }
        }

        for (requested_model, mapped_model) in &self.routing.model_overrides {
            if !self
                .routing
                .model_provider_priority
                .contains_key(mapped_model)
            {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_overrides['{}'] target '{}' is not defined in routing.model_provider_priority",
                    requested_model, mapped_model
                )));
            }
        }

        if !self.models.served.is_empty() {
            for served_model in &self.models.served {
                let logical_model = self.resolve_logical_model(served_model);
                if !self
                    .routing
                    .model_provider_priority
                    .contains_key(&logical_model)
                {
                    return Err(ConfigError::InvalidValue(format!(
                        "served model '{}' resolves to logical model '{}' but no routing.model_provider_priority entry exists",
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
            .or_else(|| self.routing.model_overrides.get("*"))
            .cloned()
            .unwrap_or_else(|| requested_model.to_string())
    }

    pub fn preferred_targets_for_model(
        &self,
        requested_model: &str,
    ) -> Option<&[RouteTargetConfig]> {
        let logical_model = self.resolve_logical_model(requested_model);
        self.routing
            .model_provider_priority
            .get(&logical_model)
            .map(Vec::as_slice)
    }

    pub fn compaction_targets(&self) -> Vec<RouteTargetConfig> {
        self.compaction
            .preferred_targets
            .iter()
            .filter_map(|logical_model| {
                self.routing
                    .model_provider_priority
                    .get(logical_model)
                    .cloned()
            })
            .flatten()
            .collect()
    }

    pub fn recovery_probe_target(&self, provider: &str) -> Option<RouteTargetConfig> {
        // Health probes should not depend on routing: routing targets might omit providers
        // that are only used as fallbacks, and probes should still work for those accounts.
        let has_enabled_account = self
            .accounts
            .iter()
            .any(|account| account.enabled && account.provider == provider);
        if !has_enabled_account {
            return None;
        }

        let catalog = self.provider_catalog(provider);
        let resolve_model = |model: &str| -> Option<String> {
            let model = model.trim();
            if model.is_empty() {
                return None;
            }
            match &catalog {
                Some(catalog) => catalog
                    .iter()
                    .any(|m| m == model)
                    .then(|| model.to_string()),
                None => Some(model.to_string()),
            }
        };

        let model = self
            .accounts
            .iter()
            .filter(|account| account.enabled && account.provider == provider)
            .filter_map(|account| account.models.as_ref())
            .flat_map(|models| models.iter())
            .find_map(|model| resolve_model(model))
            .or_else(|| {
                catalog
                    .as_ref()
                    .and_then(|catalog| catalog.first())
                    .cloned()
            })?;

        Some(RouteTargetConfig {
            provider: provider.to_string(),
            model,
            endpoint: None,
            reasoning: None,
        })
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

    pub fn provider_models_url(&self, provider: &str) -> Option<String> {
        self.providers.get(provider)?.models_url(provider)
    }

    pub fn model_metadata(&self, provider: &str, model: &str) -> Option<&ModelMetadataConfig> {
        self.model_metadata.get(provider)?.get(model)
    }

    pub fn endpoint_url(
        &self,
        provider: &str,
        endpoint: Option<&str>,
    ) -> Result<String, ConfigError> {
        self.providers
            .get(provider)
            .ok_or_else(|| ConfigError::InvalidValue(format!("Unknown provider '{}'", provider)))?
            .endpoint_url(provider, endpoint)
    }

    pub fn gemini_provider_config(
        &self,
        provider: &str,
    ) -> Result<GeminiProviderConfig, ConfigError> {
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

    pub fn openai_provider_config(
        &self,
        provider: &str,
    ) -> Result<OpenAiProviderConfig, ConfigError> {
        self.providers
            .get(provider)
            .and_then(ProviderConfig::as_openai)
            .ok_or_else(|| {
                ConfigError::InvalidValue(format!(
                    "Provider '{}' is not configured as type openai",
                    provider
                ))
            })
    }

    fn validate_route_target(
        &self,
        target: &RouteTargetConfig,
        scope: &str,
    ) -> Result<(), ConfigError> {
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

fn parse_json_with_path<T: DeserializeOwned>(content: &str) -> Result<T, String> {
    let mut deserializer = serde_json::Deserializer::from_str(content);
    let result: Result<T, _> = serde_path_to_error::deserialize(&mut deserializer);
    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            let path = e.path().to_string();
            let inner = e.into_inner();
            if path.is_empty() {
                Err(inner.to_string())
            } else {
                Err(format!("{inner} (at {path})"))
            }
        }
    }
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
                        models_url: None,
                        endpoints: HashMap::from([(
                            "priority".into(),
                            "https://priority.openai.com/v1/responses".into(),
                        )]),
                        models: vec!["gpt-4.1".into(), "gpt-4.1-mini".into()],
                        max_tokens_cap: None,
                    },
                ),
                (
                    "tabcode".into(),
                    ProviderConfig::OpenAi {
                        api_url: "https://tabcode.example/v1/responses".into(),
                        models_url: None,
                        endpoints: HashMap::new(),
                        models: vec!["gpt-4.1".into()],
                        max_tokens_cap: None,
                    },
                ),
            ]),
            models: ModelsConfig {
                served: vec!["claude-sonnet-4-6".into()],
                fallback_models: HashMap::new(),
            },
            model_discovery: ModelDiscoveryConfig::default(),
            model_metadata: ProviderModelMetadataConfig::new(),
            models_endpoint: ModelsEndpointConfig::default(),
            session: SessionConfig::default(),
            auto_compaction: AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_overrides: HashMap::new(),
                model_provider_priority: HashMap::from([(
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
            },
            health: RoutingHealthConfig::default(),
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
            access: AccessControlConfig::default(),
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 30,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: vec!["claude-sonnet-4-6".into()],
            },
        }
    }

    #[test]
    fn validates_capability_routing_config() {
        let mut cfg = base_config();
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
        let target = &cfg.routing.model_provider_priority["claude-sonnet-4-6"][0];
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

    #[test]
    fn allows_empty_compaction_targets() {
        let mut cfg = base_config();
        cfg.compaction.preferred_targets = Vec::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn deserialize_legacy_preferred_models_alias() {
        let routing: RoutingConfig = serde_json::from_value(serde_json::json!({
            "preferred_models": {
                "claude-sonnet-4-6": [
                    { "provider": "tabcode", "model": "gpt-4.1" }
                ]
            }
        }))
        .unwrap();

        assert!(
            routing
                .model_provider_priority
                .contains_key("claude-sonnet-4-6")
        );
    }

    #[test]
    fn serializes_routing_with_new_field_name_only() {
        let cfg = base_config();
        let value = serde_json::to_value(&cfg.routing).unwrap();

        assert!(value.get("model_provider_priority").is_some());
        assert!(value.get("preferred_models").is_none());
        assert!(value.get("health").is_none());
    }

    #[test]
    fn persisted_config_uses_top_level_health() {
        let cfg = base_config();
        let value = serde_json::to_value(cfg.to_persisted()).unwrap();

        assert!(value.get("health").is_some());
        assert!(
            value
                .get("routing")
                .and_then(|routing| routing.get("health"))
                .is_none()
        );
    }

    #[test]
    fn persisted_config_accepts_top_level_health() {
        let persisted: PersistedConfig = serde_json::from_value(serde_json::json!({
            "server": {
                "host": "127.0.0.1",
                "port": 8765,
                "log_level": "INFO"
            },
            "providers": {
                "tabcode": {
                    "type": "open_ai",
                    "api_url": "https://tabcode.example/v1/responses",
                    "endpoints": {},
                    "models": ["gpt-4.1"]
                }
            },
            "models": {
                "served": ["claude-sonnet-4-6"],
                "fallback_models": {}
            },
            "routing": {
                "model_overrides": {},
                "model_provider_priority": {
                    "claude-sonnet-4-6": [
                        { "provider": "tabcode", "model": "gpt-4.1" }
                    ]
                },
            },
            "health": {
                "auth_failure_immediate_unhealthy": false,
                "failure_threshold": 7,
                "cooldown_seconds": 90
            },
            "accounts": [
                {
                    "id": "tabcode-a",
                    "provider": "tabcode",
                    "enabled": true,
                    "weight": 1,
                    "models": ["gpt-4.1"],
                    "auth": { "type": "api_key", "api_key": "sk-test" }
                }
            ],
            "access": {
                "require_key": false,
                "keys": []
            },
            "reasoning": {
                "effort_levels": {
                    "none": { "budget": 0, "level": "LOW" }
                },
                "default_effort": null
            },
            "timeouts": {
                "connect_seconds": 10,
                "read_seconds": 30
            },
            "compaction": {
                "temperature": 0.1,
                "preferred_targets": []
            }
        }))
        .unwrap();

        assert_eq!(persisted.health.failure_threshold, 7);
        assert!(
            persisted
                .routing
                .model_provider_priority
                .contains_key("claude-sonnet-4-6")
        );
    }

    #[test]
    fn rejects_legacy_nested_routing_health() {
        let err = serde_json::from_value::<PersistedConfig>(serde_json::json!({
            "server": {
                "host": "127.0.0.1",
                "port": 8765,
                "log_level": "INFO"
            },
            "providers": {
                "tabcode": {
                    "type": "open_ai",
                    "api_url": "https://tabcode.example/v1/responses",
                    "endpoints": {},
                    "models": ["gpt-4.1"]
                }
            },
            "models": {
                "served": ["claude-sonnet-4-6"],
                "fallback_models": {}
            },
            "routing": {
                "model_overrides": {},
                "model_provider_priority": {
                    "claude-sonnet-4-6": [
                        { "provider": "tabcode", "model": "gpt-4.1" }
                    ]
                },
                "health": {
                    "auth_failure_immediate_unhealthy": true,
                    "failure_threshold": 3,
                    "cooldown_seconds": 30
                }
            },
            "health": {
                "auth_failure_immediate_unhealthy": true,
                "failure_threshold": 3,
                "cooldown_seconds": 30
            },
            "accounts": [],
            "access": {
                "require_key": false,
                "keys": []
            },
            "reasoning": {
                "effort_levels": {
                    "none": { "budget": 0, "level": "LOW" }
                },
                "default_effort": null
            },
            "timeouts": {
                "connect_seconds": 10,
                "read_seconds": 30
            },
            "compaction": {
                "temperature": 0.1,
                "preferred_targets": []
            }
        }))
        .unwrap_err();

        assert!(err.to_string().contains("unknown field `health`"));
    }

    #[test]
    fn rejects_legacy_sticky_routing_config() {
        let err = serde_json::from_value::<PersistedConfig>(serde_json::json!({
            "server": {
                "host": "127.0.0.1",
                "port": 8765,
                "log_level": "INFO"
            },
            "providers": {
                "tabcode": {
                    "type": "open_ai",
                    "api_url": "https://tabcode.example/v1/responses",
                    "endpoints": {},
                    "models": ["gpt-4.1"]
                }
            },
            "models": {
                "served": ["claude-sonnet-4-6"],
                "fallback_models": {}
            },
            "routing": {
                "model_overrides": {},
                "model_provider_priority": {
                    "claude-sonnet-4-6": [
                        { "provider": "tabcode", "model": "gpt-4.1" }
                    ]
                },
                "sticky_routing": {
                    "enabled": true
                }
            },
            "health": {
                "auth_failure_immediate_unhealthy": true,
                "failure_threshold": 3,
                "cooldown_seconds": 30
            },
            "accounts": [],
            "access": {
                "require_key": false,
                "keys": []
            },
            "reasoning": {
                "effort_levels": {
                    "none": { "budget": 0, "level": "LOW" }
                },
                "default_effort": null
            },
            "timeouts": {
                "connect_seconds": 10,
                "read_seconds": 30
            },
            "compaction": {
                "temperature": 0.1,
                "preferred_targets": []
            }
        }))
        .unwrap_err();

        assert!(err.to_string().contains("unknown field `sticky_routing`"));
    }

    #[test]
    fn exact_model_override_beats_wildcard() {
        let mut cfg = base_config();
        cfg.routing.model_overrides = HashMap::from([
            ("*".into(), "fallback-logical".into()),
            ("claude-sonnet-4-6".into(), "exact-logical".into()),
        ]);

        assert_eq!(
            cfg.resolve_logical_model("claude-sonnet-4-6"),
            "exact-logical"
        );
    }

    #[test]
    fn wildcard_model_override_applies_when_exact_is_missing() {
        let mut cfg = base_config();
        cfg.routing.model_overrides = HashMap::from([("*".into(), "claude-sonnet-4-6".into())]);

        assert_eq!(
            cfg.resolve_logical_model("unmapped-model"),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn wildcard_override_target_must_exist_in_model_provider_priority() {
        let mut cfg = base_config();
        cfg.routing.model_overrides = HashMap::from([("*".into(), "missing-logical".into())]);

        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("routing.model_overrides['*'] target 'missing-logical' is not defined in routing.model_provider_priority"));
    }

    #[test]
    fn preferred_targets_use_wildcard_resolved_logical_model() {
        let mut cfg = base_config();
        cfg.routing.model_overrides = HashMap::from([("*".into(), "claude-sonnet-4-6".into())]);

        let targets = cfg.preferred_targets_for_model("unmapped-model").unwrap();
        assert_eq!(targets[0].provider, "tabcode");
        assert_eq!(targets[0].model, "gpt-4.1");
    }

    #[test]
    fn recovery_probe_target_ignores_routing_target() {
        let mut cfg = base_config();
        cfg.providers.insert(
            "route-only".into(),
            ProviderConfig::OpenAi {
                api_url: "https://route-only.example/v1/responses".into(),
                models_url: None,
                endpoints: HashMap::new(),
                models: vec!["account-model".into(), "catalog-model".into()],
                max_tokens_cap: None,
            },
        );
        cfg.routing.model_provider_priority.insert(
            "route-only-logical".into(),
            vec![RouteTargetConfig {
                provider: "route-only".into(),
                model: "routed-model".into(),
                endpoint: None,
                reasoning: None,
            }],
        );
        cfg.accounts.push(AccountConfig {
            id: "route-only-a".into(),
            provider: "route-only".into(),
            enabled: true,
            weight: 1,
            models: Some(vec!["account-model".into()]),
            auth: AccountAuth::ApiKey {
                api_key: "sk-test".into(),
            },
        });

        let target = cfg.recovery_probe_target("route-only").unwrap();
        assert_eq!(target.provider, "route-only");
        assert_eq!(target.model, "account-model");
    }

    #[test]
    fn recovery_probe_target_uses_account_model_when_no_routing_target_exists() {
        let mut cfg = base_config();
        cfg.providers.insert(
            "account-only".into(),
            ProviderConfig::OpenAi {
                api_url: "https://account-only.example/v1/responses".into(),
                models_url: None,
                endpoints: HashMap::new(),
                models: vec!["account-model".into(), "other-model".into()],
                max_tokens_cap: None,
            },
        );
        cfg.accounts.push(AccountConfig {
            id: "account-only-a".into(),
            provider: "account-only".into(),
            enabled: true,
            weight: 1,
            models: Some(vec!["account-model".into()]),
            auth: AccountAuth::ApiKey {
                api_key: "sk-test".into(),
            },
        });

        let target = cfg.recovery_probe_target("account-only").unwrap();
        assert_eq!(target.model, "account-model");
    }

    #[test]
    fn recovery_probe_target_uses_provider_catalog_when_account_models_missing() {
        let mut cfg = base_config();
        cfg.providers.insert(
            "no-probe".into(),
            ProviderConfig::OpenAi {
                api_url: "https://no-probe.example/v1/responses".into(),
                models_url: None,
                endpoints: HashMap::new(),
                models: vec!["catalog-model".into()],
                max_tokens_cap: None,
            },
        );
        cfg.accounts.push(AccountConfig {
            id: "no-probe-a".into(),
            provider: "no-probe".into(),
            enabled: true,
            weight: 1,
            models: None,
            auth: AccountAuth::ApiKey {
                api_key: "sk-test".into(),
            },
        });

        let target = cfg.recovery_probe_target("no-probe").unwrap();
        assert_eq!(target.model, "catalog-model");
        assert!(cfg.recovery_probe_target("missing-provider").is_none());
    }

    #[test]
    fn preferred_targets_accept_string_or_object() {
        let compaction: CompactionConfig = serde_json::from_value(serde_json::json!({
            "temperature": 0.1,
            "preferred_targets": [
              "glm-5-turbo",
              { "model": "claude-sonnet-4-6" },
              { "logical_model": "gpt-5" }
            ]
        }))
        .expect("compaction config must parse");

        assert_eq!(
            compaction.preferred_targets,
            vec!["glm-5-turbo", "claude-sonnet-4-6", "gpt-5"]
        );
    }
}
