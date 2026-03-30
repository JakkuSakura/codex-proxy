use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
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
#[serde(deny_unknown_fields)]
pub struct ModelsConfig {
    #[serde(default)]
    pub served: Vec<String>,
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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelRouteStepConfig {
    Physical {
        provider: String,
        model: String,
        #[serde(default)]
        endpoint: Option<String>,
        #[serde(default)]
        reasoning: Option<RouteReasoningConfig>,
    },
    Logical {
        model: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ModelRouteStepWire {
    Shorthand(String),
    Structured(ModelRouteStepConfig),
}

impl ModelRouteStepWire {
    fn into_step(self) -> Result<ModelRouteStepConfig, ConfigError> {
        let input = match self {
            ModelRouteStepWire::Structured(step) => return Ok(step),
            ModelRouteStepWire::Shorthand(input) => input,
        };

        let input = input.trim();
        if input.is_empty() {
            return Err(ConfigError::InvalidValue(
                "route step shorthand must not be empty".into(),
            ));
        }
        let mut parts = input.splitn(2, ':');
        let prefix = parts.next().unwrap_or_default().trim();
        let rest = parts.next().unwrap_or_default().trim();
        if prefix.is_empty() || rest.is_empty() {
            return Err(ConfigError::InvalidValue(format!(
                "invalid route step shorthand '{input}': expected 'provider:model' or 'proxy:logical_model'"
            )));
        }
        if prefix == "proxy" {
            return Ok(ModelRouteStepConfig::Logical {
                model: rest.to_string(),
            });
        }
        if prefix == "open_router" {
            return Err(ConfigError::InvalidValue(
                "provider name 'open_router' is not supported; use 'openrouter'".into(),
            ));
        }
        Ok(ModelRouteStepConfig::Physical {
            provider: prefix.to_string(),
            model: rest.to_string(),
            endpoint: None,
            reasoning: None,
        })
    }
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
    pub model_routes: HashMap<String, Vec<ModelRouteStepConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingConfigWire {
    #[serde(default)]
    pub model_routes: HashMap<String, Vec<ModelRouteStepWire>>,
}

impl RoutingConfigWire {
    pub fn into_runtime(self) -> Result<RoutingConfig, ConfigError> {
        let mut model_routes = HashMap::with_capacity(self.model_routes.len());
        for (route_key, steps) in self.model_routes {
            let mut out_steps = Vec::with_capacity(steps.len());
            for (idx, step) in steps.into_iter().enumerate() {
                let scope = format!("routing.model_routes['{}'][{}]", route_key, idx);
                let parsed = step
                    .into_step()
                    .map_err(|e| ConfigError::InvalidValue(format!("{scope}: {e}")))?;
                out_steps.push(parsed);
            }
            model_routes.insert(route_key, out_steps);
        }
        Ok(RoutingConfig { model_routes })
    }
}

impl RoutingConfig {
    pub fn to_wire(&self) -> RoutingConfigWire {
        let mut model_routes = HashMap::with_capacity(self.model_routes.len());
        for (route_key, steps) in &self.model_routes {
            model_routes.insert(
                route_key.clone(),
                steps
                    .iter()
                    .cloned()
                    .map(ModelRouteStepWire::Structured)
                    .collect(),
            );
        }
        RoutingConfigWire { model_routes }
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
    #[serde(default)]
    pub preferred_targets: Vec<String>,
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
    pub routing: RoutingConfigWire,
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
            routing: self
                .routing
                .into_runtime()
                .expect("invalid routing configuration"),
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
    pub routing: Option<RoutingConfigWire>,
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
            },
            model_discovery: ModelDiscoveryConfig::default(),
            model_metadata: ProviderModelMetadataConfig::new(),
            models_endpoint: ModelsEndpointConfig::default(),
            session: SessionConfig::default(),
            auto_compaction: AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_routes: HashMap::new(),
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
            self.routing = routing
                .into_runtime()
                .unwrap_or_else(|e| panic!("Invalid routing config in {}: {e}", path.display()));
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
                self.routing = routing.into_runtime().unwrap_or_else(|e| {
                    panic!("Invalid routing config in {}: {e}", path.display())
                });
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
            routing: self.routing.to_wire(),
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

        if self.routing.model_routes.is_empty() {
            return Err(ConfigError::InvalidValue(
                "routing.model_routes must not be empty".into(),
            ));
        }

        for (route_key, steps) in &self.routing.model_routes {
            if route_key.trim().is_empty() {
                return Err(ConfigError::InvalidValue(
                    "routing.model_routes keys must not be empty".into(),
                ));
            }
            if steps.is_empty() {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_routes['{}'] must not be empty",
                    route_key
                )));
            }

            for (idx, step) in steps.iter().enumerate() {
                match step {
                    ModelRouteStepConfig::Physical {
                        provider,
                        model,
                        endpoint,
                        reasoning,
                    } => {
                        let target = RouteTargetConfig {
                            provider: provider.clone(),
                            model: model.clone(),
                            endpoint: endpoint.clone(),
                            reasoning: reasoning.clone(),
                        };
                        self.validate_route_target(
                            &target,
                            &format!("routing.model_routes['{}'][{}]", route_key, idx),
                        )?;
                    }
                    ModelRouteStepConfig::Logical { model } => {
                        if model.trim().is_empty() {
                            return Err(ConfigError::InvalidValue(format!(
                                "routing.model_routes['{}'][{}] logical model reference must not be empty",
                                route_key, idx
                            )));
                        }
                        if !self.routing.model_routes.contains_key(model.as_str()) {
                            return Err(ConfigError::InvalidValue(format!(
                                "routing.model_routes['{}'][{}] references undefined logical model '{}'",
                                route_key, idx, model
                            )));
                        }
                    }
                }
            }

            let targets = self.expand_model_route(route_key)?;
            if !self.has_compatible_enabled_account(&targets) {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_routes['{}'] has no compatible enabled account",
                    route_key
                )));
            }
        }

        for logical_model in &self.compaction.preferred_targets {
            if !self
                .routing
                .model_routes
                .contains_key(logical_model.as_str())
            {
                return Err(ConfigError::InvalidValue(format!(
                    "compaction.preferred_targets references logical model '{}' which is not defined in routing.model_routes",
                    logical_model
                )));
            }
        }

        if !self.models.served.is_empty() {
            for served_model in &self.models.served {
                let Some((logical_model, _)) = self.route_targets_for_model(served_model) else {
                    return Err(ConfigError::InvalidValue(format!(
                        "served model '{}' has no routing targets after overrides/fallbacks",
                        served_model
                    )));
                };

                let _ = logical_model;
            }
        } else if !self.routing.model_routes.contains_key("*") {
            return Err(ConfigError::InvalidValue(
                "models.served is empty, so routing.model_routes must define a '*' fallback route"
                    .into(),
            ));
        }

        Ok(())
    }

    /// Resolve a request model into a routing logical model plus its target list.
    ///
    /// Resolution order:
    /// 1) `routing.model_routes[requested_model]`
    /// 2) `routing.model_routes["*"]`
    pub fn route_targets_for_model(
        &self,
        requested_model: &str,
    ) -> Option<(String, Vec<RouteTargetConfig>)> {
        let entry_key = if self.routing.model_routes.contains_key(requested_model) {
            requested_model
        } else if self.routing.model_routes.contains_key("*") {
            "*"
        } else {
            return None;
        };

        let logical_key = self.canonical_logical_route_key(entry_key).ok()?;
        let targets = self.expand_model_route(&logical_key).ok()?;
        Some((logical_key, targets))
    }

    pub fn preferred_targets_for_model(
        &self,
        requested_model: &str,
    ) -> Option<Vec<RouteTargetConfig>> {
        self.route_targets_for_model(requested_model)
            .map(|(_, targets)| targets)
    }

    pub fn compaction_targets(&self) -> Vec<RouteTargetConfig> {
        let mut targets = Vec::new();
        for logical_model in &self.compaction.preferred_targets {
            if let Ok(expanded) = self.expand_model_route(logical_model) {
                targets.extend(expanded);
            }
        }
        targets
    }

    fn canonical_logical_route_key(&self, entry_key: &str) -> Result<String, ConfigError> {
        let mut current = entry_key.to_string();
        let mut seen = std::collections::HashSet::<String>::new();
        loop {
            let steps = self.routing.model_routes.get(&current).ok_or_else(|| {
                ConfigError::InvalidValue(format!(
                    "routing.model_routes references undefined model route '{}'",
                    current
                ))
            })?;
            if steps.len() != 1 {
                return Ok(current);
            }
            let ModelRouteStepConfig::Logical { model: next } = &steps[0] else {
                return Ok(current);
            };

            if !seen.insert(current.clone()) {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_routes contains a logical cycle involving '{}'",
                    current
                )));
            }
            current = next.clone();
        }
    }

    fn expand_model_route(&self, route_key: &str) -> Result<Vec<RouteTargetConfig>, ConfigError> {
        let mut stack = Vec::new();
        let targets = self.expand_model_route_inner(route_key, &mut stack)?;
        if targets.is_empty() {
            return Err(ConfigError::InvalidValue(format!(
                "routing.model_routes['{}'] expands to zero physical targets",
                route_key
            )));
        }
        Ok(targets)
    }

    fn expand_model_route_inner(
        &self,
        route_key: &str,
        stack: &mut Vec<String>,
    ) -> Result<Vec<RouteTargetConfig>, ConfigError> {
        if stack.iter().any(|k| k == route_key) {
            stack.push(route_key.to_string());
            return Err(ConfigError::InvalidValue(format!(
                "routing.model_routes contains a logical cycle: {}",
                stack.join(" -> ")
            )));
        }

        let steps = self.routing.model_routes.get(route_key).ok_or_else(|| {
            ConfigError::InvalidValue(format!(
                "routing.model_routes references undefined logical model '{}'",
                route_key
            ))
        })?;

        stack.push(route_key.to_string());
        let mut out = Vec::new();
        for step in steps {
            match step {
                ModelRouteStepConfig::Physical {
                    provider,
                    model,
                    endpoint,
                    reasoning,
                } => out.push(RouteTargetConfig {
                    provider: provider.clone(),
                    model: model.clone(),
                    endpoint: endpoint.clone(),
                    reasoning: reasoning.clone(),
                }),
                ModelRouteStepConfig::Logical { model } => {
                    out.extend(self.expand_model_route_inner(model.as_str(), stack)?);
                }
            }
        }
        stack.pop();
        Ok(out)
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
        if target.provider.trim() == "proxy" {
            return Err(ConfigError::InvalidValue(format!(
                "{scope} contains reserved provider name 'proxy'; use a logical step ('proxy:<logical_model>') instead"
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
            },
            model_discovery: ModelDiscoveryConfig::default(),
            model_metadata: ProviderModelMetadataConfig::new(),
            models_endpoint: ModelsEndpointConfig::default(),
            session: SessionConfig::default(),
            auto_compaction: AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_routes: HashMap::from([(
                    "claude-sonnet-4-6".into(),
                    vec![ModelRouteStepConfig::Physical {
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
        let (_, targets) = cfg.route_targets_for_model("claude-sonnet-4-6").unwrap();
        let target = &targets[0];
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
    fn serializes_routing_with_model_routes_only() {
        let cfg = base_config();
        let value = serde_json::to_value(&cfg.routing).unwrap();

        assert!(value.get("model_routes").is_some());
        assert!(value.get("model_provider_priority").is_none());
        assert!(value.get("model_overrides").is_none());
    }

    #[test]
    fn parses_route_step_shorthand_physical_and_logical() {
        let physical: ModelRouteStepWire =
            serde_json::from_value(serde_json::json!("tabcode:gpt-4.1")).unwrap();
        assert_eq!(
            physical.into_step().unwrap(),
            ModelRouteStepConfig::Physical {
                provider: "tabcode".into(),
                model: "gpt-4.1".into(),
                endpoint: None,
                reasoning: None,
            }
        );

        let logical: ModelRouteStepWire =
            serde_json::from_value(serde_json::json!("proxy:glm-5-turbo")).unwrap();
        assert_eq!(
            logical.into_step().unwrap(),
            ModelRouteStepConfig::Logical {
                model: "glm-5-turbo".into()
            }
        );
    }

    #[test]
    fn wire_serializes_shorthand_and_structured_steps() {
        let shorthand = ModelRouteStepWire::Shorthand("tabcode:gpt-4.1".into());
        assert_eq!(
            serde_json::to_value(shorthand).unwrap(),
            serde_json::json!("tabcode:gpt-4.1")
        );

        let structured = ModelRouteStepWire::Structured(ModelRouteStepConfig::Physical {
            provider: "tabcode".into(),
            model: "gpt-4.1".into(),
            endpoint: None,
            reasoning: Some(RouteReasoningConfig {
                effort: Some("medium".into()),
                budget: None,
                level: None,
            }),
        });
        let value = serde_json::to_value(structured).unwrap();
        assert_eq!(value["type"], "physical");
        assert_eq!(value["provider"], "tabcode");
        assert_eq!(value["model"], "gpt-4.1");
        assert_eq!(value["reasoning"]["effort"], "medium");
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
                "served": ["claude-sonnet-4-6"]
            },
            "routing": {
                "model_routes": {
                    "claude-sonnet-4-6": [
                        "tabcode:gpt-4.1"
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
                .model_routes
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
                "served": ["claude-sonnet-4-6"]
            },
            "routing": {
                "model_routes": {
                    "claude-sonnet-4-6": [
                        "tabcode:gpt-4.1"
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
                "served": ["claude-sonnet-4-6"]
            },
            "routing": {
                "model_routes": {
                    "claude-sonnet-4-6": [
                        "tabcode:gpt-4.1"
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
    fn wildcard_route_can_be_logical_alias() {
        let mut cfg = base_config();
        cfg.routing.model_routes.insert(
            "*".into(),
            vec![ModelRouteStepConfig::Logical {
                model: "claude-sonnet-4-6".into(),
            }],
        );

        cfg.validate().unwrap();
        let (logical_model, targets) = cfg.route_targets_for_model("unmapped-model").unwrap();
        assert_eq!(logical_model, "claude-sonnet-4-6");
        assert_eq!(targets[0].provider, "tabcode");
        assert_eq!(targets[0].model, "gpt-4.1");
    }

    #[test]
    fn route_steps_support_physical_then_logical_fallback() {
        let mut cfg = base_config();
        cfg.routing.model_routes.insert(
            "gpt-4.1-mini".into(),
            vec![
                ModelRouteStepConfig::Physical {
                    provider: "openai".into(),
                    model: "gpt-4.1-mini".into(),
                    endpoint: None,
                    reasoning: None,
                },
                ModelRouteStepConfig::Logical {
                    model: "claude-sonnet-4-6".into(),
                },
            ],
        );
        cfg.accounts.push(AccountConfig {
            id: "openai-a".into(),
            provider: "openai".into(),
            enabled: true,
            weight: 1,
            models: Some(vec!["gpt-4.1-mini".into()]),
            auth: AccountAuth::ApiKey {
                api_key: "sk-test".into(),
            },
        });

        cfg.validate().unwrap();
        let (logical_model, targets) = cfg.route_targets_for_model("gpt-4.1-mini").unwrap();
        assert_eq!(logical_model, "gpt-4.1-mini");
        assert_eq!(targets[0].provider, "openai");
        assert_eq!(targets[0].model, "gpt-4.1-mini");
        assert_eq!(targets[1].provider, "tabcode");
        assert_eq!(targets[1].model, "gpt-4.1");
    }

    #[test]
    fn rejects_missing_logical_reference() {
        let mut cfg = base_config();
        cfg.routing.model_routes.insert(
            "*".into(),
            vec![ModelRouteStepConfig::Logical {
                model: "missing-logical".into(),
            }],
        );

        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("references undefined logical model 'missing-logical'"));
    }

    #[test]
    fn rejects_logical_cycle() {
        let mut cfg = base_config();
        cfg.models.served = vec!["a".into()];
        cfg.compaction.preferred_targets = Vec::new();
        cfg.routing.model_routes = HashMap::from([
            (
                "a".into(),
                vec![ModelRouteStepConfig::Logical { model: "b".into() }],
            ),
            (
                "b".into(),
                vec![ModelRouteStepConfig::Logical { model: "a".into() }],
            ),
        ]);

        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("logical cycle"));
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
        cfg.routing.model_routes.insert(
            "route-only-logical".into(),
            vec![ModelRouteStepConfig::Physical {
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
    fn preferred_targets_rejects_object_items() {
        let err = serde_json::from_value::<CompactionConfig>(serde_json::json!({
            "temperature": 0.1,
            "preferred_targets": [
              "glm-5-turbo",
              { "model": "claude-sonnet-4-6" }
            ]
        }))
        .unwrap_err()
        .to_string();

        assert!(err.contains("expected a string"));
    }
}
