use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::ConfigError;

pub const GEMINI_CLI_CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";
pub const GEMINI_CLI_CLIENT_SECRET: &str = "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";

fn validate_port(port_str: &str) -> Result<u16, ConfigError> {
    let port: u16 = port_str.parse().map_err(|_| ConfigError::InvalidPort(port_str.into()))?;
    if port == 0 {
        return Err(ConfigError::InvalidPort("port must be 1-65535".into()));
    }
    Ok(port)
}

fn validate_url(url: &str, name: &str) -> Result<String, ConfigError> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(ConfigError::InvalidUrl(format!("{name} must start with http:// or https://")));
    }
    Ok(url.to_string())
}

fn validate_model_prefix(prefix: &str) -> Result<String, ConfigError> {
    if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
        return Err(ConfigError::InvalidPrefix(format!("model prefix must be lowercase alphanumeric: {prefix}")));
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
        effort_levels.insert("none".into(), EffortLevel { budget: 0, level: "LOW".into() });
        effort_levels.insert("minimal".into(), EffortLevel { budget: 2048, level: "LOW".into() });
        effort_levels.insert("low".into(), EffortLevel { budget: 4096, level: "LOW".into() });
        effort_levels.insert("medium".into(), EffortLevel { budget: 16384, level: "MEDIUM".into() });
        effort_levels.insert("high".into(), EffortLevel { budget: 32768, level: "HIGH".into() });
        effort_levels.insert("xhigh".into(), EffortLevel { budget: 65536, level: "HIGH".into() });
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
pub struct FileConfig {
    pub port: Option<u16>,
    pub log_level: Option<String>,
    pub debug_mode: Option<bool>,
    pub z_ai_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub models: Option<Vec<String>>,
    pub compaction_model: Option<String>,
    pub fallback_models: Option<HashMap<String, String>>,
    pub model_prefixes: Option<HashMap<String, String>>,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<ReasoningConfig>,
    pub z_ai_url: Option<String>,
    pub gemini_api_internal: Option<String>,
    pub gemini_api_public: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub gemini_creds_path: PathBuf,
    pub config_path: PathBuf,
    pub z_ai_url: String,
    pub gemini_api_internal: String,
    pub gemini_api_public: String,
    pub models: Vec<String>,
    pub compaction_model: Option<String>,
    pub fallback_models: HashMap<String, String>,
    pub model_prefixes: HashMap<String, String>,
    pub reasoning_effort: String,
    pub reasoning: ReasoningConfig,
    pub request_timeout_connect: u64,
    pub request_timeout_read: u64,
    pub compaction_temperature: f64,
    pub client_id: String,
    pub client_secret: String,
    pub z_ai_api_key: String,
    pub gemini_api_key: String,
    pub log_level: String,
    pub debug_mode: bool,
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
        cfg
    }

    fn defaults() -> Self {
        let port = env::var("CODEX_PROXY_PORT")
            .map(|p| validate_port(&p).unwrap_or(8765))
            .unwrap_or(8765);

        let z_ai_url = env::var("CODEX_PROXY_ZAI_URL")
            .unwrap_or_else(|_| "https://api.z.ai/api/coding/paas/v4/chat/completions".into());

        let gemini_api_internal = env::var("CODEX_PROXY_GEMINI_API_INTERNAL")
            .unwrap_or_else(|_| "https://cloudcode-pa.googleapis.com".into());

        let gemini_api_public = env::var("CODEX_PROXY_GEMINI_API_PUBLIC")
            .unwrap_or_else(|_| "https://generativelanguage.googleapis.com".into());

        let mut model_prefixes = HashMap::new();
        model_prefixes.insert("gemini".into(), "gemini".into());
        model_prefixes.insert("glm".into(), "zai".into());
        model_prefixes.insert("zai".into(), "zai".into());

        let models: Vec<String> = env::var("CODEX_PROXY_MODELS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let log_level = env::var("CODEX_PROXY_LOG_LEVEL")
            .unwrap_or_else(|_| "DEBUG".into())
            .to_uppercase();

        let debug_mode = env::var("CODEX_PROXY_DEBUG")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(true);

        let home = dirs_home();

        Self {
            host: "127.0.0.1".into(),
            port,
            gemini_creds_path: home.join(".gemini/oauth_creds.json"),
            config_path: home.join(".config/codex-proxy/config.json"),
            z_ai_url: validate_url(&z_ai_url, "Z.AI URL").unwrap(),
            gemini_api_internal: validate_url(&gemini_api_internal, "Gemini internal").unwrap(),
            gemini_api_public: validate_url(&gemini_api_public, "Gemini public").unwrap(),
            models,
            compaction_model: None,
            fallback_models: HashMap::new(),
            model_prefixes,
            reasoning_effort: "medium".into(),
            reasoning: ReasoningConfig::default(),
            request_timeout_connect: 10,
            request_timeout_read: 600,
            compaction_temperature: 0.1,
            client_id: env::var("CODEX_PROXY_GEMINI_CLIENT_ID")
                .unwrap_or_else(|_| GEMINI_CLI_CLIENT_ID.into()),
            client_secret: env::var("CODEX_PROXY_GEMINI_CLIENT_SECRET")
                .unwrap_or_else(|_| GEMINI_CLI_CLIENT_SECRET.into()),
            z_ai_api_key: env::var("CODEX_PROXY_ZAI_API_KEY").unwrap_or_default(),
            gemini_api_key: env::var("CODEX_PROXY_GEMINI_API_KEY").unwrap_or_default(),
            log_level,
            debug_mode,
        }
    }

    fn load_from_file(&mut self) {
        if !self.config_path.exists() {
            return;
        }
        let content = match fs::read_to_string(&self.config_path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to read config {}: {}", self.config_path.display(), e);
                return;
            }
        };
        let file_cfg: FileConfig = match serde_json::from_str(&content) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to parse config {}: {}", self.config_path.display(), e);
                return;
            }
        };

        if env::var("CODEX_PROXY_PORT").is_err()
            && let Some(port) = file_cfg.port { self.port = port; }
        if env::var("CODEX_PROXY_LOG_LEVEL").is_err()
            && let Some(ref level) = file_cfg.log_level { self.log_level = level.to_uppercase(); }
        if let Some(debug) = file_cfg.debug_mode { self.debug_mode = debug; }
        if env::var("CODEX_PROXY_ZAI_API_KEY").is_err()
            && let Some(ref key) = file_cfg.z_ai_api_key { self.z_ai_api_key = key.clone(); }
        if env::var("CODEX_PROXY_GEMINI_API_KEY").is_err()
            && let Some(ref key) = file_cfg.gemini_api_key { self.gemini_api_key = key.clone(); }
        if env::var("CODEX_PROXY_GEMINI_CLIENT_ID").is_err()
            && let Some(ref id) = file_cfg.client_id { self.client_id = id.clone(); }
        if env::var("CODEX_PROXY_GEMINI_CLIENT_SECRET").is_err()
            && let Some(ref secret) = file_cfg.client_secret { self.client_secret = secret.clone(); }
        if env::var("CODEX_PROXY_MODELS").is_err()
            && let Some(ref models) = file_cfg.models { self.models = models.clone(); }
        if let Some(ref compact) = file_cfg.compaction_model && !compact.is_empty() { self.compaction_model = Some(compact.clone()); }
        if let Some(ref fallbacks) = file_cfg.fallback_models { self.fallback_models = fallbacks.clone(); }
        if let Some(ref prefixes) = file_cfg.model_prefixes {
            for (prefix, provider_key) in prefixes {
                if let Ok(p) = validate_model_prefix(prefix) {
                    self.model_prefixes.insert(p, provider_key.clone());
                }
            }
        }
        if let Some(ref effort) = file_cfg.reasoning_effort { self.reasoning_effort = effort.clone(); }
        if let Some(ref reasoning) = file_cfg.reasoning { self.reasoning = reasoning.clone(); }
        if env::var("CODEX_PROXY_ZAI_URL").is_err()
            && let Some(ref url) = file_cfg.z_ai_url
                && let Ok(u) = validate_url(url, "Z.AI URL") { self.z_ai_url = u; }
        if env::var("CODEX_PROXY_GEMINI_API_INTERNAL").is_err()
            && let Some(ref url) = file_cfg.gemini_api_internal
                && let Ok(u) = validate_url(url, "Gemini internal") { self.gemini_api_internal = u; }
        if env::var("CODEX_PROXY_GEMINI_API_PUBLIC").is_err()
            && let Some(ref url) = file_cfg.gemini_api_public
                && let Ok(u) = validate_url(url, "Gemini public") { self.gemini_api_public = u; }

        info!("Loaded config from {}", self.config_path.display());
    }
}

fn dirs_home() -> PathBuf {
    env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/root"))
}
