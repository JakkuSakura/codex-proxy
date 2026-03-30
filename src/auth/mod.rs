use crate::account_pool::{Account, AccountAuth};
use crate::config::CONFIG;
use crate::error::ProxyError;
use crate::schema::json_value::JsonValue;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

pub static GEMINI_AUTH_MANAGER: Lazy<GeminiAuthManager> = Lazy::new(GeminiAuthManager::new);

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub api_key: Option<String>,
    pub access_token: Option<String>,
    pub project_id: Option<String>,
    pub auth_type: AuthType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AuthType {
    Public,
    Internal,
}

#[derive(Debug, Serialize, Deserialize)]
struct OAuthCreds {
    #[serde(rename = "type")]
    creds_type: Option<String>,
    access_token: Option<String>,
    expiry_date: Option<u64>,
    refresh_token: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OAuthRefreshResponse {
    access_token: Option<String>,
    expires_in: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct MetadataTokenResponse {
    access_token: Option<String>,
    expires_in: Option<u64>,
}

#[derive(Debug, Serialize)]
struct GeminiMetadata {
    #[serde(rename = "ideType")]
    ide_type: &'static str,
    platform: &'static str,
    #[serde(rename = "pluginType")]
    plugin_type: &'static str,
}

#[derive(Debug, Serialize)]
struct LoadCodeAssistRequest {
    metadata: GeminiMetadata,
}

#[derive(Debug, Deserialize)]
struct LoadCodeAssistResponse {
    #[serde(rename = "allowedTiers", default)]
    allowed_tiers: Vec<AllowedTier>,
    #[serde(rename = "cloudaicompanionProject", default)]
    cloudaicompanion_project: Option<ProjectField>,
}

#[derive(Debug, Deserialize)]
struct AllowedTier {
    id: String,
    #[serde(rename = "isDefault", default)]
    is_default: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ProjectField {
    Id(String),
    Obj(ProjectObj),
}

#[derive(Debug, Deserialize)]
struct ProjectObj {
    #[serde(default)]
    id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OnboardUserRequest<'a> {
    #[serde(rename = "tierId")]
    tier_id: &'a str,
    metadata: GeminiMetadata,
}

#[derive(Debug, Deserialize)]
struct OnboardUserOperation {
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PollOperationResponse {
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<JsonValue>,
    #[serde(default)]
    response: Option<PollOperationSuccess>,
}

#[derive(Debug, Deserialize)]
struct PollOperationSuccess {
    #[serde(rename = "cloudaicompanionProject", default)]
    cloudaicompanion_project: Option<ProjectObj>,
}

struct TokenCache {
    token: String,
    expiry_ms: u64,
}

pub struct GeminiAuthManager {
    clients: Mutex<HashMap<String, Arc<GeminiAuth>>>,
}

impl GeminiAuthManager {
    pub fn new() -> Self {
        Self {
            clients: Mutex::new(HashMap::new()),
        }
    }

    pub async fn get_auth_context(
        &self,
        account: &Account,
        force_refresh: bool,
    ) -> Result<AuthContext, ProxyError> {
        let auth = {
            let mut clients = self.clients.lock().unwrap();
            clients
                .entry(account.id.clone())
                .or_insert_with(|| Arc::new(GeminiAuth::new(account.clone())))
                .clone()
        };
        auth.get_auth_context(force_refresh).await
    }
}

pub struct GeminiAuth {
    account: Account,
    client: reqwest::Client,
    cache: Mutex<Option<TokenCache>>,
    project_id_cache: Mutex<Option<String>>,
}

impl GeminiAuth {
    pub fn new(account: Account) -> Self {
        Self {
            account,
            client: reqwest::Client::new(),
            cache: Mutex::new(None),
            project_id_cache: Mutex::new(None),
        }
    }

    pub async fn get_auth_context(&self, force_refresh: bool) -> Result<AuthContext, ProxyError> {
        match &self.account.auth {
            AccountAuth::ApiKey { api_key } => Ok(AuthContext {
                api_key: Some(api_key.clone()),
                access_token: None,
                project_id: None,
                auth_type: AuthType::Public,
            }),
            AccountAuth::GeminiOAuth { .. } => {
                let token = self.get_access_token(force_refresh).await?;
                let pid = self.get_project_id(&token).await?;
                Ok(AuthContext {
                    api_key: None,
                    access_token: Some(token),
                    project_id: Some(pid),
                    auth_type: AuthType::Internal,
                })
            }
        }
    }

    async fn get_access_token(&self, force_refresh: bool) -> Result<String, ProxyError> {
        if let Ok(env_token) = std::env::var("GOOGLE_CLOUD_ACCESS_TOKEN") {
            return Ok(env_token);
        }
        if !force_refresh
            && let Some(cached) = self.cache.lock().unwrap().as_ref()
            && is_token_valid(cached.expiry_ms)
        {
            return Ok(cached.token.clone());
        }
        if let Some(token) = self.try_load_from_files(force_refresh).await? {
            return Ok(token);
        }
        if let Some(token) = self.try_metadata_server().await? {
            return Ok(token);
        }
        Err(ProxyError::Auth(
            "Could not find valid Gemini credentials. Please login using 'gemini login'.".into(),
        ))
    }

    async fn try_load_from_files(&self, force_refresh: bool) -> Result<Option<String>, ProxyError> {
        let mut paths: Vec<PathBuf> = Vec::new();
        if let AccountAuth::GeminiOAuth { creds_path, .. } = &self.account.auth
            && let Some(path) = creds_path.clone()
        {
            paths.push(path);
        }
        if let Ok(cred_path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            paths.push(PathBuf::from(cred_path));
        }
        for path in paths {
            if !path.exists() {
                continue;
            }
            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    debug!("Failed to read {}: {}", path.display(), e);
                    continue;
                }
            };
            let creds: OAuthCreds = match serde_json::from_str(&content) {
                Ok(c) => c,
                Err(e) => {
                    debug!("Failed to parse {}: {}", path.display(), e);
                    continue;
                }
            };
            if creds.creds_type.as_deref() == Some("authorized_user")
                || creds.creds_type.as_deref() == Some("service_account")
            {
                if !force_refresh
                    && let (Some(token), Some(expiry)) =
                        (creds.access_token.as_ref(), creds.expiry_date)
                    && is_token_valid(expiry)
                {
                    let mut cache = self.cache.lock().unwrap();
                    *cache = Some(TokenCache {
                        token: token.clone(),
                        expiry_ms: expiry,
                    });
                    return Ok(Some(token.clone()));
                }
                if creds.refresh_token.is_some() {
                    return Ok(Some(self.refresh_token(&creds, &path).await?));
                }
            }
        }
        Ok(None)
    }

    async fn refresh_token(&self, creds: &OAuthCreds, path: &Path) -> Result<String, ProxyError> {
        info!(
            "Refreshing access token for account {} from {}",
            self.account.id,
            path.display()
        );
        let provider_cfg = CONFIG
            .gemini_provider_config(&self.account.provider)
            .map_err(ProxyError::Config)?;
        let (default_client_id, default_client_secret) = match &self.account.auth {
            AccountAuth::GeminiOAuth {
                client_id,
                client_secret,
                ..
            } => (
                client_id
                    .clone()
                    .unwrap_or_else(|| provider_cfg.default_client_id.clone()),
                client_secret
                    .clone()
                    .unwrap_or_else(|| provider_cfg.default_client_secret.clone()),
            ),
            AccountAuth::ApiKey { .. } => unreachable!(),
        };
        let client_id = creds.client_id.as_deref().unwrap_or(&default_client_id);
        let client_secret = creds
            .client_secret
            .as_deref()
            .unwrap_or(&default_client_secret);
        let refresh_token = creds
            .refresh_token
            .as_deref()
            .ok_or_else(|| ProxyError::Auth("Missing refresh_token".into()))?;
        let resp = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&[
                ("client_id", client_id),
                ("client_secret", client_secret),
                ("refresh_token", refresh_token),
                ("grant_type", "refresh_token"),
            ])
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ProxyError::Auth(format!(
                "OAuth2 refresh failed ({}): {}",
                status, body
            )));
        }
        let new_tokens: OAuthRefreshResponse = resp.json().await?;
        let access_token = new_tokens
            .access_token
            .ok_or_else(|| ProxyError::Auth("No access_token in refresh response".into()))?;
        let expires_in = new_tokens.expires_in.unwrap_or(3600);
        let expiry_ms = now_ms() + expires_in * 1000;
        let mut cache = self.cache.lock().unwrap();
        *cache = Some(TokenCache {
            token: access_token.clone(),
            expiry_ms,
        });
        if let Ok(content) = fs::read_to_string(path)
            && let Ok(mut data) = serde_json::from_str::<OAuthCreds>(&content)
        {
            data.access_token = Some(access_token.clone());
            data.expiry_date = Some(expiry_ms);
            if let Ok(serialized) = serde_json::to_string_pretty(&data)
                && let Err(e) = fs::write(path, serialized)
            {
                warn!(
                    "Could not save refreshed tokens to {}: {}",
                    path.display(),
                    e
                );
            }
        }
        Ok(access_token)
    }

    async fn try_metadata_server(&self) -> Result<Option<String>, ProxyError> {
        let resp = self
            .client
            .get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
            .header("Metadata-Flavor", "Google")
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => {
                let data: MetadataTokenResponse = r.json().await?;
                if let Some(token) = data.access_token {
                    let expires_in = data.expires_in.unwrap_or(3600);
                    let mut cache = self.cache.lock().unwrap();
                    *cache = Some(TokenCache {
                        token: token.clone(),
                        expiry_ms: now_ms() + expires_in * 1000,
                    });
                    return Ok(Some(token));
                }
            }
            _ => {}
        }
        Ok(None)
    }

    async fn get_project_id(&self, token: &str) -> Result<String, ProxyError> {
        if let Some(pid) = self.project_id_cache.lock().unwrap().as_ref() {
            return Ok(pid.clone());
        }
        if let Some(pid) = std::env::var("GOOGLE_CLOUD_PROJECT")
            .ok()
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT_ID").ok())
        {
            *self.project_id_cache.lock().unwrap() = Some(pid.clone());
            return Ok(pid);
        }
        let project_info = self.fetch_project_info(token).await?;
        let pid = match &project_info.cloudaicompanion_project {
            Some(ProjectField::Id(s)) => Some(s.clone()),
            Some(ProjectField::Obj(o)) => o.id.clone(),
            None => None,
        };
        if let Some(pid) = pid {
            *self.project_id_cache.lock().unwrap() = Some(pid.clone());
            return Ok(pid);
        }
        let tier_id = determine_tier(&project_info);
        let pid = self.onboard_user(token, &tier_id).await?;
        *self.project_id_cache.lock().unwrap() = Some(pid.clone());
        Ok(pid)
    }

    async fn fetch_project_info(&self, token: &str) -> Result<LoadCodeAssistResponse, ProxyError> {
        let url = format!(
            "{}/v1internal:loadCodeAssist",
            CONFIG
                .gemini_provider_config(&self.account.provider)
                .map_err(ProxyError::Config)?
                .api_internal
        );
        let body = LoadCodeAssistRequest {
            metadata: default_metadata(),
        };
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("User-Agent", "GeminiCLI/0.26.0 (linux; x64)")
            .json(&body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }

    async fn onboard_user(&self, token: &str, tier_id: &str) -> Result<String, ProxyError> {
        let url = format!(
            "{}/v1internal:onboardUser",
            CONFIG
                .gemini_provider_config(&self.account.provider)
                .map_err(ProxyError::Config)?
                .api_internal
        );
        let body = OnboardUserRequest {
            tier_id,
            metadata: default_metadata(),
        };
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("User-Agent", "GeminiCLI/0.26.0 (linux; x64)")
            .json(&body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?
            .error_for_status()?;
        let operation: OnboardUserOperation = resp.json().await?;
        let op_name = operation
            .name
            .as_deref()
            .ok_or_else(|| ProxyError::Auth("Onboarding failed: no operation name".into()))?;
        let result = self.poll_operation(token, op_name).await?;
        let pid = result
            .response
            .and_then(|r| r.cloudaicompanion_project)
            .and_then(|p| p.id)
            .ok_or_else(|| {
                ProxyError::Auth("Onboarding finished but no Project ID found".into())
            })?;
        Ok(pid)
    }

    async fn poll_operation(
        &self,
        token: &str,
        op_name: &str,
    ) -> Result<PollOperationResponse, ProxyError> {
        let url = format!(
            "{}/v1internal/{}",
            CONFIG
                .gemini_provider_config(&self.account.provider)
                .map_err(ProxyError::Config)?
                .api_internal,
            op_name
        );
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > std::time::Duration::from_secs(60) {
                return Err(ProxyError::Auth("Operation timed out".into()));
            }
            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {token}"))
                .timeout(std::time::Duration::from_secs(10))
                .send()
                .await?
                .error_for_status()?;
            let data: PollOperationResponse = resp.json().await?;
            if data.done {
                if let Some(err) = &data.error {
                    return Err(ProxyError::Auth(format!(
                        "Operation failed: {}",
                        serde_json::to_string(err).unwrap_or_default()
                    )));
                }
                return Ok(data);
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }
}

fn determine_tier(project_info: &LoadCodeAssistResponse) -> String {
    for tier in &project_info.allowed_tiers {
        if tier.is_default {
            return tier.id.clone();
        }
    }
    "free-tier".into()
}

fn default_metadata() -> GeminiMetadata {
    GeminiMetadata {
        ide_type: "IDE_UNSPECIFIED",
        platform: "PLATFORM_UNSPECIFIED",
        plugin_type: "GEMINI",
    }
}

fn is_token_valid(expiry_ms: u64) -> bool {
    if expiry_ms == 0 {
        return true;
    }
    now_ms() < expiry_ms.saturating_sub(300_000)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
