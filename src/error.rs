use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Provider error: {0}")]
    Provider(ProviderError),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("{0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    Auth,
    Client,
    Server,
    Network,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ProviderError {
    pub status: Option<StatusCode>,
    pub code: Option<String>,
    pub error_type: Option<String>,
    pub message: String,
}

impl ProviderError {
    pub fn new(status: Option<StatusCode>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: None,
            error_type: None,
            message: message.into(),
        }
    }

    pub fn with_details(
        status: Option<StatusCode>,
        message: impl Into<String>,
        code: Option<String>,
        error_type: Option<String>,
    ) -> Self {
        Self {
            status,
            code,
            error_type,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ProviderErrorKind {
        match self.status {
            Some(StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => ProviderErrorKind::Auth,
            Some(status) if status.is_client_error() => ProviderErrorKind::Client,
            Some(status) if status.is_server_error() => ProviderErrorKind::Server,
            Some(_) => ProviderErrorKind::Unknown,
            None => ProviderErrorKind::Network,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut details = Vec::new();
        if let Some(code) = &self.code {
            details.push(format!("code={code}"));
        }
        if let Some(error_type) = &self.error_type {
            details.push(format!("type={error_type}"));
        }
        match (self.status, details.is_empty()) {
            (Some(status), false) => write!(f, "{} (status={}, {})", self.message, status, details.join(", ")),
            (Some(status), true) => write!(f, "{} (status={})", self.message, status),
            (None, false) => write!(f, "{} ({})", self.message, details.join(", ")),
            (None, true) => write!(f, "{}", self.message),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid port: {0}")]
    InvalidPort(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Invalid model prefix: {0}")]
    InvalidPrefix(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, code, message) = (self.status_code(), self.error_code(), self.to_string());

        #[derive(Serialize)]
        struct ErrorEnvelope {
            error: ErrorBody,
        }

        #[derive(Serialize)]
        struct ErrorBody {
            code: &'static str,
            message: String,
        }

        (
            status,
            axum::Json(ErrorEnvelope {
                error: ErrorBody { code, message },
            }),
        )
            .into_response()
    }
}

impl ProxyError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            ProxyError::Validation(_) => StatusCode::BAD_REQUEST,
            ProxyError::Provider(err) => err.status.unwrap_or(StatusCode::BAD_GATEWAY),
            ProxyError::NotImplemented(_) => StatusCode::NOT_IMPLEMENTED,
            ProxyError::Auth(_) => StatusCode::UNAUTHORIZED,
            ProxyError::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::Http(e) => e
                .status()
                .and_then(|s| StatusCode::from_u16(s.as_u16()).ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            ProxyError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            ProxyError::Validation(_) => "bad_request",
            ProxyError::Provider(_) => "provider_error",
            ProxyError::NotImplemented(_) => "not_implemented",
            ProxyError::Auth(_) => "auth_error",
            ProxyError::Config(_) => "config_error",
            ProxyError::Http(_) => "http_error",
            ProxyError::Internal(_) => "internal_server_error",
        }
    }

    pub fn provider_kind(&self) -> Option<ProviderErrorKind> {
        match self {
            ProxyError::Provider(err) => Some(err.kind()),
            _ => None,
        }
    }

    pub fn provider_message(&self) -> Option<&str> {
        match self {
            ProxyError::Provider(err) => Some(err.message.as_str()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_preserves_status() {
        let err = ProxyError::Provider(ProviderError::new(
            Some(StatusCode::BAD_REQUEST),
            "bad request",
        ));
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn provider_error_kind_from_status() {
        let err = ProviderError::new(Some(StatusCode::FORBIDDEN), "nope");
        assert_eq!(err.kind(), ProviderErrorKind::Auth);
        let err = ProviderError::new(Some(StatusCode::BAD_REQUEST), "bad");
        assert_eq!(err.kind(), ProviderErrorKind::Client);
        let err = ProviderError::new(Some(StatusCode::BAD_GATEWAY), "bad");
        assert_eq!(err.kind(), ProviderErrorKind::Server);
    }
}
