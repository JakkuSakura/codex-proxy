use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("{0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid port: {0}")]
    InvalidPort(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Invalid model prefix: {0}")]
    InvalidPrefix(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            ProxyError::Validation(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg.clone()),
            ProxyError::Provider(msg) => (StatusCode::BAD_GATEWAY, "provider_error", msg.clone()),
            ProxyError::Auth(msg) => (StatusCode::UNAUTHORIZED, "auth_error", msg.clone()),
            ProxyError::Config(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "config_error",
                msg.to_string(),
            ),
            ProxyError::Http(e) => {
                let status = e
                    .status()
                    .map(|s| {
                        let u16_val: u16 = s.as_u16();
                        StatusCode::from_u16(u16_val).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                    })
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                (status, "http_error", e.to_string())
            }
            ProxyError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_server_error",
                msg.clone(),
            ),
        };

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
