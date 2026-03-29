use codex_proxy::server::{build_router, print_startup_info};
use tracing::info;
use tracing_subscriber::fmt::format::FmtSpan;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("CODEX_PROXY_LOG_LEVEL")
                .unwrap_or_else(|_| "codex_proxy=debug".parse().unwrap()),
        )
        .with_span_events(FmtSpan::NONE)
        .with_target(true)
        .init();

    info!("Starting codex-proxy...");

    let config = &codex_proxy::config::CONFIG;
    let addr = format!("{}:{}", config.host, config.port);

    print_startup_info();

    let app = build_router();
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("Failed to bind");
    info!("Server bound to {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Server error");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Shutting down...");
}
