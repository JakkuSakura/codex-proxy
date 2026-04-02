set shell := ["zsh", "-cu"]

PROXY_CONFIG := "config/config.json.local"
PROXY_HOST := "127.0.0.1"
PROXY_PORT := "9999"
PROXY_BASE_URL := "http://{{PROXY_HOST}}:{{PROXY_PORT}}/v1"

CODEX_MODEL := "glm-5-turbo"

default: help

help:
  @echo 'Usage: just [recipe]'
  @echo ''
  @echo 'Recipes:'
  @echo '  build             Build the Rust proxy binary'
  @echo '  run               Start the proxy server (release)'
  @echo '  proxy             Start the proxy with {{PROXY_CONFIG}}'
  @echo '  codex             Start Codex against the local proxy'
  @echo '  dev               Run proxy+codex together (tmux if available)'
  @echo '  test              Run the test suite'
  @echo '  check             Run clippy + test'
  @echo '  clean             Remove build artifacts'
  @echo ''
  @echo 'Variables (override via `just VAR=value ...`):'
  @echo '  PROXY_CONFIG, PROXY_HOST, PROXY_PORT, PROXY_BASE_URL, CODEX_MODEL'

build:
  cargo build --release

run:
  cargo run --release

proxy:
  cargo run -- --config {{PROXY_CONFIG}}

run-config config_path:
  cargo run -- --config {{config_path}}

codex:
  codex -c model_provider="codex-proxy" -c model="{{CODEX_MODEL}}" -c 'model_providers.codex-proxy.base_url="{{PROXY_BASE_URL}}"'

dev:
  @if command -v tmux >/dev/null 2>&1; then \
    tmux new-session -d -s codex-proxy-dev 'just proxy'; \
    tmux split-window -h 'just codex'; \
    tmux select-pane -t 0; \
    tmux attach -t codex-proxy-dev; \
  else \
    echo 'tmux not found. Run these in two terminals:'; \
    echo '  just proxy'; \
    echo '  just codex'; \
  fi

test:
  cargo test

check:
  cargo clippy -- -D warnings
  cargo test

clean:
  cargo clean

