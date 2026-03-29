# codex-proxy

[![CI](https://github.com/cornellsh/codex-proxy/workflows/CI/badge.svg)](https://github.com/cornellsh/codex-proxy/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**An OpenAI Responses API proxy for Gemini and Z.AI (GLM) providers.**

Translates OpenAI's Responses API to Gemini and Z.AI APIs. Handles wire format differences, role mapping, and SSE stream formatting so Codex can use these providers instead of GPT.

## Features

- **Responses API** - Full lifecycle with SSE events
- **Multi-Provider** - Gemini (OAuth2) and Z.AI (GLM) support
- **Context Compaction** - Both Gemini and Z.AI models support
- **Tool Support** - Function calling and web search

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
git clone https://github.com/cornellsh/codex-proxy.git
cd codex-proxy

# Install dependencies
uv sync

# Start the proxy
uv run python -m codex_proxy
# or
uv run codex-proxy
```

## Codex Configuration

Example `~/.codex/config.toml` for using codex-proxy:

```toml
model = "glm-5"
model_provider = "codex-proxy"
personality = "pragmatic"
service_tier = "fast"

[model_providers.codex-proxy]
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
requires_openai_auth = false
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Or use make
make test
make lint
make format
```

## Configuration

Configuration lives at `~/.config/codex-proxy/config.json`. Environment variables override all settings.

### Server Settings

| Env Var | Config Key | Description | Default |
|----------|-------------|-------------|----------|
| `CODEX_PROXY_PORT` | `port` | Port to listen on | `8765` |
| `CODEX_PROXY_LOG_LEVEL` | `log_level` | Logging level (DEBUG, INFO, WARNING, ERROR) | `DEBUG` |
| `CODEX_PROXY_DEBUG` | `debug_mode` | Enable debug mode (logs raw requests) | `true` |

### Authentication Settings

| Env Var | Config Key | Description | Default |
|----------|-------------|-------------|----------|
| `CODEX_PROXY_ZAI_API_KEY` | `z_ai_api_key` | Z.AI API key | - |
| `CODEX_PROXY_GEMINI_API_KEY` | `gemini_api_key` | Google AI Studio API key | - |
| `CODEX_PROXY_GEMINI_CLIENT_ID` | `client_id` | Override OAuth client ID | Built-in |
| `CODEX_PROXY_GEMINI_CLIENT_SECRET` | `client_secret` | Override OAuth client secret | Built-in |

### API Endpoints

| Env Var | Config Key | Description | Default |
|----------|-------------|-------------|----------|
| `CODEX_PROXY_ZAI_URL` | `z_ai_url` | Z.AI endpoint | `https://api.z.ai/api/coding/paas/v4/chat/completions` |
| `CODEX_PROXY_GEMINI_API_INTERNAL` | `gemini_api_internal` | Gemini internal API | `https://cloudcode-pa.googleapis.com` |
| `CODEX_PROXY_GEMINI_API_PUBLIC` | `gemini_api_public` | Gemini public API | `https://generativelanguage.googleapis.com` |

### Model Settings

| Config Key | Description | Default |
|------------|-------------|----------|
| `models` | Available models (any model accepted if empty) | `[]` |
| `compaction_model` | Model for context compaction | First configured model |
| `fallback_models` | Map model names to fallbacks | `{}` |
| `model_prefixes` | Map model prefixes to providers | `{"gemini": "gemini", "glm": "zai", "zai": "zai"}` |

### Reasoning Settings

| Config Key | Description | Default |
|------------|-------------|----------|
| `reasoning_effort` | Default reasoning effort level | `medium` |
| `reasoning.effort_levels` | Custom effort levels with budgets | Pre-configured levels |

### Timeout Settings

| Config Key | Description | Default |
|------------|-------------|----------|
| `request_timeout_connect` | Connection timeout (seconds) | `10` |
| `request_timeout_read` | Read timeout (seconds) | `600` |
| `compaction_temperature` | Temperature for compaction requests | `0.1` |

### Example Config

```json
{
  "port": 8765,
  "log_level": "INFO",
  "z_ai_api_key": "your-z-ai-key",
  "gemini_api_key": "your-google-ai-studio-key",
  "compaction_model": "glm-4.6",
  "fallback_models": {
    "gemini-3-pro-preview": "gemini-2.5-flash",
    "glm-4.7": "glm-4.6"
  },
  "reasoning": {
    "default_effort": "medium"
  },
  "model_prefixes": {
    "custom-model": "gemini"
  }
}
```

### Available Reasoning Effort Levels

`none`, `minimal`, `low`, `medium`, `high`, `xhigh`. You can customize budgets and levels in `reasoning.effort_levels` config if needed.

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide and contribution process
- [AGENTS.md](AGENTS.md) - Deep technical context for AI assistants

## License

MIT License - see [LICENSE](LICENSE)
