# codex-proxy

[![CI](https://github.com/cornellsh/codex-proxy/workflows/CI/badge.svg)](https://github.com/cornellsh/codex-proxy/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

An OpenAI Responses API proxy for Gemini, Z.AI, and OpenAI upstreams.

It accepts OpenAI-style Responses API requests, normalizes them into a shared typed internal form, resolves model/provider/account routing in one place, and then hands execution to the selected provider implementation.

## Features

- OpenAI-compatible `/responses` and `/responses/compact` endpoints
- Gemini, Z.AI, and OpenAI upstream providers
- Multi-account routing per provider
- Shared model overrides and provider-prefix routing
- Sticky routing for KV-cache reuse
- Account health tracking with cooldown-based recovery
- Structured v2 config schema with legacy flat-config startup migration

## Quick start

Requires [Rust](https://www.rust-lang.org/tools/install) (edition 2024).

```bash
git clone https://github.com/cornellsh/codex-proxy.git
cd codex-proxy
cargo run --release
```

## Codex configuration

Example `~/.codex/config.toml`:

```toml
model = "gpt-4.1"
model_provider = "codex-proxy"
personality = "pragmatic"
service_tier = "fast"

[model_providers.codex-proxy]
name = "openai"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
api_key = "dummy"
requires_openai_auth = false
```

## Configuration

Configuration lives at `~/.config/codex-proxy/config.json`.

The runtime config schema is structured around:

- `server`
- `providers`
- `models`
- `routing`
- `accounts`
- `reasoning`
- `timeouts`
- `compaction`

Legacy flat config is still accepted only at startup migration time. The process logs a warning and converts it into the internal v2 shape.

### Example v2 config

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "log_level": "INFO",
    "debug_mode": false
  },
  "providers": {
    "gemini": {
      "api_internal": "https://cloudcode-pa.googleapis.com",
      "api_public": "https://generativelanguage.googleapis.com",
      "default_client_id": "...",
      "default_client_secret": "..."
    },
    "zai": {
      "api_url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
      "allow_authorization_passthrough": false
    },
    "openai": {
      "responses_url": "https://api.openai.com/v1/responses"
    }
  },
  "models": {
    "served": ["gpt-4.1", "gemini-2.5-pro", "glm-4.6"],
    "compaction_model": "gpt-4.1-mini",
    "fallback_models": {
      "gpt-5": "gpt-4.1",
      "gemini-3-pro-preview": "gemini-2.5-pro"
    }
  },
  "routing": {
    "model_overrides": {
      "claude-sonnet-4-6": "gpt-4.1",
      "gemini-fast": "gemini-2.5-flash",
      "glm-fast": "glm-4.6"
    },
    "provider_prefixes": {
      "gpt": "open_ai",
      "o": "open_ai",
      "gemini": "gemini",
      "glm": "zai",
      "zai": "zai"
    },
    "sticky_routing": {
      "enabled": true
    },
    "health": {
      "auth_failure_immediate_unhealthy": true,
      "failure_threshold": 3,
      "cooldown_seconds": 300
    }
  },
  "accounts": [
    {
      "id": "openai-primary",
      "provider": "open_ai",
      "enabled": true,
      "weight": 1,
      "auth": {
        "type": "api_key",
        "api_key": "sk-..."
      }
    },
    {
      "id": "gemini-oauth-a",
      "provider": "gemini",
      "enabled": true,
      "weight": 1,
      "auth": {
        "type": "gemini_oauth",
        "creds_path": "/Users/you/.gemini/oauth_creds.json"
      }
    },
    {
      "id": "zai-primary",
      "provider": "zai",
      "enabled": true,
      "weight": 1,
      "auth": {
        "type": "api_key",
        "api_key": "..."
      }
    }
  ],
  "reasoning": {
    "default_effort": "medium",
    "effort_levels": {
      "medium": { "budget": 16384, "level": "MEDIUM" }
    }
  },
  "timeouts": {
    "connect_seconds": 10,
    "read_seconds": 600
  },
  "compaction": {
    "temperature": 0.1
  }
}
```

### Notes

- `accounts[]` is the source of truth for upstream credentials.
- Gemini supports either `api_key` or `gemini_oauth` account auth.
- Z.AI and OpenAI currently use straightforward account-scoped API-key auth.
- `routing.model_overrides` runs before provider resolution.
- Compaction uses the same shared routing path as normal responses.
- The OpenAI provider intentionally forwards requests upstream with minimal transformation: it swaps in the resolved upstream model and configured account auth, then forwards the OpenAI-shaped payload.

## UI

Open `http://127.0.0.1:8765/config` to inspect:

- structured config snapshot
- masked account auth data
- account health
- sticky routing stats

## Development

```bash
cargo fmt
cargo check
cargo test
```

## License

MIT License - see [LICENSE](LICENSE)
