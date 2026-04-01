# codex-proxy

[![CI](https://github.com/cornellsh/codex-proxy/workflows/CI/badge.svg)](https://github.com/cornellsh/codex-proxy/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

An OpenAI Responses API proxy for Gemini, Z.AI, and OpenAI upstreams.

It accepts OpenAI-style Responses API requests, normalizes them into a shared typed internal form, resolves logical-model to provider/account/model routing in one place, and then hands execution to the selected provider implementation.

## Features

- OpenAI-compatible `/responses` and `/responses/compact` endpoints
- Gemini, Z.AI, and OpenAI upstream providers
- Multi-account routing per provider
- Ordered preferred route targets per logical model
- Provider catalogs plus account-level optional model restrictions
- Sticky routing for KV-cache reuse on the exact resolved provider/model/account path
- Account health tracking with exponential backoff and recovery probes
- Route-scoped reasoning defaults shared across providers
- Structured config schema for providers, accounts, routing, reasoning, and compaction

## Quick start

Requires [Rust](https://www.rust-lang.org/tools/install) (edition 2024).

```bash
git clone https://github.com/cornellsh/codex-proxy.git
cd codex-proxy
cargo run --release
```

## Codex configuration

Example `~/.codex/config.toml`: see `config/codex/config.toml`.

```toml
model = "glm-5-turbo"
model_provider = "codex-proxy"
service_tier = "fast"
disable_response_storage = true
personality = "pragmatic"

[model_providers.codex-proxy]
name = "openai"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
requires_openai_auth = false
env_key = "CODEX_PROXY_API_KEY"
```

The UI-generated Codex config uses the same `CODEX_PROXY_API_KEY` pattern.

## Configuration

Configuration search order is:

1. `config/config.json.local`
2. `~/.config/codex-proxy/config.json`
3. `config/config.json`

Top-level sections:

- `server`
- `providers`
- `models`
- `routing`
- `health`
- `accounts`
- `reasoning`
- `timeouts`
- `compaction`

### Example config

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
      "default_client_secret": "...",
      "models": ["gemini-2.5-pro", "gemini-2.5-flash"]
    },
    "zai": {
      "api_url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
      "allow_authorization_passthrough": false,
      "models": ["glm-4.6"]
    },
    "openai": {
      "responses_url": "https://api.openai.com/v1/responses",
      "models": ["gpt-4.1", "gpt-4.1-mini"]
    }
  },
  "models": {
    "served": ["claude-sonnet-4-6", "claude-fast", "compact-default"],
    "fallback_models": {
      "claude-fast": "claude-sonnet-4-6"
    }
  },
  "routing": {
    "model_overrides": {
      "claude-fast": "claude-sonnet-4-6",
      "*": "claude-sonnet-4-6"
    },
    "model_provider_priority": {
      "claude-sonnet-4-6": [
        {
          "provider": "open_ai",
          "model": "gpt-4.1",
          "reasoning": { "effort": "medium" }
        },
        {
          "provider": "gemini",
          "model": "gemini-2.5-pro",
          "reasoning": { "effort": "high" }
        }
      ],
      "compact-default": [
        {
          "provider": "open_ai",
          "model": "gpt-4.1-mini",
          "reasoning": { "effort": "none" }
        }
      ]
    }
  },
  "health": {
    "auth_failure_immediate_unhealthy": true,
    "failure_threshold": 3,
    "cooldown_seconds": 60
  },
  "accounts": [
    {
      "id": "openai-primary",
      "provider": "open_ai",
      "enabled": true,
      "weight": 1,
      "models": ["gpt-4.1", "gpt-4.1-mini"],
      "auth": {
        "type": "api_key",
        "api_key": "sk-..."
      }
    },
    {
      "id": "gemini-oauth-a",
      "provider": "gemini",
      "enabled": true,
      "weight": 2,
      "auth": {
        "type": "gemini_oauth",
        "creds_path": "/Users/you/.gemini/oauth_creds.json"
      }
    }
  ],
  "auto_compaction": {
    "enabled": true,
    "max_attempts_per_request": 1,
    "tail_items_to_keep": 8
  },
  "reasoning": {
    "default_effort": "medium",
    "effort_levels": {
      "none": { "budget": 0, "level": "LOW" },
      "medium": { "budget": 16384, "level": "MEDIUM" },
      "high": { "budget": 32768, "level": "HIGH" }
    }
  },
  "timeouts": {
    "connect_seconds": 10,
    "read_seconds": 600
  },
  "compaction": {
    "temperature": 0.1,
    "preferred_targets": [
      {
        "provider": "open_ai",
        "model": "gpt-4.1-mini",
        "reasoning": { "effort": "none" }
      },
      {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "reasoning": { "effort": "minimal" }
      }
    ]
  }
}
```

### Routing model

- `routing.model_provider_priority[logical_model]` is the canonical ordered list of route targets.
- Each route target picks one upstream provider and one upstream model.
- `routing.model_overrides` maps requested models onto logical routing entries. Exact matches win first, then a literal `"*"` wildcard applies, and the resolved logical model is looked up in `routing.model_provider_priority`.
- Sticky routing is always enabled and reuses the exact chosen `(provider, model, account)` path when it is still healthy.
- If the sticky-bound path is unhealthy, routing falls through to the next compatible preferred target.
- `accounts[].models`, when omitted, means the account can use any model from that provider's catalog.
- `accounts[].weight` is used as a stable tiebreaker between otherwise equivalent compatible accounts.
- Account health settings now live in top-level `health`.
- Accounts are marked unhealthy on the first failure, back off exponentially, and only return after a recovery-probe path succeeds.

### Reasoning model

- Reusable reasoning presets live in `reasoning.effort_levels`.
- `reasoning.default_effort` is optional and acts as the fallback when a route target does not specify one.
- A route target can reference a preset with `reasoning.effort` or provide inline `budget` / `level` overrides.
- Gemini applies the selected route reasoning from execution context.
- OpenAI route reasoning is forwarded as Responses API `reasoning.effort`.
- Z.AI route reasoning is forwarded as `thinking.type` (`enabled` / `disabled`).

### Notes

- `accounts[]` is the source of truth for upstream credentials.
- Gemini supports either `api_key` or `gemini_oauth` account auth.
- Z.AI and OpenAI currently use straightforward account-scoped API-key auth.
- `routing.model_overrides` maps user-facing requested models to logical routing entries. Exact matches win first, then a literal `"*"` wildcard applies, otherwise the requested model is used unchanged.
- Compaction uses the same ordered route planning path as normal responses, but with `compaction.preferred_targets`.
- Auto-compaction prompt text is embedded in the binary; config only controls whether it runs and how aggressively it trims.
- The OpenAI provider intentionally forwards requests upstream with minimal transformation: it swaps in the resolved upstream model and configured account auth, then forwards the OpenAI-shaped payload.

## UI

Open `http://127.0.0.1:8765/config` to inspect:

- provider URLs and model catalogs
- ordered preferred routing policy
- compaction targets
- masked account auth data
- account health and recovery-probe state
- reasoning presets

## Development

```bash
cargo fmt
cargo check
cargo test
```

## License

MIT License - see [LICENSE](LICENSE)
