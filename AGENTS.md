# Codex Proxy Context

## Overview

`codex-proxy` is a Rust service that exposes an OpenAI-compatible Responses API while routing requests to multiple upstream providers. The live architecture is provider-aware and account-aware: model resolution, sticky routing, account health, and provider selection are shared runtime concerns, while protocol translation stays inside provider implementations.

## Current Architecture

### Request flow

The main request path lives in `src/server.rs` and follows this sequence:

1. Validate incoming OpenAI-style request payloads.
2. Normalize `ResponsesRequest` payloads into the internal typed `ChatRequest` form.
3. Resolve routing in one place:
   - apply model overrides
   - infer provider from configured model prefixes
   - compute sticky-routing key from normalized message content
   - choose a healthy account for that provider
4. Dispatch to the selected provider implementation.
5. Apply success/failure health updates and sticky binding updates after execution.

### Providers

Current provider modules:

- `src/providers/gemini.rs`
- `src/providers/zai.rs`
- `src/providers/openai.rs`

Provider registry responsibility is intentionally narrow: it maps `AccountProvider` to implementation. It does **not** own model prefix routing policy.

### Runtime routing and account state

Shared routing/account logic lives in:

- `src/account_pool/pool.rs`
- `src/account_pool/routing.rs`

Important runtime behaviors:

- multiple accounts per provider
- health-aware account selection
- sticky routing for KV-cache reuse
- cooldown-based unhealthy recovery
- per-account health stats surfaced to the UI/config view

### Auth model

Gemini auth is account-scoped via `src/auth/mod.rs`.

- API-key Gemini accounts use account-local keys.
- OAuth Gemini accounts maintain isolated token/project caches per account.
- Z.AI and OpenAI currently use account-scoped API key auth.

## Configuration

Configuration now uses a structured v2 schema centered on:

- `server`
- `providers`
- `models`
- `routing`
- `accounts`
- `reasoning`
- `timeouts`
- `compaction`

Key behaviors:

- `accounts[]` is the source of truth for credentials.
- `routing.model_overrides` rewrites requested models before provider resolution.
- `routing.provider_prefixes` maps model prefixes to providers.
- `models.compaction_model` routes compaction through the same shared routing path.
- legacy flat config is only supported at startup migration time and is converted into the v2 internal config.

## UI and config endpoint

`GET /config` returns a typed snapshot of:

- structured config sections
- masked account auth data
- account health snapshot
- routing stats such as sticky binding count

Treat `POST /config` as read-only/stub unless the user explicitly asks to fully implement config persistence for the new schema.

## Development expectations

When changing routing/auth/provider code:

- keep protocol translation inside provider modules
- keep provider selection/account selection in shared routing
- do not reintroduce model-prefix routing inside provider registry
- do not read provider credentials directly from global flat config
- preserve the typed request/response flow in `src/schema/*`

## Verification

Preferred verification commands:

```bash
cargo fmt
cargo check
cargo test
```

When touching routing/auth/config logic, also sanity check:

- v2 config boot path
- legacy flat config migration path
- provider override routing
- multi-account isolation
- sticky-routing failover behavior

## Notes for future changes

- OpenAI forwarding is intended to stay straightforward: forward the OpenAI-shaped request upstream after swapping in the resolved upstream model and configured account auth.
- If you add a new provider, wire it through the shared routing/account path instead of adding provider-specific dispatch in handlers.
