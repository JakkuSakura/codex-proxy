# Gemini Proxy for Codex

I built this proxy because I wanted Gemini models to work properly in Codex without the usual protocol mismatch issues. Native models in Codex use a specific Responses API that Gemini doesn't quite match out of the box—especially when it comes to reasoning streams, tool calls, and context management.

This proxy sits between the Codex binary and Google's internal APIs. It maps Gemini's output to the strict JSON SSE format Codex expects, so you get stable reasoning "Thinking" blocks, reliable terminal commands, and proper history compaction.

## Why this exists

If you've tried using Gemini directly in Codex, you probably noticed the "..." status indicator flickering or the stream disconnecting during complex tasks. I fixed that by:
- Splitting Gemini's reasoning stream into professional summaries that the terminal UI can actually track.
- Implementing a custom `/v1/responses/compact` endpoint using Gemini Flash Lite so history stays manageable.
- Mapping image perception to the native `<image>` tag standard.
- Ensuring turn-state persists across multiple requests so the agent doesn't lose its mind.

## Setup

### 1. Credentials
The proxy uses your existing Gemini CLI credentials. Ensure you have authenticated via `gemini-cli auth login` first.

You'll need to create a simple config file at `~/.gemini/proxy_config.json` with your Google OAuth client info:

```json
{
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET"
}
```

*Note: You can find these in the gemini-cli source code or your Google Cloud Console.*

### 2. Run with Docker (Recommended)
I use Docker because it keeps the environment clean. From the `codex-proxy` directory:

```bash
./scripts/dev_start.sh
```

This starts the proxy on port `8765`. It mounts your `~/.gemini` folder so it can access your tokens.

### 3. Point Codex to the Proxy
Update your `~/.codex/config.toml` (or your active profile) to use the proxy:

```toml
[model_provider]
name = "gemini-proxy"
base_url = "http://localhost:8765/v1"
wire_api = "responses"

[model]
default = "gemini-3-flash-preview"
```

## Features I added

- **High-Fidelity Reasoning:** Concurrent streaming of reasoning deltas and summaries. The terminal status indicator actually works.
- **Context Compaction:** When history gets too long, Codex calls the proxy's compact endpoint. I use Gemini 1.5 Flash Lite here to keep it fast and cheap.
- **Strict JSON Schema:** If Codex requests a specific JSON format, the proxy enforces it via Gemini's native responseSchema.
- **Automatic 429 Retries:** If you hit rate limits, the proxy handles the wait and retry logic internally.

## Safety and Soul

I stripped out all hardcoded secrets. You have to provide your own client ID and secret via the config file or environment variables (`GEMINI_CLIENT_ID`, `GEMINI_CLIENT_SECRET`). 

The code is meant to be pragmatic. I've tuned the system instructions to make Gemini act more like a senior engineer—direct, pragmatic, and focused on the root cause rather than surface-level fixes.
