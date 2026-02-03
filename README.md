# codex-proxy

Codex-proxy is an intermediary service for the Codex CLI. It bridges Codex's specialized "Responses API" with standard AI providers like Z.AI and Gemini, fixing role incompatibilities and adding features like automated context compaction.

## What it does

- **Bridge the Responses API**: Codex uses a proprietary Responses API for complex tasks. This proxy translates those requests into standard Chat Completions that most providers understand.
- **Fix role errors**: Some providers (like Z.AI) crash when they see the `developer` role. The proxy automatically maps `developer` to `system` so your requests actually go through.
- **Power up Gemini**: It has deep support for Gemini 2.0 and 3.0, including "thinking" blocks, JSON schemas, and automatic model fallbacks if you hit rate limits.
- **Auto-compact history**: When a conversation gets too long, the proxy uses a fast model (like Gemini Flash Lite) to summarize the history, keeping your context window clean without manual effort.
- **Low latency**: It's built with multi-threading and optimized JSON handling (`orjson`) to keep overhead minimal.

## Supported providers

### Gemini
The proxy handles the heavy lifting for Gemini—authentication, project IDs, and specific model settings. It supports the latest thinking/reasoning models and handles streaming responses natively.

### Z.AI (GLM)
If you're using Z.AI's GLM models, this proxy fixes the common "Incorrect role information" error (code 1214) by cleaning up the message roles before they hit the API.

## Setup

### Run with Docker (Recommended)
The easiest way to get started is using the included scripts:

```bash
# Start the proxy in the background
./scripts/dev_start.sh
```

The proxy will listen on `http://localhost:8765`.

### Run locally
If you don't want to use Docker:

1. Install the requirements: `pip install -r requirements.txt`
2. Start the service: `python -m src.codex_proxy`

## Configuration

You can tweak the proxy using environment variables or by editing `~/.gemini/proxy_config.json`.

- `PORT`: Where the proxy listens (default: `8765`)
- `Z_AI_API_KEY`: Your Z.AI key (passed via Codex)
- `GEMINI_CLIENT_ID` / `SECRET`: Credentials for Gemini internal APIs
- `DEBUG`: Set to `true` for detailed request logs

## Connecting Codex

Once the proxy is running, tell Codex to use it by updating your `~/.codex/config.toml`.

### Z.AI Example
```toml
[model_providers.z_ai]
name = "z.ai - GLM"
base_url = "http://localhost:8765"
env_key = "Z_AI_API_KEY"
wire_api = "chat"

[profiles.glm_4]
model = "glm-4.0"
model_provider = "z_ai"
```

### Gemini Example
```toml
[model_providers.gemini_proxy]
name = "Gemini Proxy"
base_url = "http://localhost:8765"
wire_api = "responses"

[profiles.gemini_3]
model = "gemini-3-pro-preview"
model_provider = "gemini_proxy"
```

## Testing & Debugging

- Use `./scripts/debug_run.sh -- "your prompt"` to rebuild the proxy and run a quick test.
- Use `./scripts/logs.sh` to see real-time proxy traffic.

