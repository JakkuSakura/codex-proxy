# Codex Proxy Context

## Overview

The `codex-proxy` is a Python-based intermediary service for the Codex system, designed to handle API orchestration and environment-specific logic. It is container-native and relies on Docker for consistent execution.

## Infrastructure: Docker

- **Container Name**: `codex-proxy`
- **Port**: `8765`
- **Mounts**:
  - `${HOME}/.gemini` is mounted to `/home/appuser/.gemini` for credential access.
  - `./src` is mounted to `/app/src` to enable hot-reloading during development.
- **Commands**: Use `docker-compose` for direct lifecycle management if scripts are not used.

## Control Script (`/scripts`)

The project uses a unified control script for all operations. Execute from the `codex-proxy/` root directory.

```bash
./scripts/control.sh <command> [options]
```

**Commands:**

- `start` - Start the proxy container in detached mode
- `stop` - Stop and remove the proxy container
- `logs` - Follow the logs of the proxy container
- `test` - Run the pytest test suite
- `run [-p|--profile <name>] -- "prompt"` - Rebuild container and run codex command through proxy

## Development Patterns

1. **Credentials**: Ensure your host machine has valid Gemini credentials at `~/.gemini`.
2. **Iteration**: Use `control.sh run` to quickly test end-to-end changes.
3. **Verification**: Always run `control.sh test` after modifications to ensure that chaining and proxy logic are still functional.

**Examples:**

```bash
# Start the proxy
./scripts/control.sh start

# Run a test command
./scripts/control.sh run -- "hello world"

# Run with specific profile
./scripts/control.sh run -p glm -- "test prompt"

# Check logs
./scripts/control.sh logs

# Run tests
./scripts/control.sh test

# Stop the proxy
./scripts/control.sh stop
```

## Engineering Mandate: Deep API Parity

The primary goal of `codex-proxy` is to ensure seamless compatibility between multiple AI providers (Gemini, Z.AI) and the OpenAI Responses API protocol used by the Codex ecosystem. The proxy normalizes different wire formats to a unified internal OpenAI-like structure.

### Core Architecture

**Multi-Provider Support:**

- **Gemini Provider** (`gemini*` models): Uses Google's internal and public APIs with OAuth2 authentication
- **Z.AI Provider** (`glm*`, `zai*` models): Uses Z.AI's coding-focused API with Bearer token authentication
- **Provider Registry**: Automatically routes requests based on model prefixes

**Request Normalization** (`normalizer.py`):

- Converts OpenAI Responses API format (`/responses` endpoint) to internal OpenAI chat format
- Handles instruction-to-system message mapping
- Processes complex input structures including reasoning blocks, tool calls, and multi-part content
- Normalizes tool definitions between flat and wrapped formats

### Parity Requirements

**OpenAI Responses API Compatibility:**

- **Request Format**: Support `/responses` endpoint with `model`, `input`, `instructions`, `previous_response_id`, `store` fields
- **Response Streaming**: Emit proper SSE events including `response.created`, `response.done`, and intermediate content chunks
- **Tool Call Handling**: Preserve function call IDs, arguments, and parallel execution capabilities
- **Reasoning Content**: Extract and properly format `reasoning_content` from Gemini thinking blocks
- **Conversation Chaining**: Handle `previous_response_id` for multi-turn conversations
- **Metadata Preservation**: Pass through headers like `x-codex-turn-state` and response metadata

**Gemini Integration:**

- **Authentication**: OAuth2 flow with proper token refresh and credential management
- **Reasoning Extraction**: Parse Gemini's internal reasoning format and convert to OpenAI-compatible `reasoning_content`
- **Tool Call Mapping**: Convert Gemini's function calling format to OpenAI's `tool_calls` structure
- **Token Usage**: Map Gemini's token counts including `thinkingTokenCount` to OpenAI usage format

**Z.AI Integration:**

- **API Endpoint**: Use dedicated coding API at `https://api.z.ai/api/coding/paas/v4/chat/completions`
- **Streaming**: Handle Z.AI's SSE streaming format with proper chunk processing
- **Tool Call Support**: Process Z.AI's tool call deltas and assemble complete function calls
- **Model Support**: Support GLM models (glm-4.6, glm-4.7, etc.) with proper model prefix routing

### Implementation Specifications

**Streaming Protocol:**

- Emit `response.created` event with full response object at start
- Stream content deltas with proper sequence numbers
- Handle tool call assembly from streaming deltas
- Emit final usage statistics and completion events

**Error Handling:**

- Map provider-specific error codes to OpenAI-compatible format
- Preserve rate limiting information and quota details
- Handle authentication failures and token refresh scenarios

**Performance Optimizations:**

- Multi-threaded HTTP server with connection reuse
- Binary JSON (orjson) for fast serialization when available
- Efficient session management with connection pooling
- Minimal request/response transformation overhead

### Source References

- **OpenAI Responses API**: `https://context7.com/websites/platform_openai/llms.txt?topic=Responses`
- **Z.AI API Documentation**: `https://context7.com/websites/z_ai/llms.txt?topic=api`
- **Reference Implementations**: Source code in `~/Work/codex-proxy/reference/` for protocol analysis

The proxy must maintain 1:1 behavioral parity with native OpenAI Responses API while seamlessly bridging the underlying provider differences.
