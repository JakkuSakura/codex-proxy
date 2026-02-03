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

## Automation Scripts (`/scripts`)
Always execute these scripts from the `codex-proxy/` root directory.

| Script | Purpose |
| :--- | :--- |
| `dev_start.sh` | Builds and starts the proxy container in the background. |
| `dev_stop.sh` | Shuts down the proxy container. |
| `debug_run.sh [prompt]` | Rebuilds the container, tails logs, and runs a test `codex exec` command. |
| `logs.sh` | Streams real-time logs from the `codex-proxy` container. |
| `test.sh` | Runs Python integration tests against the live proxy instance. |

## Development Patterns
1. **Credentials**: Ensure your host machine has valid Gemini credentials at `~/.gemini`.
2. **Iteration**: Use `debug_run.sh` to quickly test end-to-end changes.
3. **Verification**: Always run `test.sh` after modifications to ensure that chaining and proxy logic are still functional.
