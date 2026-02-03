#!/bin/bash
# Fast iteration script for testing Codex with the proxy.
# Usage: ./scripts/debug_run.sh -- "your prompt here"

set -euo pipefail

# 1. Rebuild and restart proxy
echo "Rebuilding and restarting proxy..."
docker-compose up -d --build

# 2. Wait for proxy
echo "Waiting for proxy to be ready..."
sleep 2

# 3. Run Codex test
echo "--------------------------------------------------------"
echo "Running Codex test..."
echo "--------------------------------------------------------"

# Use codex from PATH
if ! command -v codex &> /dev/null; then
    echo "Error: 'codex' command not found in PATH."
    exit 1
fi

# Handle optional -- separator
if [[ "${1:-}" == "--" ]]; then
    shift
fi

codex exec "$@"

echo "--------------------------------------------------------"
echo "Test complete. Check 'docker logs codex-proxy' for details."