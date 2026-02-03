#!/bin/bash
# Fast iteration script for testing Codex with the proxy.
# Usage: ./scripts/debug_run.sh "your prompt here"

set -e

# 1. Rebuild and restart proxy
echo "Rebuilding and restarting proxy..."
docker-compose up -d --build

# 2. Wait for proxy
echo "Waiting for proxy to be ready..."
sleep 2

# 3. Run Codex test
PROMPT="${1:-hi}"
echo "--------------------------------------------------------"
echo "Running Codex test with prompt: $PROMPT"
echo "--------------------------------------------------------"

# Ensure we use the right model if needed, but proxy handles fallback
/home/user/.npm-global/bin/codex exec "$PROMPT"

echo "--------------------------------------------------------"
echo "Test complete. Check 'docker logs codex-proxy' for details."