#!/bin/bash
# Starts the development environment
# Rebuilds if necessary to catch dependency changes
docker-compose up -d --build --force-recreate
echo "Waiting for proxy to be ready..."
sleep 2
docker-compose logs --tail=10 codex-proxy
