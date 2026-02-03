#!/bin/bash
# Follows the logs of the proxy container
set -euo pipefail

docker-compose logs -f codex-proxy
