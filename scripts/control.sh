#!/bin/bash
# Unified control script for the codex-proxy.

set -euo pipefail

# --- Utility Functions ---
usage() {
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  start         - Start the proxy container in detached mode."
    echo "  stop          - Stop and remove the proxy container."
    echo "  logs          - Follow the logs of the proxy container."
    echo "  test          - Run the pytest test suite."
    echo "  run           - Run a codex command through the proxy."
    echo
    echo "Options for 'run' command:"
    echo "  -p, --profile <name>  - Specify a codex profile to use."
    echo
    echo "Example:"
    echo "  $0 run -p glm -- 'hello'"
    exit 1
}

# --- Main Command Logic ---
COMMAND=${1:-}
if [[ -z "$COMMAND" ]]; then
    usage
fi
shift

case "$COMMAND" in
    start)
        echo "Starting proxy container..."
        docker-compose up -d --build
        ;;
    stop)
        echo "Stopping proxy container..."
        docker-compose down
        ;;
    logs)
        echo "Following logs..."
        docker-compose logs -f
        ;;
    test)
        echo "Running tests..."
        if ! command -v pytest &> /dev/null; then
            echo "Error: pytest not found. Install with: pip install pytest pytest-responses"
            exit 1
        fi
        pytest tests/ -v
        ;;

    run)
        # --- `run` command logic (from debug_run.sh) ---
        PROFILE=""
        while [[ "$#" -gt 0 ]]; do
            case $1 in
                -p|--profile)
                    if [[ -n "${2:-}" ]]; then
                        PROFILE=$2
                        shift 2
                    else
                        echo "Error: --profile requires a value" >&2
                        exit 1
                    fi
                    ;;
                --)
                    shift
                    break
                    ;;
                *)
                    break
                    ;;
            esac
        done

        echo "Rebuilding and restarting proxy..."
        docker-compose up -d --build
        echo "Waiting for proxy to be ready..."
        sleep 2

        echo "--------------------------------------------------------"
        echo "Running Codex test..."
        if [[ -n "$PROFILE" ]]; then
            echo "Using profile: $PROFILE"
        fi
        echo "--------------------------------------------------------"

        if ! command -v codex &> /dev/null; then
            echo "Error: 'codex' command not found in PATH."
            exit 1
        fi

        CMD="codex exec"
        if [[ -n "$PROFILE" ]]; then
            CMD="$CMD -p $PROFILE"
        fi

        $CMD -- "$@"

        echo "--------------------------------------------------------"
        echo "Test complete. Check 'docker logs codex-proxy' for details."
        ;;
    *)
        usage
        ;;
esac
