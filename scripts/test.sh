#!/bin/bash
# Runs the integration tests against the running container
set -euo pipefail

echo "Running verification setup..."
python3 tests/verify_setup.py

echo -e "\nRunning chaining test..."
python3 tests/test_chaining.py
