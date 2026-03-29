.PHONY: help install test lint format check run clean e2e-codex build

help:
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@echo '  install     Install dependencies via uv'
	@echo '  build       Build the Rust proxy binary'
	@echo '  run         Start the proxy server'
	@echo '  test        Run the unit/integration test suite'
	@echo '  e2e-codex   Run e2e tests with codex CLI through the proxy'
	@echo '  lint        Run ruff and mypy'
	@echo '  format      Format code with ruff'
	@echo '  check       Run lint + test'
	@echo '  clean       Remove build/cache artifacts'

install:
	uv sync

build:
	cargo build

run:
	uv run python -m codex_proxy

test:
	uv run pytest tests/ -v

e2e-codex: build
	uv run pytest tests/test_e2e_codex.py -v --timeout=300

lint:
	uv run ruff check src/ tests/
	uv run mypy src/ || true

format:
	uv run ruff format src/ tests/

check: lint test

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
