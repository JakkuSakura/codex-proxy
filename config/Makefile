.PHONY: help test lint format build run stop logs clean install

help:  Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/ /'

## Development
install:  Install dependencies
	pip install -r requirements.txt
	pip install pytest ruff mypy pytest-responses

test:  Run tests
	pytest tests/ -v

lint:  Run linters
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports || true

format:  Format code with ruff
	ruff format src/ tests/

check:  Run all checks (lint + test)
	$(MAKE) lint
	$(MAKE) test

## Docker
build:  Build Docker image
	docker-compose build

run:  Start proxy container
	docker-compose up -d

stop:  Stop proxy container
	docker-compose down

logs:  Show proxy logs
	docker-compose logs -f

restart:  Restart proxy container
	$(MAKE) stop
	$(MAKE) run

## Cleanup
clean:  Clean up build artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
