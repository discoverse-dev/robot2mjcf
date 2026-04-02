SHELL := /bin/sh

.DEFAULT_GOAL := help

UV := uv
UV_CACHE_DIR ?= .uv-cache
TARGETS := src/robot2mjcf tests
PKG_SMOKE_VENV := .pkg-smoke
PKG_SMOKE_PYTHON := $(PKG_SMOKE_VENV)/bin/python

export UV_CACHE_DIR

.PHONY: help sync format format-check lint lint-fix typecheck test check cli-smoke import-smoke build package-smoke ci clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install all dependencies
	$(UV) sync --all-groups

format: ## Format source and tests
	$(UV) run ruff format $(TARGETS)

format-check: ## Check formatting without modifying files
	$(UV) run ruff format --check $(TARGETS)

lint: ## Lint source and tests
	$(UV) run ruff check $(TARGETS)

lint-fix: ## Lint source and tests and apply fixes
	$(UV) run ruff check --fix $(TARGETS)

typecheck: ## Run mypy
	$(UV) run mypy $(TARGETS)

test: ## Run pytest
	$(UV) run pytest

check: format-check lint typecheck test ## Run core CI checks

cli-smoke: ## Verify the CLI entry points start
	$(UV) run robot2mjcf --help
	$(UV) run robot2mjcf-modelpath --help
	$(UV) run robot2mjcf-mjcf2obj --help

import-smoke: ## Run the import smoke test
	$(UV) run pytest tests/test_import.py -q --no-cov

build: ## Build the package
	$(UV) build

package-smoke: build ## Build and smoke-test the wheel in a fresh venv
	rm -rf $(PKG_SMOKE_VENV)
	$(UV) venv $(PKG_SMOKE_VENV)
	$(UV) pip install --python $(PKG_SMOKE_PYTHON) dist/*.whl
	$(PKG_SMOKE_VENV)/bin/robot2mjcf --help
	$(PKG_SMOKE_VENV)/bin/robot2mjcf-modelpath --help
	$(PKG_SMOKE_VENV)/bin/robot2mjcf-mjcf2obj --help

ci: check cli-smoke import-smoke package-smoke ## Run the local CI suite

clean: ## Remove build artifacts and caches
	rm -rf build dist .coverage htmlcov .mypy_cache .pytest_cache .ruff_cache .uv-cache $(PKG_SMOKE_VENV)
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
