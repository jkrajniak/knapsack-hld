.DEFAULT_GOAL := help
SHELL := bash

# Public CLI for the knapsack-hld reproducibility package.
# These targets are thin convenience wrappers; all heavy lifting lives
# in scripts/. Override paths/parallelism via make variables, e.g.
#   make reproduce JOBS=16 FULL_OUT=instances_full_candidate

ARCHIVE_OUT ?= instances
FULL_OUT    ?= instances_full_candidate
JOBS        ?= 4

.PHONY: help sync test lint format reproduce-quick reproduce verify

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync: ## Install and lock the environment with uv
	uv sync

test: ## Run the test suite
	uv run pytest

lint: ## Lint and format check (ruff)
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format the codebase (ruff)
	uv run ruff format .

reproduce-quick: ## Smoke archive build + verify (< 10 min; same flow as CI)
	uv run python scripts/generate_instances.py \
		--config scripts/configs/archive_smoke.yaml --out $(ARCHIVE_OUT) --jobs 2
	uv run python scripts/verify_instances.py --archive $(ARCHIVE_OUT)

reproduce: ## Full benchmark archive (long; runs in parallel)
	scripts/run_full_archive.sh --out $(FULL_OUT) --jobs $(JOBS)

verify: ## Verify an existing instance archive's manifest
	uv run python scripts/verify_instances.py --archive $(ARCHIVE_OUT)
