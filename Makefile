# Armdroid project Makefile
# Requires: Python >=3.11 in the active venv/environment.

.PHONY: help install lint typecheck format test coverage pre-commit clean

PYTHON  ?= python
SRC_DIR  = src
TEST_DIR = tests

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | \
	    awk 'BEGIN{FS=":.*## "}{printf "  %-16s %s\n", $$1, $$2}'

install:       ## Install all dev dependencies (editable)
	$(PYTHON) -m pip install -e ".[dev,hardware,anthropic]"

lint:          ## Run ruff linter and auto-fix safe issues
	$(PYTHON) -m ruff check $(SRC_DIR)/ $(TEST_DIR)/ --fix
	$(PYTHON) -m ruff format $(SRC_DIR)/ $(TEST_DIR)/

typecheck:     ## Run mypy strict type-check on source tree
	$(PYTHON) -m mypy --strict $(SRC_DIR)/

test:          ## Run the fast test suite (excludes slow/hardware/gpu)
	$(PYTHON) -m pytest -m "not slow and not hardware and not gpu" -q

test-all:      ## Run the full test suite (skips hardware by default)
	$(PYTHON) -m pytest -m "not hardware" -q

coverage:      ## Run tests with coverage report (fail under 85%)
	$(PYTHON) -m pytest --cov=armdroid --cov-fail-under=85 -q

pre-commit:    ## Run all pre-commit hooks on staged files
	pre-commit run --all-files

clean:         ## Remove __pycache__, .mypy_cache, .ruff_cache, .pytest_cache
	find . -type d -name __pycache__  -exec rm -rf {} + 2>/dev/null; true
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov dist build *.egg-info

check: lint typecheck coverage  ## Run lint + typecheck + coverage (CI gate)
