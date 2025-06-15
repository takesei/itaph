.PHONY: test lint fix

all: lint format test

lint:
	uv run ruff check ./src ./tests --fix

format:
	uv run ruff format ./src ./tests

test:
	uv run pytest

init-pj:
	pre-commit install
	pre-commit run --all-files
