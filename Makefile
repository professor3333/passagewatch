.PHONY: install lint format typecheck test check

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src/

test:
	uv run pytest -m "not slow"

check: lint typecheck test
