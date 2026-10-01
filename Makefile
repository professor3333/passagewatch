.PHONY: install lint format typecheck test check data-tiny validate-tiny manifest-tiny

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src/ scripts/

test:
	uv run pytest -m "not slow"

check: lint typecheck test

data-tiny:
	uv run python scripts/download_data.py --bundle tiny

validate-tiny:
	uv run python scripts/validate_data.py --subset tiny

manifest-tiny:
	uv run python scripts/build_manifest.py --version tiny-v1
