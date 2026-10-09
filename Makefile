.PHONY: install lint format typecheck test check data-tiny validate-tiny manifest-tiny data-kenai-dev data-kenai-holdout mlflow-import mlflow-ui

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

data-kenai-dev:
	uv run python scripts/stream_subset.py --config configs/data/kenai_subset.yaml
	uv run python scripts/build_manifest.py --version full-v2 --frames-subset configs/data/kenai_subset.yaml

data-kenai-holdout:
	uv run python scripts/stream_subset.py --config configs/data/kenai_holdout.yaml
	uv run python scripts/build_manifest.py --version full-v3 \
		--frames-subset configs/data/kenai_subset.yaml \
		--frames-subset configs/data/kenai_holdout.yaml

# Experiment tracking (docs/training.md#experiment-tracking-mlflow). MLflow runs in its own
# environment, never in the project's: it would pin protobuf for the service too.
mlflow-import:
	uv run scripts/log_to_mlflow.py

mlflow-ui:
	uvx --from 'mlflow==3.16.1' mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db \
		--host 127.0.0.1 --port 5050
