"""The MLflow importer's records (scripts/log_to_mlflow.py), without MLflow itself."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/log_to_mlflow.py"


def importer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("log_to_mlflow", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mlflow_is_imported_only_when_logging() -> None:
    # The project environment has no MLflow (it would pin protobuf for the service).
    assert "import mlflow" not in SCRIPT.read_text().split("def import_run")[0]
    importer()


def test_training_params_flatten_the_config_and_seeds() -> None:
    run = {
        "config": {"name": "r", "epochs": 30, "augment": {"contrast_range": [0.8, 1.2]}},
        "seeds": {"python": 0, "torch": 0},
        "samples": 10,
    }
    params = importer().training_params(run)

    assert params == {
        "config.name": "r",
        "config.epochs": "30",
        "config.augment.contrast_range": "[0.8, 1.2]",
        "seed.python": "0",
        "seed.torch": "0",
        "samples": "10",
    }


def test_training_metrics_use_the_global_iteration_as_step() -> None:
    lines = [
        json.dumps({"epoch": 1, "iteration": 50, "lr": 0.1, "head_only": True, "total_loss": 9.0}),
        json.dumps(
            {"epoch": 2, "iteration": 1200, "lr": 0.2, "head_only": False, "total_loss": 4.0}
        ),
    ]
    metrics = list(importer().training_metrics(lines))

    assert ("total_loss", 4.0, 1200) in metrics
    assert ("head_only", 1.0, 50) in metrics and ("head_only", 0.0, 1200) in metrics
    assert ("epoch", 2.0, 1200) in metrics
    assert not any(key == "iteration" for key, _, _ in metrics)


def test_evaluation_metrics_are_curves_per_threshold_and_the_selected_setting() -> None:
    report = {
        "best": {"epoch": 25, "threshold": 0.4},
        "results": [
            {"epoch": 20, "threshold": 0.4, "macro_nmae": 0.08, "missed_passages": 9},
            {"epoch": 25, "threshold": 0.4, "macro_nmae": 0.06, "missed_passages": 7},
            {"epoch": 25, "threshold": 0.3, "macro_nmae": 0.07, "missed_passages": 6},
        ],
    }
    metrics = list(importer().evaluation_metrics(report))

    assert ("nmae/t0.40", 0.08, 20) in metrics and ("nmae/t0.40", 0.06, 25) in metrics
    assert ("nmae/t0.30", 0.07, 25) in metrics
    assert [m for m in metrics if m[0].startswith("best/")] == [
        ("best/nmae", 0.06, 25),
        ("best/missed", 7.0, 25),
    ]


def test_an_unchanged_source_has_the_same_digest(tmp_path: Path) -> None:
    module = importer()
    a, b = tmp_path / "run.json", tmp_path / "metrics.jsonl"
    a.write_text("{}")
    b.write_text("")
    first = module.source_digest([a, b])
    assert module.source_digest([a, b]) == first
    b.write_text("{}\n")
    assert module.source_digest([a, b]) != first
