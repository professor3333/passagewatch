# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlflow==3.16.1"]
# ///
"""Record training runs and their counting evaluations in a local MLflow store.

Detectors are trained on Kaggle, which cannot reach a tracking server here, so each run
writes ``run.json`` (resolved config, seeds, git commit, device, library versions, data
manifest) and ``metrics.jsonl`` (losses and learning rate every ``log_every_iters``). This
script imports those files after a run is downloaded, along with the counting evaluations
of its checkpoints (``runs/neural/<run>/report-*.json``):

- experiment ``training``: one MLflow run per training run, with the flattened config and
  seeds as parameters, the logged losses as metrics (step = global iteration), the git
  commit, data manifest and each checkpoint's SHA-256 as tags, and the two files as
  artifacts;
- experiment ``evaluation``: one MLflow run per report, with the manifest, partition and
  tracker as parameters, and for every evaluated threshold ``nmae/t<threshold>`` (and
  missed, false, recall, precision) with step = epoch, so each threshold is a curve over
  epochs. The selected setting is tagged, and its values are logged as ``best/*``.

Importing is idempotent: a run whose source files are unchanged is skipped.

MLflow is not a project dependency. It needs protobuf below 7, which would change what the
service runs. ``uv run`` gives this script its own environment from the inline metadata
above. The store is ``mlruns/`` (not committed); browse it with ``make mlflow-ui``.

Example:
    uv run scripts/log_to_mlflow.py                    # every run under models/runs/
    uv run scripts/log_to_mlflow.py --run models/runs/yolox-tiny-t1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "mlruns"
SOURCE_TAG = "passagewatch.source_sha256"
REPORT_METRICS = {
    "macro_nmae": "nmae",
    "missed_passages": "missed",
    "false_passages": "false",
    "detection_recall": "recall",
    "detection_precision": "precision",
}
MLFLOW_BATCH = 1000  # metrics per log_batch call


def tracking_uri(store: Path) -> str:
    return f"sqlite:///{store / 'mlflow.db'}"


def source_digest(paths: list[Path]) -> str:
    """SHA-256 over the files a record comes from, so an unchanged run is not re-imported."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, str]:
    """Nested dicts as dotted keys; lists and scalars as strings (MLflow parameters)."""
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for key, item in value.items():
            out |= flatten(item, f"{prefix}{key}.")
        return out
    return {prefix.rstrip("."): json.dumps(value) if isinstance(value, list) else str(value)}


def training_params(run: dict[str, Any]) -> dict[str, str]:
    params = flatten(run["config"], "config.")
    params |= flatten(run.get("seeds", {}), "seed.")
    for key in ("samples", "clips", "iters_per_epoch", "amp", "preprocessing_version"):
        if key in run:
            params[key] = str(run[key])
    return params


def training_tags(run: dict[str, Any], checkpoints: dict[str, str]) -> dict[str, str]:
    meta = run.get("metadata", {})
    tags = {
        "passagewatch.kind": "training",
        "training_run": run["config"]["name"],
        "git_commit": str(run.get("git_commit")),
        "git_dirty": str(run.get("git_dirty")),
        "device": str(run.get("device")),
        "torch": str(run.get("torch")),
        "python": str(run.get("python")),
        "data_manifest": str(meta.get("manifest")),
        "data_manifest_sha256": str(meta.get("manifest_content_sha256")),
        "data_partition": str(meta.get("partition")),
    }
    tags |= {f"checkpoint.{name}.sha256": sha for name, sha in checkpoints.items()}
    return tags


def training_metrics(lines: list[str]) -> Iterator[tuple[str, float, int]]:
    """``(key, value, step)`` for every numeric value of every metrics line; the step is
    the global iteration."""
    for line in lines:
        row = json.loads(line)
        step = int(row["iteration"])
        for key, value in row.items():
            if key in ("iteration", "epoch"):
                continue
            if isinstance(value, int | float):  # bools too: head_only is 1.0 or 0.0
                yield key, float(value), step
        yield "epoch", float(row["epoch"]), step


def evaluation_metrics(report: dict[str, Any]) -> Iterator[tuple[str, float, int]]:
    """``(key, value, epoch)`` for every evaluated threshold, and ``best/*`` for the
    selected setting."""
    best = report.get("best") or {}
    for result in report["results"]:
        threshold = f"t{float(result['threshold']):.2f}"
        epoch = int(result["epoch"])
        for source, name in REPORT_METRICS.items():
            if result.get(source) is not None:
                yield f"{name}/{threshold}", float(result[source]), epoch
        if (result["epoch"], result["threshold"]) == (best.get("epoch"), best.get("threshold")):
            for source, name in REPORT_METRICS.items():
                if result.get(source) is not None:
                    yield f"best/{name}", float(result[source]), epoch


def evaluation_params(report: dict[str, Any]) -> dict[str, str]:
    params: dict[str, str] = {
        key: str(report[key])
        for key in ("manifest", "partition", "frames", "tracking_config", "selection_rule")
        if key in report
    }
    params |= flatten(report.get("tracker", {}), "tracker.")
    runtimes = {(r.get("runtime", "torch"), r.get("device", "")) for r in report["results"]}
    params["runtime"] = ", ".join(sorted(f"{rt} {dev}".strip() for rt, dev in runtimes))
    return params


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_run(run_dir: Path, reports_dir: Path, store: Path) -> None:
    import mlflow  # type: ignore[import-not-found]
    from mlflow.entities import Metric, Param  # type: ignore[import-not-found]

    mlflow.set_tracking_uri(tracking_uri(store))
    client = mlflow.MlflowClient()

    def experiment(name: str) -> str:
        found = client.get_experiment_by_name(name)
        if found is not None:
            return str(found.experiment_id)
        location = (store / "artifacts" / name).resolve().as_uri()
        return str(client.create_experiment(name, artifact_location=location))

    def existing(experiment_id: str, digest: str) -> str | None:
        runs = client.search_runs(
            [experiment_id], filter_string=f"tags.`{SOURCE_TAG}` = '{digest}'", max_results=1
        )
        return str(runs[0].info.run_id) if runs else None

    def log(run_id: str, metrics: list[tuple[str, float, int]], params: dict[str, str]) -> None:
        stamp = 0  # timestamps carry no meaning for imported records; steps order them
        client.log_batch(run_id, params=[Param(k, v) for k, v in params.items()])
        for i in range(0, len(metrics), MLFLOW_BATCH):
            chunk = metrics[i : i + MLFLOW_BATCH]
            client.log_batch(run_id, metrics=[Metric(k, v, stamp, s) for k, v, s in chunk])

    run_json, metrics_jsonl = run_dir / "run.json", run_dir / "metrics.jsonl"
    run = json.loads(run_json.read_text(encoding="utf-8"))
    name = run["config"]["name"]
    training_id = experiment("training")
    digest = source_digest([run_json, metrics_jsonl])
    run_id = existing(training_id, digest)
    if run_id is None:
        checkpoints = {p.stem: sha256_file(p) for p in sorted(run_dir.glob("epoch-*.pt"))}
        tags = training_tags(run, checkpoints) | {SOURCE_TAG: digest}
        created = client.create_run(training_id, run_name=name, tags=tags)
        run_id = str(created.info.run_id)
        lines = metrics_jsonl.read_text(encoding="utf-8").splitlines()
        log(run_id, list(training_metrics(lines)), training_params(run))
        for path in (run_json, metrics_jsonl, REPO_ROOT / f"configs/training/{name}.yaml"):
            if path.is_file():
                client.log_artifact(run_id, str(path))
        client.set_terminated(run_id)
        print(f"training/{name}: imported ({len(checkpoints)} checkpoints hashed)")
    else:
        print(f"training/{name}: unchanged, skipped")

    evaluation_id = experiment("evaluation")
    for report_path in sorted((reports_dir / name).glob("report-*.json")):
        report_digest = source_digest([report_path])
        label = f"{name}/{report_path.stem.removeprefix('report-')}"
        if existing(evaluation_id, report_digest) is not None:
            print(f"evaluation/{label}: unchanged, skipped")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        best = report.get("best") or {}
        tags = {
            "passagewatch.kind": "evaluation",
            "training_run": name,
            "training_mlflow_run_id": run_id,
            "best_epoch": str(best.get("epoch")),
            "best_threshold": str(best.get("threshold")),
            SOURCE_TAG: report_digest,
        }
        created = client.create_run(evaluation_id, run_name=label, tags=tags)
        eval_id = str(created.info.run_id)
        log(eval_id, list(evaluation_metrics(report)), evaluation_params(report))
        client.log_artifact(eval_id, str(report_path))
        client.set_terminated(eval_id)
        print(f"evaluation/{label}: imported ({len(report['results'])} settings)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, action="append", help="a training run directory")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "models/runs")
    parser.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "runs/neural")
    parser.add_argument("--store", type=Path, default=STORE)
    args = parser.parse_args()

    runs = args.run or sorted(p for p in args.runs_dir.iterdir() if (p / "run.json").is_file())
    if not runs:
        raise SystemExit(f"no training runs (run.json) under {args.runs_dir}")
    args.store.mkdir(parents=True, exist_ok=True)
    for run_dir in runs:
        import_run(run_dir, args.reports_dir, args.store)
    print(f"MLflow store: {tracking_uri(args.store)} (make mlflow-ui)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
