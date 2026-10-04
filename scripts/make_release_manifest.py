"""Write the immutable release manifest of a bundle: ``releases/manifests/<release>.json``.

Records the code commit (refused with uncommitted changes), the ``uv.lock`` hash, the
dataset manifest and training run, the bundle's identity, and the evaluation the release was
selected and confirmed on: nMAE with a paired clip-bootstrap 95% interval for each report,
the hashes of the report files, and, for additional ``--parity-report`` files (the same
model on other runtimes), how many clips' counts differ.

Example:
    uv run python scripts/make_release_manifest.py --bundle bundles/passagewatch-0.3.0 \\
        --training-run models/runs/yolox-tiny-t1 \\
        --report runs/neural/yolox-tiny-t1/report-val-onnx.json \\
        --report runs/neural/yolox-tiny-t1/report-holdout.json \\
        --parity-report runs/neural/yolox-tiny-t1/report-val.json \\
        --parity-report runs/neural/yolox-tiny-t1/report-val-cpu.json \\
        --selection docs/error_analysis.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from passagewatch.evaluation.bootstrap import paired_bootstrap
from passagewatch.ingestion.inventory import sha256_of
from passagewatch.service.bundle import load_bundle
from passagewatch.service.jobs import canonical_json
from passagewatch.service.release import ReleaseManifest, bundle_identity, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def result_for(report: dict[str, Any], epoch: int, threshold: float) -> dict[str, Any]:
    matches: list[dict[str, Any]] = [
        r
        for r in report["results"]
        if r["epoch"] == epoch and abs(r["threshold"] - threshold) < 1e-9
    ]
    if len(matches) != 1:
        raise SystemExit(f"no single result for epoch {epoch}, threshold {threshold}")
    return matches[0]


def device_of(path: Path, result: dict[str, Any]) -> str:
    """Older reports do not record the device: ``-onnx``/``-cpu`` reports ran on the CPU,
    and unsuffixed ones on Apple's GPU (MPS)."""
    if result.get("device"):
        return str(result["device"])
    return "cpu" if path.stem.endswith(("-onnx", "-cpu")) else "mps"


def summary(path: Path, epoch: int, threshold: float, checkpoint: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result = result_for(report, epoch, threshold)
    if result["checkpoint_sha256"] != checkpoint:
        raise SystemExit(f"{path} is for checkpoint {result['checkpoint_sha256'][:12]}")
    reference = [c["reference"] for c in result["clips"]]
    predicted = [c["predicted"] for c in result["clips"]]
    interval = paired_bootstrap(reference, predicted, predicted).a
    return {
        "report": str(path.relative_to(REPO_ROOT)),
        "report_sha256": sha256_of(path),
        "manifest": report["manifest"],
        "partition": report["partition"],
        "runtime": result.get("runtime", "torch"),
        "device": device_of(path, result),
        "clips": len(reference),
        "passages": int(sum(sum(r) for r in reference)),
        "nmae": round(interval.estimate, 4),
        "nmae_95ci": [round(interval.low, 4), round(interval.high, 4)],
        "missed_passages": result["missed_passages"],
        "false_passages": result["false_passages"],
        "detection_recall": round(result["detection_recall"], 4),
        "detection_precision": round(result["detection_precision"], 4),
    }


def parity(base: Path, other: Path, epoch: int, threshold: float) -> dict[str, Any]:
    a = result_for(json.loads(base.read_text(encoding="utf-8")), epoch, threshold)
    b = result_for(json.loads(other.read_text(encoding="utf-8")), epoch, threshold)
    pa = {c["clip_name"]: c["predicted"] for c in a["clips"]}
    pb = {c["clip_name"]: c["predicted"] for c in b["clips"]}
    if set(pa) != set(pb):
        raise SystemExit(f"{base} and {other} cover different clips")
    return {
        "report": str(other.relative_to(REPO_ROOT)),
        "report_sha256": sha256_of(other),
        "runtime": b.get("runtime", "torch"),
        "device": device_of(other, b),
        "clips": len(pa),
        "clips_with_different_counts": sum(pa[k] != pb[k] for k in pa),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--parity-report", type=Path, action="append", default=[])
    parser.add_argument("--selection", required=True, help="where the selection is documented")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true", help="for testing only")
    args = parser.parse_args(argv)

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        parser.error("the working tree has uncommitted changes; commit them first")
    bundle = load_bundle(args.bundle.resolve())
    detector = bundle.detector
    run = json.loads((args.training_run / "run.json").read_text(encoding="utf-8"))
    training_meta = bundle.provenance.get("training_metadata", {})
    reports = [
        summary(p.resolve(), detector.epoch, detector.score_threshold, detector.checkpoint_sha256)
        for p in args.report
    ]
    checks = [
        parity(args.report[0].resolve(), p.resolve(), detector.epoch, detector.score_threshold)
        for p in args.parity_report
    ]
    manifest = ReleaseManifest(
        release=bundle.pipeline_version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        code_commit=git("rev-parse", "HEAD"),
        uv_lock_sha256=sha256_of(REPO_ROOT / "uv.lock"),
        dataset_manifest={
            "version": training_meta.get("manifest"),
            "content_sha256": training_meta.get("manifest_content_sha256"),
            "partition": training_meta.get("partition"),
            "clips": training_meta.get("clips"),
        },
        training={
            "run": run["config"]["name"],
            "config_file": f"configs/training/{run['config']['name']}.yaml",
            "config_sha256": sha256_of(
                REPO_ROOT / "configs/training" / f"{run['config']['name']}.yaml"
            ),
            "resolved_config_sha256": hashlib.sha256(
                canonical_json(run["config"]).encode()
            ).hexdigest(),
            "training_commit": run.get("git_commit"),
            "device": run.get("device"),
            "torch": run.get("torch"),
        },
        bundle=bundle_identity(bundle),
        evaluation={
            "selection": args.selection,
            "reports": reports,
            "parity": checks,
            "test_locations": "not evaluated (Stage 11, once, for a declared release)",
        },
    )
    out = args.out or REPO_ROOT / "releases/manifests" / f"{bundle.pipeline_version}.json"
    write_manifest(manifest, out)
    print(f"{bundle.pipeline_version} -> {out}")
    for r in reports:
        print(f"  {r['partition']:8} {r['runtime']:12} nMAE {r['nmae']} {r['nmae_95ci']}")
    for c in checks:
        print(
            f"  parity {c['report']}: "
            f"{c['clips_with_different_counts']} of {c['clips']} clips differ"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
