from __future__ import annotations

import pytest
from pydantic import ValidationError

from passagewatch.service.bundle import DetectorSpec

BASE = {
    "model_size": "tiny",
    "checkpoint": "detector.pt",
    "checkpoint_sha256": "a" * 64,
    "input_height": 960,
    "input_width": 416,
    "score_threshold": 0.2,
    "training_run": "r",
    "epoch": 1,
}


def test_an_onnx_runtime_needs_its_file_and_hash_and_only_it() -> None:
    DetectorSpec.model_validate(BASE)
    DetectorSpec.model_validate(
        BASE | {"runtime": "onnxruntime", "onnx_file": "d.onnx", "onnx_sha256": "b" * 64}
    )
    with pytest.raises(ValidationError, match="onnx_file"):
        DetectorSpec.model_validate(BASE | {"runtime": "onnxruntime"})
    with pytest.raises(ValidationError, match="onnx_file"):
        DetectorSpec.model_validate(BASE | {"onnx_file": "d.onnx", "onnx_sha256": "b" * 64})
