from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from passagewatch.detection.neural import build_yolox
from passagewatch.detection.onnx import export_onnx, onnx_detector
from passagewatch.inference.neural import detect_frames
from passagewatch.preprocessing.letterbox import PREPROCESSING_VERSION


def checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    torch.save(
        {
            "config": {"name": "t", "model_size": "tiny", "input_height": 96, "input_width": 64},
            "preprocessing_version": PREPROCESSING_VERSION,
            "model": build_yolox("tiny").state_dict(),
            "state": {"epoch": 1},
        },
        path,
    )
    return path


def test_onnx_runtime_reproduces_the_torch_network(tmp_path: Path) -> None:
    detector = export_onnx(checkpoint(tmp_path / "c.pt"), tmp_path / "c.onnx")
    onnx = onnx_detector(detector, tmp_path / "c.onnx", threads=1)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (120, 80), dtype=np.uint8) for _ in range(3)]
    batch = torch.from_numpy(rng.uniform(0, 255, (3, 3, 96, 64)).astype(np.float32))

    with torch.no_grad():
        expected = detector.model(batch).float()
    actual = onnx.model(batch)

    assert actual.shape == expected.shape
    assert float((actual - expected).abs().max()) < 1e-2  # box pixels; scores far closer
    assert float((actual[..., 4:] - expected[..., 4:]).abs().max()) < 1e-4
    # The same letterboxing, decoding and NMS: the same detections. Random weights score
    # about 1e-4 (YOLOX's 0.01 priors for objectness and class), so a tiny threshold keeps
    # boxes to compare.
    a = detect_frames(detector, frames, score_threshold=5e-5)
    b = detect_frames(onnx, frames, score_threshold=5e-5)
    assert all(len(d.scores) > 0 for d in a)
    assert [len(d.scores) for d in a] == [len(d.scores) for d in b]
    for da, db in zip(a, b, strict=True):
        # Random scores are near-ties, so compare the boxes as sets, not in score order.
        def rows(d: object) -> np.ndarray:
            table = np.column_stack([d.boxes, d.scores])  # type: ignore[attr-defined]
            return table[np.lexsort(np.round(table[:, :4], 1).T[::-1])]

        np.testing.assert_allclose(rows(da), rows(db), atol=1e-2)
