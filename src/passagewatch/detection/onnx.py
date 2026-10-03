"""ONNX export of a trained detector, and an ONNX Runtime stand-in for the PyTorch model.

Only the network moves to ONNX. :class:`OnnxModel` is called like the PyTorch model, a
``(n, 3, H, W)`` float tensor in, the decoded ``(n, anchors, 5 + classes)`` tensor out, so
:func:`passagewatch.inference.neural.detect_frames` keeps the exact same letterboxing, box
decoding, NMS and mapping back to frame pixels. YOLOX decodes its grid inside the forward
pass in inference mode, and that decoding is part of the exported graph.

The export fixes the input size (the checkpoint's) and leaves the batch dimension dynamic.
CPU execution only: the serving target is CPU, and other execution providers could change
scores.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from passagewatch.inference.neural import LoadedDetector, load_detector

OPSET = 17
INPUT_NAME = "images"
OUTPUT_NAME = "outputs"


def export_onnx(checkpoint: Path, out: Path, *, opset: int = OPSET) -> LoadedDetector:
    """Export ``checkpoint``'s network to ``out``; returns the loaded PyTorch detector."""
    detector = load_detector(checkpoint, torch.device("cpu"))
    size = detector.input_size
    example = torch.zeros(1, 3, size.height, size.width)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tracer warnings about Python-side shape logic
        torch.onnx.export(
            detector.model,
            (example,),
            str(tmp),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    tmp.replace(out)
    return detector


class OnnxModel:
    """Runs an exported detector with ONNX Runtime on the CPU, called like the torch model."""

    def __init__(self, path: Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        array = batch.detach().cpu().numpy().astype(np.float32, copy=False)
        (outputs,) = self.session.run([OUTPUT_NAME], {INPUT_NAME: array})
        return torch.from_numpy(outputs)

    def eval(self) -> OnnxModel:  # parity with nn.Module, for callers that set eval mode
        return self


def onnx_detector(
    detector: LoadedDetector, path: Path, threads: int | None = None
) -> LoadedDetector:
    """``detector`` (loaded from the same checkpoint) with its network replaced by ONNX."""
    from dataclasses import replace

    return replace(detector, model=OnnxModel(path, threads), device=torch.device("cpu"))  # type: ignore[arg-type]
