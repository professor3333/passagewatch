# Vendored YOLOX model code

This directory contains the YOLOX model definition (backbone, neck, head, and loss) from
[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), commit
`6ddff48` (2025-06-08), under the Apache License 2.0 (`LICENSE` in this directory).

Only the model code is vendored. YOLOX's training engine, data pipeline, and install-time
dependencies (a C++ extension, `onnx-simplifier==0.4.10`, `loguru`, `tensorboard`,
`pycocotools`, …) are not used, so the package installs on Python 3.12 with current PyTorch.

| File | Source | Changed |
|---|---|---|
| `darknet.py` | `yolox/models/darknet.py` | no |
| `network_blocks.py` | `yolox/models/network_blocks.py` | no |
| `yolo_pafpn.py` | `yolox/models/yolo_pafpn.py` | no |
| `yolox.py` | `yolox/models/yolox.py` | no |
| `yolo_head.py` | `yolox/models/yolo_head.py` | yes, see its header |
| `losses.py` | `yolox/models/losses.py` | yes, see its header |
| `boxes.py` | `yolox/utils/boxes.py`, `yolox/utils/compat.py` (excerpts) | yes, see its header |

The changes are limited to running on any PyTorch device (CUDA, CPU, and Apple MPS): a
coordinate grid cached on one device is recomputed on another, and legacy tensor type strings
are replaced with `.to(...)`. Their effect is checked by `tests/unit/detection`.

This code is excluded from the project's lint and strict type checks; the PassageWatch code
that uses it (`passagewatch.detection.neural`) is not. Pretrained COCO weights are listed with
their SHA-256 in `configs/training/pretrained.yaml`.
