# Third-Party Notices

PassageWatch uses the third-party data and software listed below. Datasets are downloaded
from their publishers by `scripts/download_data.py` and are **not** redistributed in this
repository.

## Caltech Fish Counting Dataset (CFC)

- **Publisher:** California Institute of Technology, via CaltechDATA
  - Version 1.0: <https://data.caltech.edu/records/1y23m-j8r69>
  - Version 1.1: <https://data.caltech.edu/records/g945x-41103>
- **Code and documentation:** <https://github.com/visipedia/caltech-fish-counting>
- **License:** MIT. The CaltechDATA record metadata lists the rights as `mit`. The license
  text below is reproduced from `CFC/LICENSE` in the repository above.
- **Used for:** training and evaluation data (sonar frames, bounding-box and track
  annotations, clip metadata) and the published ECCV 2022 baseline tracking results.
  PassageWatch's counting rule and nMAE metric reimplement the logic of the official
  evaluator (`CFC/evaluate.py`). No CFC code is vendored. The per-clip reference in
  `tests/regression/cfc_official_nmae_eccv22.json` was computed by running that evaluator
  on the published results.
- **Baseline++ input (method only):** the optional temporal input
  (`passagewatch.preprocessing.temporal`, `letterbox-temporal3-v1`) follows the 3-channel
  Baseline++ encoding of the CFC paper and `CFC/convert.py` (frame, background-subtracted
  frame, frame difference). It is reimplemented with documented differences; no CFC code is
  copied.
- **TrackEval** (<https://github.com/JonathonLuiten/TrackEval>, MIT, commit `bcd03a6`) is
  used only offline, by `tools/cfc_official_nmae/`, to produce that reference. It is not a
  dependency and is not distributed.
- **Citation:**

  > Kay, J., Kulits, P., Stathatos, S., Deng, S., Young, E., Beery, S., Van Horn, G., and
  > Perona, P. The Caltech Fish Counting Dataset: A Benchmark for Multiple-Object Tracking
  > and Counting. *European Conference on Computer Vision (ECCV)*, 2022.

```
MIT License

Copyright (c) 2022 Justin Kay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## YOLOX

- **Source:** <https://github.com/Megvii-BaseDetection/YOLOX>, commit `6ddff48`
  (2025-06-08)
- **License:** Apache License 2.0. Copyright (c) 2021-2022 Megvii Inc. The full license
  text is in `src/passagewatch/detection/yolox/LICENSE`. The repository has no `NOTICE` file.
- **Used for:** the YOLOX model definition (backbone, neck, detection head, and loss),
  **vendored** in `src/passagewatch/detection/yolox/`. Four files are unchanged. Three are
  modified (`yolo_head.py`, `losses.py`, and `boxes.py`, which is assembled from excerpts),
  and each says so in its header. The changes let the code run on any PyTorch device; the
  full list is in that directory's `README.md`.
- **Pretrained weights:** COCO-pretrained `yolox_tiny.pth` and `yolox_s.pth` from the YOLOX
  GitHub release `0.1.1rc0`, under the same license. They are downloaded at run time and
  verified by size and SHA-256 (`configs/training/pretrained.yaml`). They are not
  committed or redistributed.
- **Citation:**

  > Ge, Z., Liu, S., Wang, F., Li, Z., and Sun, J. YOLOX: Exceeding YOLO Series in 2021.
  > arXiv:2107.08430, 2021.

## Python dependencies

Python dependencies are installed from PyPI (PyTorch on Linux from PyTorch's CPU wheel index)
and pinned in `uv.lock`. Each package is distributed under its own license.

## ByteTrack (method only)

`passagewatch.tracking.bytetrack` implements the ByteTrack association method from its
paper. No ByteTrack code is vendored or installed.

> Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., and Wang, X.
> ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *European Conference
> on Computer Vision (ECCV)*, 2022.
