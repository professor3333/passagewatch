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
  PassageWatch's counting rule and nMAE metric reproduce the logic of the official
  evaluator (`CFC/evaluate.py`).
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

## Python dependencies

Python dependencies are installed from PyPI and pinned in `uv.lock`. Each package is
distributed under its own license. Detector and tracker code (YOLOX, ByteTrack), along with
any pretrained weights, will be listed here with their pinned versions and licenses when
they are added.
