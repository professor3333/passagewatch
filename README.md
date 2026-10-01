# PassageWatch

[![ci](https://github.com/professor3333/passagewatch/actions/workflows/ci.yml/badge.svg)](https://github.com/professor3333/passagewatch/actions/workflows/ci.yml)

Auditable fish-passage counting and review for sonar video.

PassageWatch detects and tracks fish in sonar recordings, estimates **directional
passage counts** (left/right, mapped to upstream/downstream when orientation is
configured), and helps technicians verify those counts through timestamped
evidence and a prioritized review queue. Automatic and human-reviewed counts stay
distinguishable, and every report is traceable to the model and pipeline versions
that produced it.

> **Status: early development (Stage 1 of 12).** The scope and counting policy are
> defined; the pipeline is not implemented yet. No results have been measured, and this
> README will report numbers only after they are. See the [roadmap](docs/roadmap.md).

## Planned pipeline

Recording → validation → frame decoding → preprocessing → neural detection →
tracking → directional counting → review prioritization → human corrections → export.

## Documentation

| Document | Contents |
|---|---|
| [Scope](docs/scope.md) | What Version 1 accepts, produces, and deliberately excludes |
| [Counting policy](docs/counting_policy.md) | Coordinate convention, the `cfc-compatible-v1` counting rule, direction mapping, and the evaluation metric |
| [Dataset card](docs/dataset_card.md) | CFC source, verified format facts, validation, splits, manifests, and the Kenai development subset |
| [Classical baseline](docs/classical_baseline.md) | Method, tuning protocol, and measured results of the non-learned baseline |
| [Roadmap](docs/roadmap.md) | Twelve stages, each with its completion test |
| [Design](docs/design.md) | The full system design: data, models, evaluation, service, and operations |

## Data

Development uses the [Caltech Fish Counting (CFC) dataset](https://data.caltech.edu/records/g945x-41103)
(MIT license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Data is downloaded from
the publisher and never committed to this repository.

```bash
make data-tiny       # ~1.5 GB: tiny subset, MOT annotations, clip metadata, baseline results
make validate-tiny   # check annotations, metadata, and every frame; write the report
make manifest-tiny   # validate, then write the versioned split manifest (tiny-v1)
make data-kenai-dev  # stream 44 GB once, keep ~19 GB of Kenai train/val frames (full-v2)
uv run python scripts/view_clip.py --location kenai-train --list    # list clips
uv run python scripts/view_clip.py --location kenai-train --sheet   # boxes on frames, as PNG
```

Every publisher file is listed with its size and MD5 in
[`configs/data/cfc_sources.yaml`](configs/data/cfc_sources.yaml). Downloads resume after
interruption. A file is kept only after its checksum matches, and every extracted file is
recorded with its SHA-256 in `data/manifests/inventory/`.

Validation (`scripts/validate_data.py`) converts the 1-based MOT annotations to the internal
convention once and checks them against the clip metadata and the frames. A clip with any
error is **quarantined** with its reasons, never silently skipped. The reports are in
`data/manifests/validation/cfc/`, and the issue codes are listed in the
[dataset card](docs/dataset_card.md#validation).

Each clip's partition is recorded in an immutable, versioned manifest
(`data/manifests/splits/cfc/<version>.parquet`). The partition follows the publisher's split
by location, and test-location clips are marked as never usable for tuning. The schema and
versioning rules are in the [dataset card](docs/dataset_card.md#splits-and-manifests).

> The `tiny` bundle includes clips from the official **test** locations. Those clips are
> never used for training or tuning.

## Counting and evaluation

Counting follows the versioned policy `cfc-compatible-v1`, which reproduces the CFC
benchmark's start/end-side rule: a fish that crosses the line and returns contributes zero.
The nMAE evaluator matches CFC's official evaluator exactly on its published baseline tracks
([details](docs/counting_policy.md#validation-against-the-official-evaluator)):

```bash
uv run python scripts/evaluate_counts.py --tracker baseline++   # kenai-val only by default
```

The tuned classical baseline (`classical-v2`, tuned on kenai-train only) reaches nMAE
0.235 on kenai-val. CFC's published learned Baseline reaches 0.049 on the same clips
([results](docs/classical_baseline.md#results)):

```bash
uv run python scripts/run_classical.py --manifest full-v2 --partitions val \
    --frames-dir data/extracted/cfc/kenai-dev-v1 --config configs/tracking/classical-v2.yaml
```

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install    # create the virtual environment and install dependencies
make check      # lint, type-check, and run fast tests
```

## License

MIT — see [LICENSE](LICENSE).
