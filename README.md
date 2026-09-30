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
| [Roadmap](docs/roadmap.md) | Twelve stages, each with its completion test |
| [Design](docs/design.md) | The full system design: data, models, evaluation, service, and operations |

## Data

Development uses the [Caltech Fish Counting (CFC) dataset](https://data.caltech.edu/records/g945x-41103)
(MIT license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Data is downloaded from
the publisher and never committed to this repository.

```bash
make data-tiny       # ~1.5 GB: tiny subset, MOT annotations, clip metadata, baseline results
make validate-tiny   # check annotations, metadata, and every frame; write the report
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

> The `tiny` bundle includes clips from the official **test** locations. Those clips are
> never used for training or tuning.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install    # create the virtual environment and install dependencies
make check      # lint, type-check, and run fast tests
```

## License

MIT — see [LICENSE](LICENSE).
