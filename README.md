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
(MIT license). Data is downloaded by scripts and never committed to this repository.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install    # create the virtual environment and install dependencies
make check      # lint, type-check, and run fast tests
```

## License

MIT — see [LICENSE](LICENSE).
