# PassageWatch

[![ci](https://github.com/professor3333/passagewatch/actions/workflows/ci.yml/badge.svg)](https://github.com/professor3333/passagewatch/actions/workflows/ci.yml)

Auditable fish-passage counting and review for sonar video.

PassageWatch detects and tracks fish in sonar recordings, estimates **directional
passage counts** (left/right, mapped to upstream/downstream when orientation is
configured), and helps technicians verify those counts through timestamped
evidence and a prioritized review queue. Automatic and human-reviewed counts stay
distinguishable, and every report is traceable to the model and pipeline versions
that produced it.

> **Status: early development.** Nothing below the design stage is implemented yet.
> No results have been measured; this README will only report numbers once they are.

## Planned pipeline

Recording → validation → frame decoding → preprocessing → neural detection →
tracking → directional counting → review prioritization → human corrections → export.

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
