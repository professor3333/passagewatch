"""Write the README's SVG figures from the committed results, or check they are current.

Examples:
    uv run python scripts/make_readme_figures.py           # rewrite docs/images/*.svg
    uv run python scripts/make_readme_figures.py --check   # exit 1 if any figure is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from passagewatch.visualization.readme_figures import all_figures, load_test_results

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION = REPO_ROOT / "releases/evaluations/passagewatch-0.3.0-test.json"
CFC_REFERENCE = REPO_ROOT / "tests/regression/cfc_official_nmae_eccv22.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs/images")
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument("--cfc-reference", type=Path, default=CFC_REFERENCE)
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)

    figures = all_figures(load_test_results(args.evaluation, args.cfc_reference))
    stale = []
    for name, svg in figures.items():
        path = args.out / name
        if args.check:
            if not path.exists() or path.read_text() != svg:
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(svg)
            print(f"wrote {path}")
    if stale:
        print("stale figures (run scripts/make_readme_figures.py): " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
