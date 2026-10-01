"""Produce the official nMAE reference by running CFC's own evaluator.

This runs outside the project environment, because CFC's pinned TrackEval needs
NumPy < 1.24 (Python <= 3.10). It writes the per-clip reference that
``tests/regression/test_official_nmae.py`` compares PassageWatch against.

Setup (from this directory):
    git clone https://github.com/JonathonLuiten/TrackEval.git lib/TrackEval
    git -C lib/TrackEval checkout bcd03a6cc5f4fa0074da89d95e859fe77e264c3e
    gh api repos/visipedia/caltech-fish-counting/contents/CFC/evaluate.py?ref=81380c9 \\
        --jq .content | base64 -d > evaluate.py
    uv venv -p 3.10 .venv && VIRTUAL_ENV=.venv uv pip install numpy==1.23.5 scipy

Run:
    .venv/bin/python run_official.py ../../data/extracted/cfc out.json
"""

import json
import sys

import evaluate as ev  # CFC/evaluate.py; it puts lib/TrackEval on sys.path
import trackeval

TRACKERS = ["baseline", "baseline++"]
LOCATIONS = ["kenai-val", "kenai-rightbank", "kenai-channel", "elwha", "nushagak"]


def main(extracted: str, out_path: str) -> None:
    results = {}
    for tracker in TRACKERS:
        for location in LOCATIONS:
            ds = ev.get_default_ds_config(
                f"{extracted}/fish_counting_annotations/annotations/{location}",
                f"{extracted}/fish_counting_results/results/{location}",
                tracker,
            )
            metrics = ev.get_default_metrics_config()
            metrics["PRINT_CONFIG"] = False
            metrics["METRICS"] = ["nMAE"]
            with open(f"{extracted}/fish_counting_metadata/metadata/{location}.json") as fh:
                clips = json.load(fh)
            # Same configuration as CFC's evaluate(): sequence lengths and frame sizes.
            ds["SEQ_INFO"] = {c["clip_name"]: c["num_frames"] for c in clips}
            metrics["SEQ_DIMS"] = {c["clip_name"]: (c["width"], c["height"]) for c in clips}
            evaluator = trackeval.Evaluator(ev.get_default_eval_config(quiet=True))
            output, _ = evaluator.evaluate(
                [trackeval.datasets.MotChallenge2DBox(ds)], [ev.nMAE(metrics)]
            )
            per_seq = output["MotChallenge2DBox"][tracker]
            results.setdefault(tracker, {})[location] = {
                seq: [v["pedestrian"]["nMAE"]["nMAE_numer"], v["pedestrian"]["nMAE"]["nMAE_denom"]]
                for seq, v in sorted(per_seq.items())
                if seq != "COMBINED_SEQ"
            }
            combined = per_seq["COMBINED_SEQ"]["pedestrian"]["nMAE"]
            print(
                f"{tracker:11} {location:16} {combined['nMAE_numer']:5d} / "
                f"{combined['nMAE_denom']:5d} = {combined['nMAE']:.6f}"
            )
    reference = {
        "source": (
            "CFC/evaluate.py at visipedia/caltech-fish-counting 81380c9 with TrackEval "
            "bcd03a6, run on fish_counting_results.tar.gz "
            "(md5 ef8d517ad45419edce7af2e7dc5016be)"
        ),
        "format": "tracker -> location -> clip -> [nMAE_numer, nMAE_denom]",
        "results": results,
    }
    with open(out_path, "w") as fh:
        fh.write(json.dumps(reference, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
