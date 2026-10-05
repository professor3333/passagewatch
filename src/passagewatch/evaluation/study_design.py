"""Clip selection for the usability study (``docs/usability_study.md``, declared beforehand).

1. Eligible clips: 150-450 frames lasting 15-45 s, 1-8 reference passages, not named in the
   documentation.
2. Strata: camera (LeftFar or LeftNear) x reference passages (1-2, 3-4, 5-8).
3. Trial clips: two per stratum, drawn with a seeded generator (12 clips).
4. Sets: each of the two sets gets one clip of every stratum, so camera mix and passage mix
   match by construction. Of the 2^6 = 64 splits, those keeping total reference passages
   (within 2), total duration (within 10%) and the release's total automatic count error
   (within 2) equal are allowed, and the most balanced one is chosen (smallest summed
   normalized differences; ties by clip names).
5. Practice clips: one per camera, drawn from the remaining eligible clips.

(Amended on 2026-10-05, before any session: the first rule drew 14 clips round-robin and set
two aside for practice, which left an odd number of clips per camera, so no split could
match the camera mix. Eligibility also limits duration to 15-45 s, because frame rates vary:
a 200-frame clip can last 60 s, and then no split could match total duration.)
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from passagewatch.ingestion.metadata import ClipMetadata

PASSAGE_BINS = ((1, 2), (3, 4), (5, 8))
MIN_FRAMES, MAX_FRAMES = 150, 450
MIN_SECONDS, MAX_SECONDS = 15.0, 45.0
SEED = 0


@dataclass(frozen=True)
class Candidate:
    clip_name: str
    num_frames: int
    framerate: float
    reference: tuple[int, int]  # right, left
    automatic: tuple[int, int]  # the release's counts

    @property
    def camera(self) -> str:
        match = re.search(r"_(LeftFar|LeftNear|RightFar|RightNear)_", self.clip_name)
        return match[1] if match else "other"

    @property
    def passages(self) -> int:
        return sum(self.reference)

    @property
    def automatic_error(self) -> int:
        return abs(self.automatic[0] - self.reference[0]) + abs(
            self.automatic[1] - self.reference[1]
        )

    @property
    def seconds(self) -> float:
        return self.num_frames / self.framerate

    @property
    def stratum(self) -> tuple[str, int]:
        bin_index = next(i for i, (lo, hi) in enumerate(PASSAGE_BINS) if lo <= self.passages <= hi)
        return self.camera, bin_index


def eligible(candidates: list[Candidate], documented: set[str]) -> list[Candidate]:
    return [
        c
        for c in candidates
        if MIN_FRAMES <= c.num_frames <= MAX_FRAMES
        and MIN_SECONDS <= c.seconds <= MAX_SECONDS
        and PASSAGE_BINS[0][0] <= c.passages <= PASSAGE_BINS[-1][1]
        and c.clip_name not in documented
    ]


def _by_stratum(pool: list[Candidate], seed: int) -> dict[tuple[str, int], list[Candidate]]:
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, int], list[Candidate]] = {}
    for c in sorted(pool, key=lambda c: c.clip_name):
        strata.setdefault(c.stratum, []).append(c)
    for key in sorted(strata):
        rng.shuffle(strata[key])
    return strata


def select(
    pool: list[Candidate], seed: int = SEED
) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """Practice clips (one per camera) and the two matched trial sets of six."""
    strata = _by_stratum(pool, seed)
    if len(strata) != 6 or any(len(m) < 2 for m in strata.values()):
        raise ValueError("every one of the six strata needs at least two eligible clips")
    pairs = {key: (strata[key][0], strata[key][1]) for key in sorted(strata)}
    s1, s2 = match_sets(pairs)
    used = {c.clip_name for c in (*s1, *s2)}
    rest = [c for c in pool if c.clip_name not in used]
    rng = np.random.default_rng(seed + 1)
    practice = []
    for camera in sorted({c.camera for c in pool}):
        options = sorted((c for c in rest if c.camera == camera), key=lambda c: c.clip_name)
        practice.append(options[int(rng.integers(len(options)))])
    return practice, s1, s2


def cameras(clips: list[Candidate]) -> list[str]:
    return sorted(c.camera for c in clips)


def match_sets(
    pairs: dict[tuple[str, int], tuple[Candidate, Candidate]],
) -> tuple[list[Candidate], list[Candidate]]:
    """One clip of every stratum's pair into each set, as balanced as the tolerances allow."""
    keys = sorted(pairs)
    clips = [c for key in keys for c in pairs[key]]
    total_seconds = sum(c.seconds for c in clips)
    best: tuple[float, tuple[str, ...]] | None = None
    for choice in itertools.product((0, 1), repeat=len(keys)):
        if choice[0] == 1:  # each split once: the first stratum's first clip is in set 1
            continue
        a = [pairs[k][i] for k, i in zip(keys, choice, strict=True)]
        b = [pairs[k][1 - i] for k, i in zip(keys, choice, strict=True)]
        d_pass = abs(sum(c.passages for c in a) - sum(c.passages for c in b))
        d_sec = abs(sum(c.seconds for c in a) - sum(c.seconds for c in b))
        d_err = abs(sum(c.automatic_error for c in a) - sum(c.automatic_error for c in b))
        if d_pass > 2 or d_sec > 0.1 * total_seconds / 2 or d_err > 2:
            continue
        score = d_pass / 2 + d_sec / (0.1 * total_seconds / 2) + d_err / 2
        key = tuple(c.clip_name for c in a)
        if best is None or (score, key) < best:
            best = (score, key)
    if best is None:
        raise ValueError("no split of the trial clips meets the matching rules")
    first = set(best[1])
    return [c for c in clips if c.clip_name in first], [
        c for c in clips if c.clip_name not in first
    ]


def latin_square(participants: list[str]) -> dict[str, list[dict[str, str]]]:
    """Blocks per participant: rows (M,S1)(A,S2), (A,S1)(M,S2), (M,S2)(A,S1), (A,S2)(M,S1)."""
    rows = [
        (("manual", "S1"), ("assisted", "S2")),
        (("assisted", "S1"), ("manual", "S2")),
        (("manual", "S2"), ("assisted", "S1")),
        (("assisted", "S2"), ("manual", "S1")),
    ]
    return {
        p: [{"condition": c, "set": s} for c, s in rows[i % len(rows)]]
        for i, p in enumerate(participants)
    }


def candidates_from_report(
    report: dict[str, Any], metadata: dict[str, ClipMetadata]
) -> list[Candidate]:
    """Candidates from an ``evaluate_neural.py`` report: its selected epoch and threshold."""
    best = report["best"]
    (result,) = (
        r
        for r in report["results"]
        if r["epoch"] == best["epoch"] and r["threshold"] == best["threshold"]
    )
    return [
        Candidate(
            clip_name=c["clip_name"],
            num_frames=metadata[c["clip_name"]].num_frames,
            framerate=metadata[c["clip_name"]].framerate,
            reference=(c["reference"][0], c["reference"][1]),
            automatic=(c["predicted"][0], c["predicted"][1]),
        )
        for c in result["clips"]
    ]


def documented_names(texts: list[str], names: list[str]) -> set[str]:
    """Clip names (or their recording prefix) that appear in any documentation text."""
    return {n for n in names if any(n in t for t in texts)}


def assign_codes(
    practice: list[Candidate], s1: list[Candidate], s2: list[Candidate], seed: int = SEED
) -> tuple[dict[str, Candidate], list[str], dict[str, list[str]]]:
    """Neutral codes and a fixed shuffled presentation order.

    Practice clips are ``p01``, ``p02`` (block 1, block 2); trial clips ``c01``-``c12`` in a
    shuffled order across both sets, so a code reveals neither set nor stratum.
    """
    rng = np.random.default_rng(seed + 2)
    trial = sorted((*s1, *s2), key=lambda c: c.clip_name)
    order = rng.permutation(len(trial))
    codes = {c.clip_name: f"c{i + 1:02d}" for i, c in zip(order, trial, strict=True)}
    clips = {f"p{i + 1:02d}": c for i, c in enumerate(practice)}
    clips |= {codes[c.clip_name]: c for c in trial}
    sets = {}
    for name, members in (("S1", s1), ("S2", s2)):
        keys = sorted(codes[c.clip_name] for c in members)
        sets[name] = [keys[i] for i in rng.permutation(len(keys))]
    return clips, sorted(k for k in clips if k.startswith("p")), sets
