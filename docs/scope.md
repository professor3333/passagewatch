# PassageWatch Version 1 — Scope

This document fixes what the first version does and does not do. Anything not listed
under **In scope** is out of scope for Version 1 unless this document is amended.

## Purpose

Help a fisheries technician turn a short sonar recording into **directional
fish-passage counts they can verify, correct, and defend**. The system proposes counts;
a human reviews them; the exported report keeps automatic and reviewed results apart and
traceable to the exact pipeline that produced them.

## Users and workflow

| User | Version 1 use |
|---|---|
| Fisheries technician | Upload a recording, review flagged trajectories, correct counts, export a report |
| Fisheries biologist | Inspect recordings with unusual counts; compare summaries |
| Research team | Re-run the documented pipeline on public data |
| Program supervisor | Audit corrections and trace a report to its model and pipeline versions |

The workflow deliberately matches existing practice:

> export recording from the sonar software → upload → confirm counting setup → analyze →
> review → export CSV/JSON → import into the team's existing analysis system

## Inputs

| Input | Version 1 definition |
|---|---|
| Recording | A **short, already-rendered grayscale sonar recording**, either an MP4 (H.264) video or a ZIP archive of ordered still frames (JPEG/PNG) in the CFC layout |
| Frame timing | Taken from container timestamps (video) or from a declared frame rate (frame archives). Frame order and timing are preserved exactly; frames are never silently dropped |
| Counting line | A vertical line at a normalized horizontal position `line_x_normalized`. The default and benchmark-compatible value is `0.5` |
| Orientation | Optional `upstream_direction`: `left` or `right`, meaning the image direction that corresponds to upstream |

**"Short" means:** by default at most **5 minutes** and **500 MB** per upload. The public
demo uses a stricter limit. Both limits are configuration, not code. (CFC clips average
roughly 40 seconds.)

The user confirms the counting line and orientation on a preview frame **before** analysis
starts. Both values are stored with the job and repeated in every export.

## Outputs

| Output | Definition |
|---|---|
| **Directional counts** | `rightward` and `leftward` passage counts (image space). When orientation is configured, these are also reported as `upstream`, `downstream`, and `net = upstream − downstream`. Counts follow the versioned [counting policy](counting_policy.md) |
| **Trajectories** | For each track: its detections over time (frame index, timestamp, box, score), its count contribution, and its review state. A track ID identifies a trajectory within one analyzed recording only — never an individual fish across recordings |
| **Review flags** | Each track has one of three states: `suggested` (strong evidence), `needs_review` (ambiguous or low reliability), or `unresolved` (not enough evidence to assign a direction). Tracks are ranked by a **review score**. The score is not presented as a probability unless a calibrator has been validated |
| **Evidence** | For every flagged track, the timestamps (and replayable segment) where it starts, crosses the line, and ends |
| **Report export (CSV and JSON)** | Recording identifier and SHA-256; counting configuration (policy version, line position, orientation); **automatic counts**; **reviewed counts**; remaining unresolved cases; model, pipeline, and configuration versions; evidence timestamps; review history summary |

Original automatic predictions are immutable. Every human correction is an appended
review event that produces a new reviewed revision.

## Review actions

A reviewer can **accept** a proposed track, **reject** it (for example, a false detection
or a duplicate), **change its direction**, **merge** fragments of one fish, **mark it
unresolved**, and **add** a missed passage by timestamp. Each action records the reviewer,
the time, the base revision, and an optional reason.

## Out of scope for Version 1

- Species identification. The label space is *fish*, not species.
- Raw sonar vendor formats (for example ARIS or DIDSON files) and direct sonar-hardware
  integration. Users export rendered video or frames from their existing software.
- Live or continuous streaming analysis. Version 1 analyzes uploaded files as batch jobs.
- Individual-fish re-identification across recordings.
- Fish length or biomass estimation.
- Multi-host or distributed processing. Version 1 targets one CPU host with one worker.
- Claims about false-alarm rates on long, mostly empty monitoring footage. The benchmark
  data was selected around fish activity (see the [dataset limitations](design.md#quality-and-limitations)).

## Success criteria

The measurable targets are in [design.md §27](design.md#27-success-metrics). All of them
are planning assumptions until they are measured, and no result is reported before it has been measured.
