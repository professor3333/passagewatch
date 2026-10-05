# Test Results: `passagewatch-0.3.0` on the official CFC test locations

The one-time evaluation of the declared release on the four official test locations of the
Caltech Fish Counting dataset: kenai-rightbank (another camera on the training river),
kenai-channel (a side channel), elwha and nushagak (other rivers). Run on 2026-10-04/05.

## Protocol

- **Frozen release.** `passagewatch-0.3.0` (YOLOX-Tiny with temporal input, epoch 25,
  threshold 0.4, `classical-v2` tracker, `cfc-compatible-v1` counting), exactly as recorded
  in `releases/manifests/passagewatch-0.3.0.json`. The evaluation refused to run unless the
  bundle matched that manifest. Every setting had been chosen on kenai-val and confirmed on
  kenai-holdout-v1 before this run. Nothing was selected, tuned or changed on test data.
- **Systems on the same clips:**
  - the release;
  - the single-frame release `passagewatch-0.2.0` (the neural baseline);
  - the classical baseline `classical-v2`.
- **Where it ran.** On Kaggle (`tools/kaggle/evaluate_release_on_kaggle.sh`, two sessions).
  The 87 GB of test images do not fit on the development machine. Each location's archive
  was streamed (MD5-checked), its frames inventoried by SHA-256, evaluated, and deleted.
  - Code: the sessions cloned `main` at the time. The result files do not record the commit,
    but every file that determines the results was identical on all `main` commits since
    `4220b30`, the earliest code either session could have used. Only `uv.lock` gained
    `prometheus-client`, which the evaluation does not use.
  - Detection: PyTorch on a Tesla T4 GPU.
- **Runtime parity on test data.** For 5 clips per location, the release was also run with
  its ONNX export on the CPU, the deployed runtime: **0 of 20 clips** had different counts.
- **Completeness.** All 1,021 test clips were evaluated, and none was quarantined.
- **Statistics.** nMAE per location with 95% intervals from a bootstrap over clips; the
  macro average over the four locations (the headline) with a bootstrap stratified by
  location; and paired differences on the same resamples.
- **Evidence.**
  - Per-clip counts: `releases/evaluations/passagewatch-0.3.0-test/`.
  - Summary: `releases/evaluations/passagewatch-0.3.0-test.json`, written by
    `scripts/summarize_release_evaluation.py`.

## Results

Directional nMAE (lower is better):

| Location | Clips | Passages | **Release 0.3.0** | Single-frame 0.2.0 | `classical-v2` |
|---|---:|---:|---|---|---|
| elwha | 223 | 334 | **0.168** [0.124, 0.211] | 1.183 [0.997, 1.401] | 0.419 [0.341, 0.508] |
| kenai-channel | 69 | 164 | **0.256** [0.184, 0.335] | 5.226 [4.067, 6.596] | 0.506 [0.393, 0.633] |
| kenai-rightbank | 657 | 2,144 | **0.075** [0.062, 0.089] | 1.115 [0.971, 1.271] | 0.241 [0.212, 0.274] |
| nushagak | 72 | 2,654 | **0.355** [0.312, 0.396] | 0.499 [0.464, 0.532] | 0.384 [0.344, 0.424] |
| **Macro average** | 1,021 | 5,296 | **0.213** [0.189, 0.238] | 2.006 [1.708, 2.350] | 0.388 [0.351, 0.428] |

Paired differences of the macro average:

| Comparison | Difference [95% CI] |
|---|---|
| Release − `classical-v2` | **−0.174 [−0.215, −0.137]** |
| Release − single-frame 0.2.0 | −1.792 [−2.135, −1.498] |

The release has the lowest error at every location. Missed and false passages:

| Location | Release | Single-frame 0.2.0 | `classical-v2` |
|---|---|---|---|
| elwha | 46 missed / 10 false | 96 / 299 | 46 / 94 |
| kenai-channel | 19 / 23 | 27 / 830 | 46 / 37 |
| kenai-rightbank | 111 / 49 | 166 / 2,225 | 203 / 314 |
| nushagak | 915 / 26 | 1,242 / 81 | 999 / 19 |

### The CFC authors' published methods on the same clips

The CFC authors published the tracks of their two methods. PassageWatch's evaluator
reproduces CFC's official evaluator exactly on them (`docs/counting_policy.md`), so their
nMAE on the same test clips is directly comparable:

| Location | **Release 0.3.0** | CFC Baseline | CFC Baseline++ |
|---|---:|---:|---:|
| elwha | **0.168** | 0.323 | 0.213 |
| kenai-channel | 0.256 | 0.530 | **0.122** |
| kenai-rightbank | 0.075 | 0.118 | **0.037** |
| nushagak | 0.355 | 0.140 | **0.088** |
| Macro average | 0.213 | 0.278 | **0.115** |

The release beats CFC's Baseline on average and CFC's Baseline++ on elwha. CFC's Baseline++
is better overall, especially on nushagak. It is not a like-for-like comparison: CFC's
detector is larger (YOLOv5m against YOLOX-Tiny) and was trained on all fifteen kenai-train
days, against five here.

## Findings

1. **The temporal input is what makes the model transfer.** Without it, the single-frame
   model counts static clutter on unfamiliar sonars as moving fish: 830 false passages
   against 164 real ones on kenai-channel, and 2,225 against 2,144 on kenai-rightbank. With
   the background-subtracted and motion channels, false passages stay at 10–49 per location.
   This is a much larger effect than the 44% gain measured on Kenai development data.
2. **The release generalizes best to the same river.** kenai-rightbank, another camera on
   the training river and days, has the lowest error (0.075). The other rivers are harder
   (0.168 and 0.355).
3. **Dense traffic is the main remaining weakness.** On nushagak (2,654 passages in 72
   clips) every system misses about a third or more of the fish. The release misses 915 and
   adds only 26. Fish passing close together merge into one track, the failure that Stage 8
   identified and that experiments 2 and 5 could not fix with gating alone.
4. **Development numbers were a fair guide.** The holdout's 0.117 lies between
   kenai-rightbank's 0.075 and the other rivers. The single-day kenai-val number (0.066) was
   optimistic, as the holdout had shown.

## What this does and does not establish

- **Established:** on these four sites the release counts better than both of the project's
  baselines, by a wide margin and with tight intervals. The deployed ONNX runtime counts like
  the evaluated one on test clips too.
- **Not established:**
  - performance on continuous, mostly empty footage (CFC clips were selected around fish
    activity);
  - performance on other sonars, rivers or seasons;
  - anything about species.
- **The test set is now a known benchmark.** Its results and failures have been inspected.
  Any later change chosen with these numbers in view must be confirmed on fresh held-out
  data before it is claimed as an improvement.

## Frame inventories

SHA-256 of the frame inventories written while streaming (kept with the run outputs, not
committed):

| Inventory | SHA-256 |
|---|---|
| `test-elwha-1.parquet` | `61b4f489d7b38bbea3ec36e8ed9daf8710df6b8f515bd0f1a0e6c2b59455ef0f` |
| `test-kenai-channel-1.parquet` | `f923edcd36ad595740c0013c65aff2c0ee8b1a40f97392dfcad9bd8283ec4772` |
| `test-kenai-rightbank-1.parquet` | `b8f7ac3cee277acb8bf166882981bc10a6773dbbdc8f89c7f693d1f40eb6475e` |
| `test-nushagak-1.parquet` | `1b57ff6cda968067c445f3c55fa57fab2fa55d4bb3652b06f5154e4c524c023a` |
