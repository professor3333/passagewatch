# Independent check of the documentation

The last Stage 12 item ([roadmap](roadmap.md)): a person **who did not build PassageWatch**
reads the main documents, follows the README, and reports what was unclear or did not work.
It takes about an hour. No fisheries or machine-learning background is needed.

## What to do

Note the time you start and stop each part, and anything that is unclear, wrong or
missing, however small. Do not ask the developer during the check: a question you would
have asked is a finding.

**1. Read (about 25 minutes).** Read the [README](../README.md), then the
[model card](model_card.md), the [dataset card](dataset_card.md) and the
[architecture guide](architecture.md). Then answer, without looking back:

- What problem does PassageWatch solve, and for whom?
- What does it output for a recording, and what does a technician do with it?
- What does "a fish that crosses and comes back counts zero" mean?
- How accurate is it on the official test locations, and against what baselines?
- Name two things it should **not** be used for.
- Where do the data come from, and under what license?

**2. Run (about 25 minutes).** On a machine with [uv](https://docs.astral.sh/uv/) (and,
for the second part, Docker):

```bash
git clone https://github.com/professor3333/passagewatch.git && cd passagewatch
make install && make check          # tests should pass without network or data
```

With Docker, follow the README's **Quick start**: fetch the release, start the service,
load the demo examples, open http://127.0.0.1:8000/, open the **Difficult** demo, and try to
correct its counts with the review interface. Export the report.

**3. Report (about 10 minutes).** Open a GitHub issue with the
[**Documentation check** form](https://github.com/professor3333/passagewatch/issues/new?template=documentation-check.yml).
It asks for your answers above, what failed or confused you, and how long each part took.

## What happens with the report

Each finding is fixed or answered in the documentation, and the issue is closed with links
to the changes. The check counts as done when one complete report has been received and
its findings are addressed. The report stays public in the issue tracker.

## Status

**The independent check has not been done.** No report from a person who did not build
PassageWatch has been received.

### Developer run-through (2026-10-17, not independent)

The developer followed this page on a fresh clone of `main` (commit `9507649`) on the
development Mac (Apple M1, Docker), to find what could be found without an outside reader.
It does not replace the independent check: the developer knows the answers to the questions
above.

| Step | Result | Time |
|---|---|---|
| `git clone` | Worked (a first attempt failed on a network error) | under 1 min |
| `make install && make check` | 413 tests passed | 3 min |
| Frontend `npm ci && npm test` | 37 tests passed | under 1 min |
| `fetch_bundle.py --release passagewatch-0.3.1 --activate` | Verified against the manifest | under 1 min |
| `docker compose up -d --build` | Built and healthy (API, worker, Prometheus) | 9 min |
| `load_demos.py` | Three demos analysed: clear 11/0, difficult 2/0, unfamiliar camera 4/2 (the Mac's documented result; x86 gives 4/3) | 7 min |
| `verify_deployment.py` | Release matches the manifest; example job succeeded | under 1 min |
| Review the **Difficult** demo, add a missed fish, export | Worked, but see the first finding | — |

Findings, each fixed:

- **Fast keyboard review lost decisions.** A key pressed while the previous decision was
  still saving was refused as stale (409), and the page said someone else had saved first.
  Fixed in PR #69: review actions now run one at a time. No 409 had occurred in the
  usability study or the reviewed sample.
- **The architecture guide was out of date** (release 0.2.0, no ONNX, no Prometheus, missing
  settings, Stage 9 and 11 features listed as not built). Updated.
- **The dataset card's status** still said the full imagery was not downloaded, and the card
  did not say how the test locations were handled. Updated.
- **The README** said "unseen rivers" for test locations that include two other Kenai
  cameras, did not mention the reviewed-sample result, and gave no build or demo-loading
  times. Updated. A roadmap link pointed to a README section that no longer exists. Fixed.
- **The model card's release process** did not mention release manifests, the release
  workflow or deployment checks, and did not say that review can make counts worse. Updated.
