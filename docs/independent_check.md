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

**2. Run (about 25 minutes, plus about 15 for the first Docker build and the demos).** On
your own machine if you can, with [uv](https://docs.astral.sh/uv/) (and, for the second
part, Docker). If you use someone else's machine, say whose in the report.

```bash
git clone https://github.com/professor3333/passagewatch.git && cd passagewatch
make install && make check          # tests should pass without network or data
```

With Docker, follow the README's **Quick start**: fetch the release, start the service
(check that `/v1/model-info` names the release, as the README shows), load the demo
examples, and open the service in the browser. Open the **Difficult** demo and correct its
counts with the review interface as you think right. Then click **Export CSV** or **Export
JSON** and open the downloaded file.

**3. Report (about 10 minutes).** Open a GitHub issue with the
[**Documentation check** form](https://github.com/professor3333/passagewatch/issues/new?template=documentation-check.yml).
It asks for your answers above, what failed or confused you, whether `make check` passed,
whether the export worked, and how long each part took. If someone else posts it for you,
they should say so in the issue.

## What happens with the report

Each finding is fixed or answered in the documentation, and the issue is closed with links
to the changes. The check counts as done when one complete report has been received and
its findings are addressed. The report stays public in the issue tracker.

## Status

**One independent check has been done** ([issue #75](https://github.com/professor3333/passagewatch/issues/75),
2026-10-19), and its findings are addressed below. A second reader has not checked the
revised documents.

### Report 1 (issue #75)

A person who did not build PassageWatch followed this page. Conditions, as the issue states:

- **Machine:** the developer's MacBook Air (M1, Docker Desktop), not their own. Another of the
  developer's programs was using port 8000 on it.
- **Posted** by the repository owner on the reader's behalf, with the reader's answers.
- **Help:** one. When `load_demos.py` failed, the reader got help, relayed through the owner,
  to move the service to port 8010. Nothing else.
- **Time:** reading 30, running 10, reporting 5–10 minutes. The image had been built on that
  machine before, so the build was quick.

What happened:

| Part | Result |
|---|---|
| `make install && make check` | Passed |
| Quick start | `docker compose up` started without an error, but port 8000 belonged to another program: `curl /health/ready` answered "ready" from it, and `load_demos.py` failed with a 404 traceback. Worked on port 8010 |
| Difficult demo, review and export | Done; CSV/JSON downloaded. The reader counted all 13 uncounted tracks as passages: automatic 2/0, reviewed 15/0, CFC reference 5/0 |
| The six questions | 2 right (output and what the technician does; data source and license), 1 partly (no mention of sonar or of who uses it), 3 wrong: cross-and-return ("must be on the right side of the dashed line"), accuracy ("pretty much accurate", no number), and what it is not for ("training or tuning") |
| "What was unclear?" | "Everything was clear" |

Findings and what was done (PR #76):

- **A program already on port 8000 made the Quick start fail without a clear cause.** The
  port note was below the Quick start's last command. Now: a warning box in the Quick start,
  with how to check the port and how to use another; the README checks `/v1/model-info`
  for the release name instead of `/health/ready`; `load_demos.py` and `verify_deployment.py`
  stop with "is not a PassageWatch service" and the fix, instead of a 404 traceback.
- **The cross-and-return rule, the accuracy and the limits of use did not get across**,
  although the README, model card and test results state them. Now: a "What to know first"
  table near the top of the README, with a worked example of the start/end rule, the accuracy
  in plain words (count errors per 100 true passages) next to the baselines, and the list of
  what it is not for.
- **"Never used for training or tuning"**, said of the test locations in the README, the model
  card and a demo card, was read as a limit of use. Reworded ("the model never saw these
  locations during training or setting selection"). The demo card's new wording appears in
  the next release's image.
- **The README's numbered steps start with uploading a recording**, so a reader with only the
  demos had no step to follow. Now they say to open a demo and start at step 3.
- **This page** now asks for the export, the `make check` result and the machine's owner,
  and gives the build time.
- **The reader counted every uncounted track in the Difficult demo.** This is the review-page
  weakness found in the usability study and the reviewed sample
  ([results](review_sample_results.md)), the third time. Documentation cannot fix it. The
  review page now asks whether such a track is a different fish from the counted tracks at
  overlapping times ([same-fish question](review.md#same-fish-question-on-the-review-page)).
  Its effect is not measured.

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
