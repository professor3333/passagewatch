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
