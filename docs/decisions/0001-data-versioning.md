# 0001: Data versioning without DVC for publisher data

- **Status:** accepted (Stage 2)
- **Revisit:** at Stage 4, when the first trained checkpoints and derived datasets exist

## Context

PassageWatch must be able to say exactly which data a result came from, and must never
commit raw archives, frames, or weights to git. The usual tool for this is DVC, which
stores content-hashed pointers in git and the files themselves in a remote cache.

The data used so far comes entirely from the publisher:

- CaltechDATA hosts every file with a fixed size and MD5, and the records are versioned
  (v1.0 `1y23m-j8r69`, v1.1 `g945x-41103`).
- The full data is large (Kenai train/val ≈ 44 GB, all raw imagery ≈ 131 GB). A DVC remote
  would hold a second copy of data that the publisher already serves and checksums.
- This is a one-person project with no shared storage. A DVC remote would be an extra
  service to pay for and maintain, holding nothing that cannot be downloaded again.

The repository already records a chain of hashes from the publisher's files to every clip:

1. `configs/data/cfc_sources.yaml`: size and MD5 of every publisher file. Downloads are
   rejected unless they match.
2. `data/manifests/inventory/cfc/*.parquet`: the SHA-256 of every extracted file.
3. `data/manifests/validation/cfc/*.json`: the validation result for every clip.
4. `data/manifests/splits/cfc/<version>.{parquet,json}`: one row per clip with its
   partition, status, and `gt.txt` SHA-256. The sidecar holds the row-content hash and the
   SHA-256 of the validation report and inventories it was built from. Versions are
   immutable.

## Decision

**Do not use DVC for publisher data.** It is reproduced by downloading it again and
verifying it against the committed checksums. The manifest's content hash identifies the
dataset snapshot, and release manifests will record it.

**Do not add DVC yet.** Artifacts that cannot be downloaded again will be content-addressed
by SHA-256 and recorded in release manifests. These are trained checkpoints, expensive
preprocessed caches, calibration sets, and validated user corrections. Whether they also
need DVC (or another remote) is decided at Stage 4, when the first ones exist. The deciding
question is whether they must move between machines, for example from a GPU host back to
this one.

## Consequences

- Fewer tools and no remote storage cost. `git clone` + `make data-tiny` + the committed
  manifests reproduce the exact dataset snapshot.
- If the publisher changes or removes a file, the MD5 check fails loudly instead of the data
  changing silently. A new publisher version means new entries in `cfc_sources.yaml`, new
  inventories, and a new manifest version.
- Large derived data stays out of git (`data/interim/`, `data/processed/`, `data/cache/` are
  ignored), but nothing yet versions it. That gap is accepted until Stage 4 and must be
  closed before a model is released.
