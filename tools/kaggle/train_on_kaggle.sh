#!/usr/bin/env bash
# Train a PassageWatch detector on a Kaggle GPU notebook. Run from the repository root of a
# fresh clone (the notebook cell in docs/training.md does this). Steps:
#   1. a Python 3.12 environment with the locked dependencies, and the CUDA build of the
#      locked torch/torchvision versions;
#   2. CFC labels (checksum-verified) and the kenai-dev-v1-train frames, streamed from
#      CaltechDATA; every frame is checked against the committed SHA-256 inventory;
#   3. training, resuming from <run>/latest.pt if it exists.
# Use .venv/bin/python, never `uv run`: uv run would re-sync to the CPU-only torch.
set -euo pipefail
# Print progress immediately: Kaggle background runs capture output through a pipe.
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/training/yolox-tiny-v1.yaml}"
CUDA_INDEX="${CUDA_INDEX:-cu126}"
WORK="${WORK:-/kaggle/working}"
DATA="${DATA:-/tmp/passagewatch-data}"
RESUME_FROM="${RESUME_FROM:-}"   # optional: a previous run directory that has latest.pt

echo "== commit: $(git log -1 --format='%H %s')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== environment"
pip install --quiet --disable-pip-version-check uv
uv sync --locked --no-dev --no-install-package torch --no-install-package torchvision
read -r TORCH TORCHVISION < <(.venv/bin/python - <<'PY'
import tomllib
lock = tomllib.load(open("uv.lock", "rb"))
versions = {p["name"]: p["version"].split("+")[0] for p in lock["package"] if p["name"] in ("torch", "torchvision")}
print(versions["torch"], versions["torchvision"])
PY
)
uv pip install --python .venv/bin/python --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}" \
  "torch==${TORCH}+${CUDA_INDEX}" "torchvision==${TORCHVISION}+${CUDA_INDEX}"
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "No CUDA GPU visible: set the notebook accelerator to a GPU."
print("torch", torch.__version__, "on", torch.cuda.get_device_name(0))
PY

echo "== data"
EXTRACT="${DATA}/extracted/cfc"
.venv/bin/python scripts/download_data.py --bundle labels \
  --raw-dir "${DATA}/raw/cfc" --extract-dir "${EXTRACT}" --inventory-dir "${DATA}/inventory"
.venv/bin/python scripts/stream_subset.py --config configs/data/kenai_subset_train.yaml \
  --extract-dir "${EXTRACT}" --inventory-dir "${DATA}/inventory"
.venv/bin/python scripts/verify_frames.py --root "${EXTRACT}/kenai-dev-v1-train" \
  --inventory data/manifests/inventory/cfc/kenai-dev-v1.parquet

echo "== training"
RUN="${WORK}/runs/$(basename "${CONFIG}" .yaml)"
mkdir -p "${RUN}"
if [[ -n "${RESUME_FROM}" && ! -f "${RUN}/latest.pt" ]]; then
  cp "${RESUME_FROM}"/latest.pt "${RESUME_FROM}"/metrics.jsonl "${RUN}/"
  cp "${RESUME_FROM}"/epoch-*.pt "${RUN}/" 2>/dev/null || true
fi
ARGS=(--config "${CONFIG}" --manifest full-v2 --extract-dir "${EXTRACT}"
      --frames-dir "${EXTRACT}/kenai-dev-v1-train" --out "${RUN}" --device cuda)
[[ -f "${RUN}/latest.pt" ]] && ARGS+=(--resume)
.venv/bin/python scripts/train_yolox.py "${ARGS[@]}"
echo "== done: ${RUN}"
ls -la "${RUN}"
