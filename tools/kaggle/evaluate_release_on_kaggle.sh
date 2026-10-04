#!/usr/bin/env bash
# Evaluate a declared PassageWatch release on the official CFC test locations, on a Kaggle
# GPU notebook (docs/training.md, "Evaluating a release on the test locations"). This is the
# project's frozen, one-time test evaluation: the release bundle must match its committed
# manifest, and nothing is selected. Steps:
#   1. the locked environment with the CUDA build of torch (as for training);
#   2. CFC labels and metadata;
#   3. for each location in LOCATIONS: plan chunks of whole recording days that fit the disk,
#      then for each chunk stream the location's archive keeping only those days (frames are
#      MD5-checked and SHA-256-inventoried), evaluate the release, its baseline and the
#      classical pipeline on them, and delete the frames.
# Results are written per clip to ${OUT}/results/<location>.json, so a session that stops can
# be resumed with the same output (RESUME_FROM). Use .venv/bin/python, never `uv run`.
set -euo pipefail
export PYTHONUNBUFFERED=1

BUNDLES="${BUNDLES:-/kaggle/input}"   # searched for <release>/bundle.json (any depth up to 6)
RELEASE="${RELEASE:-passagewatch-0.3.0}"
BASELINE="${BASELINE:-passagewatch-0.2.0}"
LOCATIONS="${LOCATIONS:-kenai-channel nushagak elwha kenai-rightbank}"
CLASSICAL="${CLASSICAL:-1}"          # 0 skips the classical baseline (saves CPU time)
PARITY_CLIPS="${PARITY_CLIPS:-5}"    # clips per location also run with ONNX on the CPU
BUDGET_GB="${BUDGET_GB:-auto}"       # disk per chunk; auto = 70% of the free space
CUDA_INDEX="${CUDA_INDEX:-cu126}"
WORK="${WORK:-/kaggle/working}"
DATA="${DATA:-/tmp/passagewatch-data}"
RESUME_FROM="${RESUME_FROM:-}"       # optional: a previous session's release-eval directory
OUT="${WORK}/release-eval"

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
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'set the accelerator to a GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "== release"
# Kaggle mounts an uploaded dataset under /kaggle/input with a layout that varies (an extra
# folder level, or datasets/<user>/<name>), so find the bundles instead of assuming a path.
FOUND=$(find "${BUNDLES}" -maxdepth 6 -path "*/${RELEASE}/bundle.json" -print -quit 2>/dev/null || true)
if [[ -z "${FOUND}" ]]; then
  echo "no ${RELEASE}/bundle.json under ${BUNDLES}; is the bundles dataset attached? It contains:"
  find "${BUNDLES}" -maxdepth 4 2>/dev/null | head -40
  exit 1
fi
BUNDLES=$(dirname "$(dirname "${FOUND}")")
echo "bundles: ${BUNDLES}"
for version in "${RELEASE}" "${BASELINE}"; do
  test -f "${BUNDLES}/${version}/bundle.json" || { echo "missing ${BUNDLES}/${version}"; exit 1; }
  test -f "releases/manifests/${version}.json" || { echo "missing manifest of ${version}"; exit 1; }
done

echo "== labels"
EXTRACT="${DATA}/extracted/cfc"
mkdir -p "${OUT}/results" "${OUT}/inventory" "${DATA}/chunks"
if [[ -n "${RESUME_FROM}" ]]; then cp -n "${RESUME_FROM}"/results/*.json "${OUT}/results/" 2>/dev/null || true; fi
.venv/bin/python scripts/download_data.py --bundle labels \
  --raw-dir "${DATA}/raw/cfc" --extract-dir "${EXTRACT}" --inventory-dir "${DATA}/inventory"

if [[ "${BUDGET_GB}" == "auto" ]]; then
  FREE=$(df -B1 --output=avail "${DATA}" | tail -1)
  BUDGET_GB=$(.venv/bin/python -c "print(round(0.7 * ${FREE} / 1e9, 1))")
fi
echo "== chunk budget: ${BUDGET_GB} GB (free: $(df -h --output=avail "${DATA}" | tail -1))"

EXTRA=()
[[ "${CLASSICAL}" == "1" ]] && EXTRA+=(--classical-config configs/tracking/classical-v2.yaml)
for LOCATION in ${LOCATIONS}; do
  echo "== ${LOCATION}"
  mapfile -t CHUNKS < <(.venv/bin/python scripts/plan_test_chunks.py --location "${LOCATION}" \
    --budget-gb "${BUDGET_GB}" --extract-dir "${EXTRACT}" --out-dir "${DATA}/chunks")
  for CHUNK in "${CHUNKS[@]}"; do
    NAME=$(basename "${CHUNK}" .yaml)
    echo "-- ${NAME}"
    .venv/bin/python scripts/stream_subset.py --config "${CHUNK}" \
      --extract-dir "${EXTRACT}" --inventory-dir "${OUT}/inventory"
    if [[ -z "$(ls -A "${EXTRACT}/${NAME}/${LOCATION}" 2>/dev/null)" ]]; then
      echo "no frames were kept for ${NAME}: the archive layout differs from what is expected"
      exit 1
    fi
    .venv/bin/python scripts/evaluate_release.py \
      --release-manifest "releases/manifests/${RELEASE}.json" --bundle "${BUNDLES}/${RELEASE}" \
      --baseline-manifest "releases/manifests/${BASELINE}.json" --baseline-bundle "${BUNDLES}/${BASELINE}" \
      "${EXTRA[@]}" --partition test --location "${LOCATION}" --extract-dir "${EXTRACT}" \
      --frames-dir "${EXTRACT}/${NAME}" --device cuda --parity-clips "${PARITY_CLIPS}" --out "${OUT}/results"
    rm -rf "${EXTRACT:?}/${NAME}"
  done
done

echo "== summary (final numbers are computed again from the results after download)"
.venv/bin/python scripts/summarize_release_evaluation.py --results "${OUT}/results" \
  --out "${OUT}/summary.json" || true
echo "== done: ${OUT}"
ls -la "${OUT}" "${OUT}/results"
