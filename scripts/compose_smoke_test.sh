#!/usr/bin/env bash
# Deployment smoke test: build the image, start the Compose stack with a random-weights
# bundle, upload a generated clip, and require the job to succeed end to end. Checks the
# wiring (image, entrypoints, volumes, bundle loading, readiness, job flow), not counting
# quality. Cleans up the stack and its volume on exit.
set -euo pipefail

cd "$(dirname "$0")/.."
WORK="$(mktemp -d)"
chmod 755 "$WORK"  # the container user must be able to read the mounted bundle
export COMPOSE_PROJECT_NAME="passagewatch-smoke"
export PASSAGEWATCH_BUNDLES="$WORK/bundles"
export PASSAGEWATCH_HTTP_PORT="${PASSAGEWATCH_HTTP_PORT:-18000}"
URL="http://127.0.0.1:${PASSAGEWATCH_HTTP_PORT}"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    docker compose logs --no-color --tail 50 || true
  fi
  docker compose down --volumes --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT

echo "== bundle"
uv run python scripts/make_smoke_bundle.py --bundles-dir "$PASSAGEWATCH_BUNDLES"
uv run python - "$WORK/clip.zip" <<'PY'
import io, sys, zipfile
import cv2, numpy as np
with zipfile.ZipFile(sys.argv[1], "w") as zf:
    for i in range(16):
        frame = np.full((120, 80), 40, dtype=np.uint8)
        frame[60:66, 4 + 4 * i : 14 + 4 * i] = 240
        zf.writestr(f"{i}.png", cv2.imencode(".png", frame)[1].tobytes())
PY

echo "== start"
docker compose up -d --build --wait --wait-timeout 300 api worker

echo "== hardening"
for service in api worker; do
  id=$(docker compose ps -q "$service")
  [ "$(docker inspect -f '{{.State.Health.Status}}' "$id")" = "healthy" ]
  [ "$(docker inspect -f '{{.HostConfig.ReadonlyRootfs}}' "$id")" = "true" ]
  docker inspect -f '{{.HostConfig.CapDrop}}' "$id" | grep -q ALL
done
echo "api and worker are healthy, read-only and without capabilities"

echo "== readiness"
for _ in $(seq 1 60); do
  if curl -fsS "$URL/health/ready" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "$URL/health/ready"; echo
INFO=$(curl -fsS "$URL/v1/model-info")
echo "$INFO" | grep -q '"pipeline_version":"smoke-0"'
echo "$INFO" | grep -q '"runtime":"onnxruntime"'
echo "$INFO" | grep -q '"preprocessing_version":"letterbox-temporal3-v1"'
echo "$INFO" | grep -q '"calibration_version":"review-v0"'
curl -fsS "$URL/" | grep -q "<title>PassageWatch Review</title>"

echo "== job"
CLIP=$(curl -fsS -F "file=@$WORK/clip.zip" -F framerate=10 -F x_meter_start=-1 \
  -F x_meter_stop=1 -F y_meter_start=3 -F y_meter_stop=0 "$URL/v1/clips" |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["clip_id"])')
JOB=$(curl -fsS -X POST "$URL/v1/jobs" -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: smoke-1' -d "{\"clip_id\": \"$CLIP\"}" |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["job_id"])')
STATUS=""
for _ in $(seq 1 90); do
  STATUS=$(curl -fsS "$URL/v1/jobs/$JOB" | python3 -c 'import json, sys; print(json.load(sys.stdin)["status"])')
  case "$STATUS" in succeeded|failed) break ;; esac
  sleep 2
done
echo "job $JOB: $STATUS"
[ "$STATUS" = "succeeded" ]
curl -fsS "$URL/v1/jobs/$JOB/results"; echo
curl -fsS "$URL/v1/jobs/$JOB/audit" | grep -q '"calibration_version":"review-v0"'
docker compose logs --no-color worker | grep -q '"message": "job succeeded"'
echo "== smoke test passed"
