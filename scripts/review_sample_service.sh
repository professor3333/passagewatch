#!/usr/bin/env bash
# The review sample's service on this machine, without Docker (docs/review_sample.md):
# the API with the review interface and one worker, running the active release bundle.
#
#   scripts/review_sample_service.sh start   # then open the links in review_sample/R1.md
#   scripts/review_sample_service.sh status
#   scripts/review_sample_service.sh stop
#
# Its database and uploads live in var/review-sample/ (not committed), apart from any
# other deployment. Uploads are kept for 90 days. Stopping keeps all data.
set -euo pipefail

cd "$(dirname "$0")/.."
DATA="var/review-sample"
PORT="${PASSAGEWATCH_HTTP_PORT:-8011}"
URL="http://127.0.0.1:${PORT}"

export PASSAGEWATCH_DATA_DIR="$PWD/$DATA"
export PASSAGEWATCH_BUNDLE_DIR="$PWD/bundles/active"
export PASSAGEWATCH_FRONTEND_DIR="$PWD/frontend/dist"
export PASSAGEWATCH_UPLOAD_RETENTION_HOURS=2160
export PASSAGEWATCH_TORCH_THREADS="${PASSAGEWATCH_TORCH_THREADS:-4}"

running() {
  [ -f "$DATA/$1.pid" ] && kill -0 "$(cat "$DATA/$1.pid")" 2>/dev/null
}

start() {
  mkdir -p "$DATA"
  if running api || running worker; then
    echo "already running: $URL/"
    return
  fi
  npm --prefix frontend run build --silent >/dev/null
  nohup .venv/bin/uvicorn passagewatch.service.api.app:app_from_env --factory \
    --host 127.0.0.1 --port "$PORT" >>"$DATA/api.log" 2>&1 &
  echo $! >"$DATA/api.pid"
  nohup .venv/bin/python -m passagewatch.service.worker >>"$DATA/worker.log" 2>&1 &
  echo $! >"$DATA/worker.pid"
  for _ in $(seq 1 60); do
    if curl -sf "$URL/health/ready" >/dev/null; then
      echo "ready: $URL/"
      return
    fi
    sleep 2
  done
  echo "not ready after 120 s; see $DATA/api.log and $DATA/worker.log" >&2
  exit 1
}

stop() {
  for name in api worker; do
    if running "$name"; then
      kill "$(cat "$DATA/$name.pid")"
      echo "stopped $name"
    fi
    rm -f "$DATA/$name.pid"
  done
}

status() {
  for name in api worker; do
    if running "$name"; then echo "$name: running"; else echo "$name: stopped"; fi
  done
  curl -s "$URL/health/ready" && echo
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *)
    echo "usage: $0 start|stop|status" >&2
    exit 2
    ;;
esac
