#!/usr/bin/env bash
# The usability study's service on this machine, without Docker (docs/usability_study.md):
# the API with the study pages and one worker, running the active release bundle.
#
#   scripts/study_service.sh start    # then open http://127.0.0.1:8010/#/study
#   scripts/study_service.sh status
#   scripts/study_service.sh stop
#
# Its database, uploads and study records live in var/study/ (not committed), apart from any
# other deployment. Uploads are kept for 90 days. Stopping keeps all data.
set -euo pipefail

cd "$(dirname "$0")/.."
DATA="var/study"
PORT="${PASSAGEWATCH_HTTP_PORT:-8010}"
URL="http://127.0.0.1:${PORT}"

export PASSAGEWATCH_DATA_DIR="$PWD/$DATA"
export PASSAGEWATCH_BUNDLE_DIR="$PWD/bundles/active"
export PASSAGEWATCH_FRONTEND_DIR="$PWD/frontend/dist"
export PASSAGEWATCH_STUDY_MODE=true
export PASSAGEWATCH_STUDY_PLAN="$PWD/study/plan.json"
export PASSAGEWATCH_UPLOAD_RETENTION_HOURS=2160
export PASSAGEWATCH_TORCH_THREADS="${PASSAGEWATCH_TORCH_THREADS:-4}"

running() {
  [ -f "$DATA/$1.pid" ] && kill -0 "$(cat "$DATA/$1.pid")" 2>/dev/null
}

start() {
  mkdir -p "$DATA"
  if running api || running worker; then
    echo "already running: $URL/#/study"
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
      echo "ready: $URL/#/study"
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
