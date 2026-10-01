#!/bin/sh
# Select the service process. Both use the same image, settings and volumes.
set -eu
case "${1:-api}" in
  api)
    exec uvicorn passagewatch.service.api.app:app_from_env --factory \
      --host 0.0.0.0 --port "${PASSAGEWATCH_PORT:-8000}" --no-server-header
    ;;
  worker)
    exec python -m passagewatch.service.worker
    ;;
  *)
    exec "$@"
    ;;
esac
