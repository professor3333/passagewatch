"""Container health check for the worker: ``python -m passagewatch.service.worker.health``.

Exits 0 if a worker process on this host (worker IDs are ``<hostname>-<pid>``) has
heartbeated within ``worker_stale_seconds``, which it does while idle and during jobs once
its model has loaded; otherwise exits 1. A worker that hangs or dies is then reported
unhealthy, and ``/health/ready`` stops counting it.
"""

from __future__ import annotations

import socket
import sys
from datetime import datetime, timedelta

from passagewatch.service.db import connect
from passagewatch.service.jobs import iso, utc_now
from passagewatch.service.settings import ServiceSettings


def is_healthy(settings: ServiceSettings, hostname: str, now: datetime) -> bool:
    if not settings.db_path.exists():
        return False
    conn = connect(settings.db_path)
    try:
        cutoff = iso(now - timedelta(seconds=settings.worker_stale_seconds))
        row = conn.execute(
            "SELECT COUNT(*) FROM workers WHERE worker_id LIKE ? AND heartbeat_at >= ?",
            (f"{hostname}-%", cutoff),
        ).fetchone()
        return bool(row[0])
    finally:
        conn.close()


def main() -> int:
    settings = ServiceSettings.from_env()
    return 0 if is_healthy(settings, socket.gethostname(), utc_now()) else 1


if __name__ == "__main__":
    sys.exit(main())
