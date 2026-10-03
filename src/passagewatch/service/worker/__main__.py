"""Worker entry point: ``python -m passagewatch.service.worker``.

Loads the active release bundle once (refusing a mismatched one), then leases and runs jobs
until SIGTERM or SIGINT. A job interrupted by shutdown is recovered by the next worker once
its lease expires.
"""

from __future__ import annotations

import signal
import sys
import threading

import torch

from passagewatch.service.logs import configure_logging
from passagewatch.service.settings import ServiceSettings
from passagewatch.service.worker.loop import default_worker_id, run_worker
from passagewatch.service.worker.pipeline import InferencePipeline


def main() -> int:
    configure_logging()
    settings = ServiceSettings.from_env()
    if settings.torch_threads is not None:
        torch.set_num_threads(settings.torch_threads)
    pipeline = InferencePipeline.load(
        settings.bundle_dir.resolve(),
        settings.device,
        settings.inference_batch,
        threads=settings.torch_threads,
    )
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    run_worker(settings, pipeline, default_worker_id(), stop=stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
