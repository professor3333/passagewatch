"""FastAPI application: uploads, jobs, results, tracks, model info, and health.

The API never runs inference. ``POST /v1/jobs`` records a job and returns ``202 Accepted``
at once; a worker process leases and runs it. Each request opens its own SQLite connection
(WAL mode allows concurrent readers alongside the worker's writes).

Status codes: 404 unknown resource; 409 idempotency-key reuse for another request, results
of an unfinished job, or deleting a clip with active jobs; 410 a deleted clip; 413 upload
too large; 422 invalid input; 503 queue full or no release loaded.
"""

# No `from __future__ import annotations`: FastAPI must see the dependency annotations
# defined inside create_app() as objects, not as unresolvable strings.

import shutil
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path, PurePath
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from passagewatch.counting.policy import Direction, DirectionalCounts, to_river_directions
from passagewatch.service import artifacts, media, reviews, study
from passagewatch.service.api.schemas import (
    AuditOut,
    ClipOut,
    DirectionalCountsOut,
    JobAccepted,
    JobIn,
    JobOut,
    ModelInfoOut,
    ObservationsOut,
    ResultsOut,
    ReviewIn,
    ReviewOut,
    ReviewStateOut,
    TracksOut,
)
from passagewatch.service.bundle import ReleaseBundle, load_bundle
from passagewatch.service.catalog import (
    ClipRecord,
    add_clip,
    get_clip,
    get_pipeline_version,
    live_workers,
    mark_clip_deleted,
    register_pipeline_version,
)
from passagewatch.service.db import connect
from passagewatch.service.export import automatic_tracks, build_report, to_csv
from passagewatch.service.jobs import (
    QUEUED,
    RUNNING,
    SUCCEEDED,
    IdempotencyConflictError,
    Job,
    JobStore,
    QueueFullError,
    iso,
    new_id,
    utc_now,
)
from passagewatch.service.logs import configure_logging
from passagewatch.service.metrics import ServiceCollector
from passagewatch.service.settings import ServiceSettings

API_VERSION = "v1"
MAX_IDEMPOTENCY_KEY = 200
MAX_OBSERVATION_FRAMES = 500


class _UploadSizeLimit(BaseHTTPMiddleware):
    """Reject oversized uploads from their Content-Length, before reading the body."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        length = request.headers.get("content-length")
        if request.url.path == "/v1/clips" and length and int(length) > self.max_bytes + 64 * 1024:
            return JSONResponse(
                {"detail": f"upload exceeds the {self.max_bytes}-byte limit"}, status_code=413
            )
        return await call_next(request)


def create_app(settings: ServiceSettings) -> FastAPI:
    state: dict[str, Any] = {"bundle": None, "bundle_error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.media_dir.mkdir(parents=True, exist_ok=True)
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        conn = connect(settings.db_path)
        try:
            bundle = load_bundle(settings.bundle_dir)
            register_pipeline_version(conn, bundle.pipeline_version, bundle.config(), now=utc_now())
            state["bundle"] = bundle
        except (OSError, ValueError) as exc:
            state["bundle_error"] = str(exc)
        finally:
            conn.close()
        yield

    app = FastAPI(title="PassageWatch", version=API_VERSION, lifespan=lifespan)
    app.add_middleware(_UploadSizeLimit, max_bytes=settings.max_upload_bytes)

    def db() -> Iterator[sqlite3.Connection]:
        conn = connect(settings.db_path)
        try:
            yield conn
        finally:
            conn.close()

    Db = Annotated[sqlite3.Connection, Depends(db)]

    def store(conn: sqlite3.Connection) -> JobStore:
        return JobStore(
            conn,
            max_queue=settings.max_queue,
            max_attempts=settings.max_attempts,
            deadline_seconds=settings.job_deadline_seconds,
        )

    def active_bundle() -> ReleaseBundle:
        bundle: ReleaseBundle | None = state["bundle"]
        if bundle is None:
            raise HTTPException(503, f"no release bundle is loaded: {state['bundle_error']}")
        return bundle

    def require_job(conn: sqlite3.Connection, job_id: str) -> Job:
        job = store(conn).get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return job

    def require_succeeded(conn: sqlite3.Connection, job_id: str) -> Job:
        job = require_job(conn, job_id)
        if job.status != SUCCEEDED:
            raise HTTPException(409, f"job {job_id} is {job.status}; results are not available")
        return job

    # -- clips ---------------------------------------------------------------------

    def clip_out(clip: ClipRecord) -> ClipOut:
        return ClipOut(
            clip_id=clip.clip_id,
            sha256=clip.sha256,
            media_kind=clip.media_kind,  # type: ignore[arg-type]
            num_frames=clip.num_frames,
            width=clip.width,
            height=clip.height,
            framerate=clip.framerate,
            duration_seconds=clip.duration_seconds,
            meters={
                "x_start": clip.x_meter_start,
                "x_stop": clip.x_meter_stop,
                "y_start": clip.y_meter_start,
                "y_stop": clip.y_meter_stop,
            },
            created_at=clip.created_at,
            expires_at=clip.expires_at,
        )

    @app.post("/v1/clips", status_code=201, response_model=ClipOut)
    def upload_clip(
        conn: Db,
        response: Response,
        file: Annotated[UploadFile, File(description="ZIP of frames 0..N-1, or a video")],
        x_meter_start: Annotated[float, Form()],
        x_meter_stop: Annotated[float, Form()],
        y_meter_start: Annotated[float, Form()],
        y_meter_stop: Annotated[float, Form()],
        framerate: Annotated[float | None, Form(gt=0)] = None,
    ) -> ClipOut:
        if x_meter_start == x_meter_stop or y_meter_start == y_meter_stop:
            raise HTTPException(422, "the sonar window must have a non-zero extent in meters")
        clip_id = new_id("clip")
        clip_dir = settings.media_dir / clip_id
        suffix = PurePath(file.filename or "").suffix.lower()
        suffix = suffix if suffix.isascii() and suffix[1:].isalnum() and len(suffix) <= 6 else ""
        try:
            saved = media.save_upload(
                file.file,
                clip_dir / f"upload{suffix or '.bin'}",
                max_bytes=settings.max_upload_bytes,
            )
            info = media.probe(saved.path, max_frames=settings.max_frames)
            rate = framerate if framerate is not None else info.framerate
            if rate is None:
                raise media.UploadRejectedError("framerate is required for a ZIP of frames")
            now = utc_now()
            clip = add_clip(
                conn,
                clip_id=clip_id,
                source="upload",
                sha256=saved.sha256,
                media_kind=info.kind,
                media_path=str(saved.path.relative_to(settings.media_dir)),
                num_frames=info.num_frames,
                width=info.width,
                height=info.height,
                framerate=rate,
                meters=(x_meter_start, x_meter_stop, y_meter_start, y_meter_stop),
                now=now,
                expires_at=now + timedelta(hours=settings.upload_retention_hours),
            )
        except media.UploadTooLargeError as exc:
            shutil.rmtree(clip_dir, ignore_errors=True)
            raise HTTPException(413, str(exc)) from None
        except media.UploadRejectedError as exc:
            shutil.rmtree(clip_dir, ignore_errors=True)
            raise HTTPException(422, str(exc)) from None
        except BaseException:
            shutil.rmtree(clip_dir, ignore_errors=True)
            raise
        response.headers["Location"] = f"/v1/clips/{clip_id}"
        return clip_out(clip)

    @app.delete("/v1/clips/{clip_id}", status_code=204)
    def delete_clip(clip_id: str, conn: Db) -> Response:
        clip = get_clip(conn, clip_id)
        if clip is None:
            raise HTTPException(404, f"unknown clip {clip_id}")
        active = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE clip_id = ? AND status IN (?, ?)",
            (clip_id, QUEUED, RUNNING),
        ).fetchone()[0]
        if active:
            raise HTTPException(409, f"clip {clip_id} has {active} queued or running jobs")
        shutil.rmtree(settings.media_dir / clip_id, ignore_errors=True)
        mark_clip_deleted(conn, clip_id, now=utc_now())
        return Response(status_code=204)

    def require_clip_media(conn: sqlite3.Connection, clip_id: str) -> ClipRecord:
        clip = get_clip(conn, clip_id)
        if clip is None:
            raise HTTPException(404, f"unknown clip {clip_id}")
        if clip.deleted_at is not None:
            raise HTTPException(410, f"clip {clip_id} was deleted")
        return clip

    @app.get("/v1/clips/{clip_id}", response_model=ClipOut)
    def get_clip_info(clip_id: str, conn: Db) -> ClipOut:
        return clip_out(require_clip_media(conn, clip_id))

    @app.get("/v1/clips/{clip_id}/frames/{index}")
    def get_frame(clip_id: str, index: int, conn: Db) -> Response:
        clip = require_clip_media(conn, clip_id)
        if not 0 <= index < clip.num_frames:
            raise HTTPException(404, f"frame {index} is outside [0, {clip.num_frames})")
        try:
            data, media_type = media.read_frame(
                settings.media_dir / clip.media_path, clip.media_kind, index
            )
        except IndexError as exc:
            raise HTTPException(404, str(exc)) from None
        # A recording never changes after upload, so its frames can be cached.
        return Response(
            data, media_type=media_type, headers={"Cache-Control": "private, max-age=86400"}
        )

    # -- jobs ----------------------------------------------------------------------

    def job_out(job: Job) -> JobOut:
        return JobOut(
            job_id=job.job_id,
            clip_id=job.clip_id,
            status=job.status,
            progress=job.progress,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            error=job.error,
            pipeline_version=job.pipeline_version,
            counting=job.counting,
            cached_from=job.cached_from,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            results_url=f"/v1/jobs/{job.job_id}/results" if job.status == SUCCEEDED else None,
        )

    @app.post("/v1/jobs", status_code=202, response_model=JobAccepted)
    def create_job(
        request: JobIn,
        conn: Db,
        response: Response,
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=MAX_IDEMPOTENCY_KEY)
        ] = None,
    ) -> JobAccepted:
        bundle = active_bundle()
        clip = get_clip(conn, request.clip_id)
        if clip is None:
            raise HTTPException(404, f"unknown clip {request.clip_id}")
        if clip.deleted_at is not None:
            raise HTTPException(410, f"clip {request.clip_id} was deleted")
        try:
            created = store(conn).create(
                clip_id=clip.clip_id,
                clip_sha256=clip.sha256,
                pipeline_version=bundle.pipeline_version,
                pipeline_config_sha256=bundle.config_sha256(),
                counting=request.counting.model_dump(mode="json"),
                idempotency_key=idempotency_key,
                now=utc_now(),
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(409, str(exc)) from None
        except QueueFullError as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "30"}) from None
        status_url = f"/v1/jobs/{created.job.job_id}"
        response.headers["Location"] = status_url
        return JobAccepted(
            job_id=created.job.job_id,
            status=created.job.status,
            pipeline_version=created.job.pipeline_version,
            status_url=status_url,
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str, conn: Db) -> JobOut:
        return job_out(require_job(conn, job_id))

    def counts_out(right: int, left: int, upstream_direction: str | None) -> DirectionalCountsOut:
        orientation = None if upstream_direction is None else Direction(upstream_direction)
        river = to_river_directions(DirectionalCounts(right=right, left=left), orientation)
        return DirectionalCountsOut(
            right=right,
            left=left,
            upstream=None if river is None else river.upstream,
            downstream=None if river is None else river.downstream,
            net_upstream=None if river is None else river.net,
        )

    @app.get("/v1/jobs/{job_id}/results", response_model=ResultsOut)
    def get_results(job_id: str, conn: Db) -> ResultsOut:
        job = require_succeeded(conn, job_id)
        jobs = store(conn)
        automatic = jobs.result(job_id, revision=0)
        latest = jobs.result(job_id)
        clip = get_clip(conn, job.clip_id)
        pipeline = get_pipeline_version(conn, job.pipeline_version)
        assert automatic is not None and latest is not None and clip is not None
        assert pipeline is not None
        direction = job.counting.get("upstream_direction")
        reviewed = None
        if latest.revision > 0:
            reviewed = counts_out(latest.counts["right"], latest.counts["left"], direction)
        return ResultsOut(
            job_id=job_id,
            clip_id=job.clip_id,
            recording_sha256=clip.sha256,
            status=job.status,
            cached=job.cached_from is not None,
            counting=job.counting,
            pipeline_version=job.pipeline_version,
            pipeline_config_sha256=pipeline.config_sha256,
            automatic=counts_out(automatic.counts["right"], automatic.counts["left"], direction),
            reviewed=reviewed,
            review=ReviewStateOut(
                revision=latest.revision,
                kind=latest.kind,  # type: ignore[arg-type]
                state="reviewed" if latest.revision > 0 else "pending",
            ),
            tracks=automatic.counts.get("tracks", 0),
            tracks_url=f"/v1/jobs/{job_id}/tracks",
            created_at=automatic.created_at,
        )

    @app.get("/v1/jobs/{job_id}/tracks", response_model=TracksOut)
    def get_tracks(
        job_id: str,
        conn: Db,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1)] = 50,
        order: Annotated[str, Query(pattern="^(track|queue)$")] = "track",
    ) -> TracksOut:
        """Tracks by ID, or in review-queue order (``order=queue``): unresolved, then needs
        review, then suggested; passages before other tracks; lowest review score first."""
        if limit > settings.tracks_page_limit:
            raise HTTPException(422, f"limit must be at most {settings.tracks_page_limit}")
        require_succeeded(conn, job_id)
        result = store(conn).result(job_id)
        assert result is not None
        job_dir = settings.artifacts_dir / result.tracks_artifact
        if order == "queue":
            ranked = sorted(artifacts.read_all_tracks(job_dir), key=reviews.queue_key)
            page = artifacts.TrackPage(len(ranked), ranked[offset : offset + limit])
        else:
            page = artifacts.read_tracks(job_dir, offset=offset, limit=limit)
        tracks = []
        for track in page.tracks:
            state, final = reviews.track_outcome(
                reviews.AutomaticTrack(track["track_id"], track["direction"]), result.decisions
            )
            tracks.append(track | {"review_state": state, "final_direction": final})
        return TracksOut(
            job_id=job_id,
            revision=result.revision,
            total=page.total,
            offset=offset,
            limit=limit,
            tracks=tracks,
        )

    @app.get("/v1/jobs/{job_id}/observations", response_model=ObservationsOut)
    def get_observations(
        job_id: str,
        conn: Db,
        start: Annotated[int, Query(ge=0)],
        stop: Annotated[int, Query(ge=1)],
    ) -> ObservationsOut:
        """Every tracked box in frames ``[start, stop)``, for overlays."""
        if not 0 < stop - start <= MAX_OBSERVATION_FRAMES:
            raise HTTPException(422, f"stop - start must be in [1, {MAX_OBSERVATION_FRAMES}]")
        require_succeeded(conn, job_id)
        result = store(conn).result(job_id)
        assert result is not None
        boxes = artifacts.read_observations(
            settings.artifacts_dir / result.tracks_artifact, start=start, stop=stop
        )
        return ObservationsOut(job_id=job_id, start=start, stop=stop, boxes=boxes)

    # -- review and export -----------------------------------------------------------

    @app.post("/v1/jobs/{job_id}/reviews", status_code=201, response_model=ReviewOut)
    def post_review(job_id: str, review: ReviewIn, conn: Db) -> ReviewOut:
        job = require_succeeded(conn, job_id)
        clip = get_clip(conn, job.clip_id)
        result = store(conn).result(job_id, 0)
        assert clip is not None and result is not None
        job_dir = settings.artifacts_dir / result.tracks_artifact
        rows = artifacts.read_all_tracks(job_dir)
        audit = artifacts.read_audit(job_dir)
        try:
            revision = reviews.submit_review(
                conn,
                job_id,
                reviews.ReviewEvent(**review.model_dump()),
                automatic=automatic_tracks(rows),
                num_frames=clip.num_frames,
                now=utc_now(),
                audit_windows=0 if audit is None else len(audit["windows"]),
            )
        except reviews.StaleRevisionError as exc:
            raise HTTPException(409, str(exc)) from None
        except reviews.InvalidReviewError as exc:
            raise HTTPException(422, str(exc)) from None
        latest = store(conn).result(job_id, revision)
        assert latest is not None
        counts = latest.counts
        return ReviewOut(
            job_id=job_id,
            revision=revision,
            reviewed=counts_out(
                counts["right"], counts["left"], job.counting.get("upstream_direction")
            ),
            unresolved=counts["unresolved"],
            added_passages=counts["added_passages"],
            results_url=f"/v1/jobs/{job_id}/results",
        )

    @app.get("/v1/jobs/{job_id}/audit", response_model=AuditOut)
    def get_audit(job_id: str, conn: Db) -> AuditOut:
        """Random windows of unflagged footage to watch for fish the system missed, and
        which of them a reviewer has checked. Empty without a calibration version."""
        require_succeeded(conn, job_id)
        result = store(conn).result(job_id)
        assert result is not None
        view = reviews.audit_view(
            artifacts.read_audit(settings.artifacts_dir / result.tracks_artifact),
            result.decisions,
        )
        if view is None:
            return AuditOut(job_id=job_id, revision=result.revision, windows=[])
        return AuditOut(
            job_id=job_id,
            revision=result.revision,
            calibration_version=view["calibration_version"],
            unflagged_frames=view["unflagged_frames"],
            windows=view["windows"],
        )

    @app.get("/v1/jobs/{job_id}/reviews")
    def get_reviews(job_id: str, conn: Db) -> dict[str, Any]:
        require_job(conn, job_id)
        return {"job_id": job_id, "events": reviews.review_history(conn, job_id)}

    @app.get("/v1/jobs/{job_id}/export")
    def export(
        job_id: str,
        conn: Db,
        format: Annotated[str, Query(pattern="^(json|csv)$")] = "json",
        revision: Annotated[int | None, Query(ge=0)] = None,
    ) -> Response:
        require_succeeded(conn, job_id)
        try:
            report = build_report(conn, settings.artifacts_dir, job_id, revision)
        except KeyError:
            raise HTTPException(404, f"job {job_id} has no revision {revision}") from None
        name = f"passagewatch_{job_id}_r{report['revision']}"
        if format == "csv":
            return PlainTextResponse(
                to_csv(report),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
            )
        return JSONResponse(
            report, headers={"Content-Disposition": f'attachment; filename="{name}.json"'}
        )

    # -- release and health ----------------------------------------------------------

    registry = CollectorRegistry()
    registry.register(
        ServiceCollector(
            connect=lambda: connect(settings.db_path),
            bundle=lambda: state["bundle"],
            now=utc_now,
            stale_seconds=settings.worker_stale_seconds,
        )
    )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus metrics (docs/operations.md)."""
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model-info", response_model=ModelInfoOut)
    def model_info() -> ModelInfoOut:
        bundle = active_bundle()
        return ModelInfoOut(
            pipeline_version=bundle.pipeline_version,
            pipeline_config_sha256=bundle.config_sha256(),
            detector=bundle.detector.model_dump(mode="json"),
            preprocessing_version=bundle.preprocessing_version,
            tracker=bundle.tracker.model_dump(mode="json"),
            counting_policy=bundle.counting_policy,
            calibration_version=bundle.calibration_version,
            provenance=bundle.provenance,
        )

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(conn: Db) -> JSONResponse:
        bundle: ReleaseBundle | None = state["bundle"]
        checks = {
            "database": _check(lambda: conn.execute("SELECT 1").fetchone()),
            "storage": _check(lambda: _writable(settings.media_dir)),
            "release": bundle is not None,
            # A worker with the active version has loaded its model and is alive.
            "worker": bundle is not None
            and bool(
                live_workers(
                    conn,
                    bundle.pipeline_version,
                    now=utc_now(),
                    stale_seconds=settings.worker_stale_seconds,
                )
            ),
        }
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ready" if ok else "not ready", "checks": checks},
            status_code=200 if ok else 503,
        )

    # The review interface, mounted last so it never shadows an API route.
    if settings.study_mode:
        _add_study_routes(app, settings, store)

    if (settings.frontend_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="ui")

    return app


def _check(probe: Any) -> bool:
    try:
        probe()
    except Exception:
        return False
    return True


def _writable(directory: Path) -> None:
    marker = directory / ".write-test"
    marker.write_text("ok")
    marker.unlink()


def app_from_env() -> FastAPI:
    """Entry point for ``uvicorn passagewatch.service.api.app:app_from_env --factory``."""
    configure_logging()
    return create_app(ServiceSettings.from_env())


def _add_study_routes(
    app: FastAPI, settings: ServiceSettings, store: Callable[[sqlite3.Connection], JobStore]
) -> None:
    """Usability-study endpoints (docs/usability_study.md), only in study mode.

    The plan is read on each request, so the service can run while
    ``scripts/prepare_usability_study.py`` uploads the clips and writes it (503 until then).
    """
    plan_path = settings.study_plan
    if plan_path is None:
        raise ValueError("study mode needs PASSAGEWATCH_STUDY_PLAN")
    study_db = settings.data_dir / "study.db"

    def current_plan() -> study.StudyPlan:
        if not plan_path.is_file():
            raise HTTPException(503, "the study plan has not been prepared yet")
        return study.load_plan(plan_path)

    Plan = Annotated[study.StudyPlan, Depends(current_plan)]

    def study_conn() -> Iterator[sqlite3.Connection]:
        conn = study.connect_study(study_db)
        try:
            yield conn
        finally:
            conn.close()

    def service_conn() -> Iterator[sqlite3.Connection]:
        conn = connect(settings.db_path)
        try:
            yield conn
        finally:
            conn.close()

    StudyDb = Annotated[sqlite3.Connection, Depends(study_conn)]
    ServiceDb = Annotated[sqlite3.Connection, Depends(service_conn)]

    @app.get("/v1/study/plan")
    def study_plan(plan: Plan) -> dict[str, Any]:
        return study.public_plan(plan)

    @app.get("/v1/study/progress/{participant}")
    def study_progress(participant: str, conn: StudyDb) -> dict[str, Any]:
        return study.progress(conn, participant)

    @app.post("/v1/study/trials", status_code=201)
    def study_trial(
        trial: study.TrialIn, plan: Plan, conn: StudyDb, service: ServiceDb
    ) -> dict[str, Any]:
        try:
            clip, job_id = study.check_trial(plan, trial)
        except study.StudyError as exc:
            raise HTTPException(422, str(exc)) from None
        if job_id is not None:
            result = store(service).result(job_id)
            if result is None:
                raise HTTPException(409, f"job {job_id} has no result")
            final = (int(result.counts["right"]), int(result.counts["left"]))
            revision: int | None = result.revision
        else:
            assert trial.right is not None and trial.left is not None
            final, revision = (trial.right, trial.left), None
        try:
            study.record_trial(conn, trial, clip, final, job_id, revision)
        except study.StudyError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"clip": trial.clip, "final": list(final), "revision": revision}

    @app.post("/v1/study/questionnaires", status_code=201)
    def study_questionnaire(q: study.QuestionnaireIn, plan: Plan, conn: StudyDb) -> dict[str, str]:
        if q.participant not in plan.participants:
            raise HTTPException(422, f"unknown participant {q.participant}")
        study.record_questionnaire(conn, q, iso(utc_now()))
        return {"status": "recorded"}

    @app.get("/v1/study/export/{table}.csv")
    def study_export(table: str, conn: StudyDb) -> PlainTextResponse:
        if table not in ("trials", "questionnaires"):
            raise HTTPException(404, f"no study table {table}")
        return PlainTextResponse(study.export_csv(conn, table), media_type="text/csv")  # type: ignore[arg-type]
