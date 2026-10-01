# One image for both service processes; docker/entrypoint.sh selects "api" or "worker".
# PyTorch is the CPU-only build (the serving target is CPU), as locked in uv.lock for Linux.
# Release bundles (weights) are mounted at /bundles at run time and never copied in.

FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Dependencies first (cached unless the lock changes), then the project itself.
COPY pyproject.toml uv.lock README.md ./
# uv's download cache lives in a BuildKit cache mount, not in the image layers.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN useradd --create-home --uid 10001 passagewatch \
    && mkdir -p /data /bundles \
    && chown passagewatch /data
USER passagewatch

ENV PASSAGEWATCH_DATA_DIR=/data \
    PASSAGEWATCH_BUNDLE_DIR=/bundles/active
EXPOSE 8000
ENTRYPOINT ["entrypoint"]
CMD ["api"]
