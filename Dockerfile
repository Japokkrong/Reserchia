# Reserchia's web UI, for the compose stack.
#
# Two stages so the runtime image carries no build tooling: the builder resolves
# dependencies with uv, the runtime copies the finished venv.
#
# Runs as uid 1000 to match the host user. The paper library is bind-mounted
# from ~/.local/share/reserchia, and a root container would leave files the host
# user could not rewrite -- which would quietly break `uv run reserchia` on the
# host, since both share one library.

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer is cached until
# the lockfile itself changes -- editing application code does not re-resolve.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --group ui

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group ui


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# --no-create-home: the home directory is a volume mount, not image content.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Chainlit creates .files/ at import time, and WORKDIR leaves /app owned by
# root -- so the non-root user cannot start without this.
RUN mkdir -p /app/.files && chown -R app:app /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app ui/ ./ui/
COPY --chown=app:app public/ ./public/
COPY --chown=app:app .chainlit/ ./.chainlit/
COPY --chown=app:app chainlit.md ./

USER app

EXPOSE 8000

# Chainlit serves its own health surface at /; compose watches this rather than
# guessing at readiness with a sleep.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request as r; r.urlopen('http://127.0.0.1:8000/', timeout=4)"

# --host 0.0.0.0 binds inside the container only. The published port is pinned
# to 127.0.0.1 in compose.yaml, which is what keeps it off the network.
CMD ["chainlit", "run", "ui/app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
