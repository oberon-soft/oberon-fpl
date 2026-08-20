FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY fpl/ ./fpl/
RUN uv sync --frozen --no-dev

# One image, three commands. The CronJobs differ only in `args`, which is what
# guarantees the ingest and the model are always the same commit.
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "fpl"]
CMD ["status"]
