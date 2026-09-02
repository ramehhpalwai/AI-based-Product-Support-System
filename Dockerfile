FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY trained_models ./trained_models
RUN mkdir -p data/artifacts

EXPOSE 8000
CMD ["uv", "run", "--frozen", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
