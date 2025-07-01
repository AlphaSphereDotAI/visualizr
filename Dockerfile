FROM python:3.10@sha256:33f72df2ad8c9f777bf0adb35b9d89c5d62935cee2af1f9c3224fb6f7da1dc6b AS builder

SHELL ["/bin/bash", "-c"]

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:6c1e19020ec221986a210027040044a5df8de762eb36d5240e382bc41d7a9043 /uv /uvx /bin/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --no-install-project --no-dev --locked --no-editable

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --locked --no-editable

FROM python:3.10-slim@sha256:9dd6774a1276178f94b0cc1fb1f0edd980825d0ea7634847af9940b1b6273c13 AS production

SHELL ["/bin/bash", "-c"]

ENV GRADIO_SERVER_PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0

# skipcq: DOK-DL3008
RUN groupadd app && \
    useradd -m -g app -s /bin/bash app && \
    apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends espeak-ng ffmpeg && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

USER app

EXPOSE ${GRADIO_SERVER_PORT}

CMD ["/app/.venv/bin/vocalizr"]
