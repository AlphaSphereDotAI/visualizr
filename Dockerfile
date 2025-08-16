FROM python:3.13@sha256:3b2f1b9c9948e9dc96e1a2f4668ba9870ff43ab834f91155697476142b3bc299 AS builder

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

FROM python:3.13-slim@sha256:4d55aff3915a8622fdb8e6ab3645992de771bd97c3dd1279860cd5e18bcd7582 AS production

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
