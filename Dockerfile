FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_FROZEN=1 \
    PATH="/root/.local/bin:$PATH" \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0

RUN groupadd visualizr && \
    useradd -g visualizr -s /bin/bash -d /app visualizr && \
    uv tool install --quiet huggingface-hub[cli] && \
    huggingface-cli download taocode/anitalker_ckpts --quiet --local-dir /app/ckpts && \
    uv tool uninstall --quiet huggingface-hub

WORKDIR /app

RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=src,target=/app/src \
    uv export --no-hashes --no-editable --no-dev --quiet -o pylock.toml && \
    uv pip sync pylock.toml

RUN chown -R visualizr:visualizr /app

COPY --chown=visualizr:visualizr . /app

USER visualizr

EXPOSE ${GRADIO_SERVER_PORT}

CMD ["python", "src/visualizr"]