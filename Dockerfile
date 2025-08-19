FROM cgr.dev/chainguard/wolfi-base:latest@sha256:d7032b02de971d3ef61b8e90bbd7c9e16e7889db9391f38ceb5482089dd35c2e AS builder

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:b05b3d61eb2b264ed785265b71155738a0d3d382ea0699e048d4b36f90b88788 \
     /uv /uvx /usr/bin/

# skipcq: DOK-DL3018
RUN apk add --no-cache build-base

USER nonroot

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv tool install visualizr -p "$(cat .python-version)"

FROM cgr.dev/chainguard/wolfi-base:latest@sha256:d7032b02de971d3ef61b8e90bbd7c9e16e7889db9391f38ceb5482089dd35c2e AS production

ENV GRADIO_SERVER_PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PATH=/home/nonroot/.local/bin:$PATH

# skipcq: DOK-DL3018
RUN apk add --no-cache curl glib mesa-gl

USER nonroot

WORKDIR /home/nonroot

COPY --from=builder --chown=nonroot:nonroot --chmod=555 /home/nonroot/.local/ /home/nonroot/.local/

EXPOSE ${GRADIO_SERVER_PORT}

CMD ["visualizr"]
