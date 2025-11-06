FROM cgr.dev/chainguard/wolfi-base:latest@sha256:9b359e55a1d5dcdcab3c0534a5184386ebaa2661f8e24bc27355bc6bdcd7e843 AS builder

ARG INSTALL_SOURCE
ARG PYTHON_VERSION

# skipcq: DOK-DL3018
RUN apk add --no-cache build-base git uv

USER nonroot

RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install "${INSTALL_SOURCE}" --python "${PYTHON_VERSION}"

FROM cgr.dev/chainguard/wolfi-base:latest@sha256:9b359e55a1d5dcdcab3c0534a5184386ebaa2661f8e24bc27355bc6bdcd7e843 AS production

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
