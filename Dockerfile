FROM cgr.dev/chainguard/wolfi-base:latest@sha256:0e09bcd548cf2dfb9a3fd40af1a7389aa8c16b428de4e8f72b085f015694ce3d AS builder

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:ca74b4b463d7dfc1176cbe82a02b6e143fd03a144dcb1a87c3c3e81ac16c6f6d \
     /uv /uvx /usr/bin/

# skipcq: DOK-DL3018
RUN apk add --no-cache build-base

USER nonroot

SHELL ["/bin/ash", "-o", "pipefail", "-c"]

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv tool install visualizr --python "$(grep -E 'requires-python' pyproject.toml | grep -o '[0-9]\+\.[0-9]\+')"

FROM cgr.dev/chainguard/wolfi-base:latest@sha256:0e09bcd548cf2dfb9a3fd40af1a7389aa8c16b428de4e8f72b085f015694ce3d AS production

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    PATH=/home/nonroot/.local/bin:$PATH

# skipcq: DOK-DL3018
RUN apk add --no-cache curl glib mesa-gl

USER nonroot

WORKDIR /home/nonroot

COPY --from=builder --chown=nonroot:nonroot --chmod=555 /home/nonroot/.local/ /home/nonroot/.local/

EXPOSE ${GRADIO_SERVER_PORT}

CMD ["visualizr"]
