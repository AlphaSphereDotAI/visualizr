FROM cgr.dev/chainguard/wolfi-base:latest@sha256:202dc2241806cbce4472fd287d8e87eb8bd38585d6f8946deeb30672e1c2dc84 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:f3660c56d5b08d6c516360981bedc439f499b9bf37f46a216018da3777a74011 \
     /uv /uvx /usr/bin/

# skipcq: DOK-DL3018
RUN apk add --no-cache build-base

USER nonroot

SHELL ["/bin/ash", "-o", "pipefail", "-c"]

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv tool install visualizr --python "$(grep -E 'requires-python' pyproject.toml | grep -o '[0-9]\+\.[0-9]\+')"

FROM cgr.dev/chainguard/wolfi-base:latest@sha256:202dc2241806cbce4472fd287d8e87eb8bd38585d6f8946deeb30672e1c2dc84 AS production

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    PATH=/home/nonroot/.local/bin:$PATH

# skipcq: DOK-DL3018
RUN apk add --no-cache curl glib mesa-gl

USER nonroot

WORKDIR /home/nonroot

COPY --from=builder --chown=nonroot:nonroot --chmod=555 /home/nonroot/.local/ /home/nonroot/.local/

EXPOSE ${GRADIO_SERVER_PORT}

CMD ["visualizr"]
