FROM cgr.dev/chainguard/wolfi-base:latest@sha256:898e4f30d920607c58acc01eabdfc9ac0725fb83b780d695542b3c3a3d265e48 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:f64ad69940b634e75d2e4d799eb5238066c5eeda49f76e782d4873c3d014ea33 \
     /uv /uvx /usr/bin/

# skipcq: DOK-DL3018
RUN apk add --no-cache build-base

USER nonroot

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv tool install visualizr -p "$(cat .python-version)"

FROM cgr.dev/chainguard/wolfi-base:latest@sha256:898e4f30d920607c58acc01eabdfc9ac0725fb83b780d695542b3c3a3d265e48 AS production

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
