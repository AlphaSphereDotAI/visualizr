FROM ghcr.io/prefix-dev/pixi:jammy-cuda-12.6.3

SHELL ["/bin/bash", "-c"]

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    UV_NO_CACHE=true \
    PATH="/root/.pixi/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1-mesa-glx libsndfile1 x264 && \
    apt-get full-upgrade -y && \
    apt-get autoremove && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

ADD https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar ./checkpoints/mapping_00109-model.pth.tar
ADD https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar ./checkpoints/mapping_00229-model.pth.tar
ADD https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors ./checkpoints/SadTalker_V0.0.2_256.safetensors
ADD https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors ./checkpoints/SadTalker_V0.0.2_512.safetensors
ADD https://huggingface.co/vinthony/SadTalker-V002rc/resolve/main/epoch_00190_iteration_000400000_checkpoint.pt?download=true ./checkpoints/epoch_00190_iteration_000400000_checkpoint.pt
ADD https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth ./gfpgan/weights/alignment_WFLW_4HG.pth
ADD https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth ./gfpgan/weights/detection_Resnet50_Final.pth
ADD https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth ./gfpgan/weights/GFPGANv1.4.pth
ADD https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth ./gfpgan/weights/parsing_parsenet.pth

COPY pyproject.toml .

RUN pixi global install uv && \
    uv python install 3.9 && \
    uv lock --upgrade && \
    uv sync

COPY . .

EXPOSE 8002

CMD ["uv", "run", "fastapi", "dev", "--host", "0.0.0.0", "--port", "8002"]
