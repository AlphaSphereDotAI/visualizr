from huggingface_hub import snapshot_download

from visualizr.settings import Settings


class Model:
    def __init__(self, settings: Settings):
        snapshot_download(
            repo_id=settings.model.repo_id,
            local_dir=settings.directory.checkpoint,
            repo_type="model",
            rev="main",
        )
