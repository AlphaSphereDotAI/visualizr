import os
import shutil

from dist_utils import barrier, get_rank

from visualizr import logger


def use_cached_dataset_path(source_path, cache_path):
    if (get_rank() == 0) and (not os.path.exists(cache_path)):
        logger.info(f"copying the data: {source_path} to {cache_path}")
        shutil.copytree(source_path, cache_path)
    barrier()
    return cache_path
