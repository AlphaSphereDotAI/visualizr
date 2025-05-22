from os.path import exists
from shutil import copytree

from visualizr.dist_utils import barrier, get_rank


def use_cached_dataset_path(source_path, cache_path):
    if get_rank() == 0:
        if not exists(cache_path):
            # shutil.rmtree(cache_path)
            print(f"copying the data: {source_path} to {cache_path}")
            copytree(source_path, cache_path)
    barrier()
    return cache_path
