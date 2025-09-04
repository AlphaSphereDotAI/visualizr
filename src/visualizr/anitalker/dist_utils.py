from torch import Tensor, distributed


def barrier():
    if distributed.is_initialized():
        distributed.barrier()


def broadcast(data, src):
    if distributed.is_initialized():
        distributed.broadcast(data, src)


def all_gather(data: list, src: Tensor):
    if distributed.is_initialized():
        distributed.all_gather(data, src)
    else:
        data[0] = src


def get_rank():
    return distributed.get_rank() if distributed.is_initialized() else 0


def get_world_size():
    return distributed.get_world_size() if distributed.is_initialized() else 1
