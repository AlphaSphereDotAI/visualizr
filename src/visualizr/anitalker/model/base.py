import torch


class BaseModule(torch.nn.Module):
    def __init__(self) -> None:
        super(BaseModule, self).__init__()
