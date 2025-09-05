from torch.nn import Module


class BaseModule(Module):
    def __init__(self) -> None:
        super().__init__()
