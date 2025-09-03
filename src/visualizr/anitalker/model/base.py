import numpy as np
import torch


class BaseModule(torch.nn.Module):
    def __init__(self) -> None:
        super(BaseModule, self).__init__()

    @property
    def nparams(self) -> int:
        """Returns the number of trainable parameters of the module."""
        return sum(
            np.prod(param.detach().cpu().numpy().shape)
            for name, param in self.named_parameters()
            if param.requires_grad
        )

    def relocate_input(self, x: list) -> list:
        """Relocates provided tensors to the same device set for the module."""
        try:
            device = next(self.parameters()).device
        except StopIteration as e:
            msg = "Cannot infer device from model parameters: model has no parameters"
            raise ValueError(msg) from e
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x
