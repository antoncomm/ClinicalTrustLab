import torch
from .base_attack import BaseAttack


class UniversalNoiseAdd(BaseAttack):
    """Add universal noise to all images in dataset."""

    def __init__(
        self,
        model: torch.nn.Module,
        mean: tuple, 
        std: tuple, 
        height: int,
        width: int,
        eps: float = 0.05,
        **kwargs
    ) -> None:
        super().__init__(model, mean, std, height, width)
        self.perturbation = torch.empty(self.image_size).uniform_(-eps, eps)

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.device = x.device
        x = self._transform_input(x)
        x_adv = self._transform_output((x + self.perturbation.to(self.device)).clamp(0, 1))
        return x_adv, y
