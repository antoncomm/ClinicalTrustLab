from functools import partial
import torch
from art.attacks.poisoning import PoisoningAttackBackdoor
import numpy as np
from math import floor
from .base_attack import BaseAttack


class PoisonARTAttack(BaseAttack):
    """Poison attack on model."""

    def __init__(
        self,
        model: torch.nn.Module,
        **poison_params
    ) -> None:
        """
        Init pgd attack.

        Parameters
        ----------
        model: torch.nn.Module,
            Model
        """
        super().__init__(model=model)

        self.perturbation = self._make_perturbation
        self.poison_art = PoisoningAttackBackdoor(self.perturbation)

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate pgd adversarial examples using examples and labels.

        Parameters
        ----------
        x: torch.Tensor
            Examples in tool format.
        y: torch.Tensor
            Labels in tool format.

        Return
        ------
        torch.Tensor: adversarial examples in tool format.
        """
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return torch.from_numpy(self.poison_art.generate(x, y)), y

    def _make_perturbation(self, x: torch.Tensor):
        patch_size = 0.1
        patch_size = floor(min(patch_size * x.shape[1], patch_size * x.shape[2]))
        x_pois = x.clone()
        x_pois[:, 0:patch_size, 0:patch_size] = (torch.arange(patch_size**2) % 2).reshape(patch_size, patch_size)
        return x_pois
