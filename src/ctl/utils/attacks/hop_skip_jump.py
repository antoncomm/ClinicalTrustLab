from math import sqrt
import torch
from .base_attack import BaseAttack


class HopSkipJumpAttack(BaseAttack):
    """Hop skip jump attack on model."""

    def __init__(
        self,
        model: torch.nn.Module,
        mean: tuple, 
        std: tuple, 
        height: int,
        width: int,
        **kwargs
    ) -> None:
        super().__init__(model, mean, std, height, width)

        self.max_iter = kwargs["max_iter"]
        self.init_trials = kwargs["init_trials"]
        self.grad_samples = kwargs["grad_estimation_samples"]
        self.gamma = kwargs["gamma"]
        self.verbose = kwargs["verbose"]
        self.num_classes = kwargs["num_classes"]

        self.clip_min = 0.0
        self.clip_max = 1.0
        self.bin_search_step = 10
        self.epsilon = 1e-4
        self.num_const = 1e-8
        self.delta_min = 1e-6
        self.delta_factor = 0.1
        self.num_tr = 10

    def _is_adversarial(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        x = x.clamp(self.clip_min, self.clip_max)
        y_adv = self.model(x.unsqueeze(0))
        if self.num_classes > 2:
            y_adv = y_adv.argmax(1)
        return (y != y_adv).item()

    def _init_adversarial(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        best_adv = None
        best_dist = torch.inf

        for _ in range(self.init_trials):
            noise = torch.rand_like(x)
            noise = noise * (self.clip_max - self.clip_min) + self.clip_min

            if self._is_adversarial(noise, y):
                dist = torch.abs(noise - x).max().item()
                if dist < best_dist:
                    best_adv = noise
                    best_dist = dist

        if best_adv is None:
            return x, False

        return best_adv, True

    def _binary_search(self, x_orig: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        low, high = 0.0, 1.0
        for _ in range(self.bin_search_step):
            mid = (low + high) / 2.0
            x_mid = (1 - mid) * x_orig + mid * x_adv
            if self._is_adversarial(x_mid, y):
                high = mid
            else:
                low = mid
        return (1 - high) * x_orig + high * x_adv

    def _estimate_direction(self, x: torch.Tensor, y: torch.Tensor, delta: float) -> torch.Tensor:
        grad = torch.zeros_like(x)

        for _ in range(self.grad_samples):
            noise = torch.randn_like(x)
            noise = noise / torch.abs(noise).max().item()

            x_perturbed = x + delta * noise

            if self._is_adversarial(x_perturbed, y):
                grad += noise
            else:
                grad -= noise
        return grad.sign()

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._transform_input(x)
        x = x.clone().detach()
        x_adv = x.clone().detach()

        for idx, xi in enumerate(x_adv):

            xi_adv, success = self._init_adversarial(xi, y[idx])
            if not success:
                x_adv[idx] = xi_adv
                continue

            xi_adv = self._binary_search(xi, xi_adv, y[idx])

            for i in range(self.max_iter):
                dist = torch.abs(xi_adv - xi).max().item()
                delta = max(self.delta_min, self.delta_factor * dist)

                direction = self._estimate_direction(xi_adv, y[idx], delta)

                step = self.gamma * dist / sqrt(i + 1)
                success = False

                for _ in range(self.num_tr):
                    candidate = xi_adv + step * direction

                    if self._is_adversarial(candidate, y[idx]):
                        xi_adv = self._binary_search(xi, candidate, y[idx])
                        success = True
                        break

                    step *= 0.5

                if not success:
                    break
            x_adv[idx] = xi_adv

        x_adv.data.clamp_(self.clip_min, self.clip_max)
        x_adv = self._transform_output(x_adv.to(self.device))
        return x_adv.detach(), y
