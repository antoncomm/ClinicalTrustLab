from math import sqrt
import torch
from .base_attack import BaseAttack


#rewrite get grad and predict for torch model

class HopSkipJumpAttack(BaseAttack):
    """Hop skip jump attack on model."""

    def __init__(
        self,
        model: torch.nn.Module,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        max_iter: int = 20,
        init_trials: int = 100,
        grad_estimation_samples: int = 20,
        gamma: float = 1.0,
        verbose: bool = True,
    ) -> None:
        """
        Execute hop skip jump attack.

        Parameters
        ----------
        model: torch.nn.Module,
            Model
        clip_min: float
            The minimum allowable value of the features after adding perturbations.
        clip_max: float
            The maximum allowable value of the features after adding perturbations.
        max_iter: int
            The maximum number of iterations for searching the minimal perturbation for each example.
        init_trials: int
            The number of random trials used to find an initial adversarial example.
        grad_estimation_samples: int
            The number of random directions used to estimate the gradient during perturbation optimization.
        gamma: int
            A multiplier used to compute the step size for updating the example at each iteration in
            the direction of the estimated gradient.
        examples_number: int | float
            The number of batch on which the attack is performed and the model
            robustness is evaluated (absolute or relative).
        verbose: bool
            A flag indicating whether to output detailed information about the attack process.
        """
        super().__init__(model=model)

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.max_iter = max_iter
        self.init_trials = init_trials
        self.grad_samples = grad_estimation_samples
        self.gamma = gamma
        self.verbose = verbose

        self.bin_search_step = 10
        self.epsilon = 1e-4
        self.num_const = 1e-8
        self.delta_min = 1e-6
        self.delta_factor = 0.1
        self.num_tr = 10

    def _is_adversarial(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """
        Check if example is adversarial.

        Parameters
        ----------
        x: torch.Tensor
            Example for check.
        y: torch.Tensor
            True labels.

        Return
        ------
        bool: true-adversarial, false-not.
        """
        x = x.clamp(self.clip_min, self.clip_max)
        y_adv = self.model(x)
        return (y != y_adv).item()

    def _init_adversarial(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        Init adversarial examples.

        Parameters
        ----------
        x: torch.Tensor
            True example.
        y: torch.Tensor
            True label.

        Return
        ------
        tuple: (adversarial_example, bool value true=adversarial)
        """
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
        """
        Find best adversarial example using binary search.

        Parameters
        ----------
        x_orig: torch.Tensor
            True example.
        x_adv: torch.Tensor
            Begin adversarial examples.
        y: torch.Tensor
            True labels.

        Return
        ------
        torch.Tensor: new adversarial example.
        """
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
        """
        Function that estimate gradient.

        Parameters
        ----------
        x: torch.Tensor
            Example.
        y: torch.Tensor
            True label.
        delta: float
            Noise value.

        Return
        ------
        torch.Tensor: gradient estimation.
        """
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
        """
        Generate hop skip jump adversarial examples using examples and labels.

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
        x = x.clone().detach()
        x_adv = x.clone().detach()
        y = self._set_targets(y)

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
        return x_adv.detach(), y

    def _set_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Uses for unsqueeze labels for all kind of tasks.

        Parameters
        ----------
        y: torch.Tensor
            Labels.

        Return
        ------
        torch.Tensor: Unsqueezed labels.
        """
        return y.unsqueeze(1)
