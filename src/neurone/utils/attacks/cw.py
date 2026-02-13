import torch
from .base_attack import BaseAttack


class CWAttack(BaseAttack):
    """CW linf attack on model."""

    def __init__(
        self,
        model: torch.nn.Module,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        c: float = 1.0,
        kappa: float = 0.0,
        lr: float = 1e-3,
        max_iter: int = 200,
    ) -> None:
        """
        Init cw linf attack.

        Parameters
        ----------
        model: torch.nn.Module
            Model
        clip_min: float
            The minimum allowable value of the features after adding perturbations.
        clip_max: float
            The maximum allowable value of the features after adding perturbations.
        c: float
            A weighting coefficient for the model loss in the attack objective.
            Larger values place more emphasis on misclassification.
        kappa: float
            A confidence threshold that increases the margin between the true and target classes,
            used to generate stronger and more targeted attacks.
        lr: float
            The step size by which the variable w in tanh space is updated at each iteration.
        max_iter: int
            The maximum number of iterations of the algorithm.
        """
        super().__init__(model=model)
        self.loss_fn = self.get_loss()

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.c = c
        self.kappa = kappa
        self.lr = lr
        self.max_iter = max_iter

        self._tanh_const = 1e-7

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate cw linf adversarial examples using examples and labels.

        Parameters
        ----------
        x: torch.Tensor
            Examples in tool format.
        y: torch.Tensor|list
            Labels in tool format.

        Return
        ------
        torch.Tensor: adversarial examples in tool format.
        """
        self.device = x.device
        imgs_tanh = self._to_tanh_space(x.clone())
        w = imgs_tanh.clone().detach().requires_grad_(True)

        for _ in range(self.max_iter):
            x_adv = self._from_tanh_space(w)
            outputs = self.model(x_adv)
            loss = self.loss_fn(outputs, y, x, x_adv)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=True, create_graph=True)
            grad *= (1 - torch.square(torch.tanh(w))) / (2 * self._tanh_const)
            w = (w - self.lr * grad.sign()).detach()

        return x_adv.detach(), y

    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert x to tanh space.

        Parameters
        ----------
        x: torch.Tensor
            Example in feature space.

        Return
        ------
        torch.Tensor: example in tanh space.
        """
        return torch.arctanh(x * (2 - self._tanh_const * 2) - 1 + self._tanh_const)

    def _from_tanh_space(self, w: torch.Tensor) -> torch.Tensor:
        """
        Convert w to feature space.

        Parameters
        ----------
        w: torch.Tensor
            Example in tanh space.

        Return
        ------
        torch.Tensor: example in feature space.
        """
        return (torch.tanh(w) + 1) / 2

    def get_loss(self) -> callable:
        """Return loss function for cw attack."""

        def loss_fn(output, target, x, x_adv):
            loss = -torch.nn.CrossEntropyLoss(output, target) * self.c
            loss = loss.sum()
            loss += torch.abs(x_adv.to(self.device) - x.to(self.device)).reshape(x.shape[0], -1).max(1).values.sum()
            return loss

        return loss_fn
