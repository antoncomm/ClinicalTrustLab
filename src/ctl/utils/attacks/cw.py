import torch
from .base_attack import BaseAttack


class CWAttack(BaseAttack):
    """CW linf attack on model."""

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

        self.num_classes = kwargs["num_classes"]
        self.lossbce = torch.nn.BCEWithLogitsLoss() if self.num_classes == 2 else torch.nn.CrossEntropyLoss()
        self.loss_fn = self.get_loss()

        self.clip_min = 0.0
        self.clip_max = 1.0

        self.c = kwargs["c"]
        self.kappa = kwargs["kappa"]
        self.lr = kwargs["lr"]
        self.max_iter = kwargs["max_iter"]

        self._tanh_const = 1e-7

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.device = x.device
        x = self._transform_input(x)
        imgs_tanh = self._to_tanh_space(x.clone())
        w = imgs_tanh.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([w], lr=self.lr)

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            x_adv = self._from_tanh_space(w)
            x_adv = self._transform_output(x_adv)

            outputs = self.model(x_adv)
            loss = self.loss_fn(outputs, y, x, x_adv)

            loss.backward()
            optimizer.step()

        return x_adv.detach(), y

    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arctanh(x * (2 - self._tanh_const * 2) - 1 + self._tanh_const)

    def _from_tanh_space(self, w: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(w) + 1) / 2

    def get_loss(self) -> callable:

        def loss_fn(output, target, x, x_adv):
            if self.num_classes == 2 and target.ndim == 1:
                target = target.unsqueeze(1)
                target = torch.tensor(target, dtype=torch.float32, device=self.device)
            loss = -self.lossbce(output, target) * self.c
            loss = loss.sum()
            loss += torch.abs(x_adv.to(self.device) - x.to(self.device)).reshape(x.shape[0], -1).max(1).values.sum()
            return loss

        return loss_fn
