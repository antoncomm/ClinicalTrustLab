import torch
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators import BaseEstimator, LossGradientsMixin
import numpy as np
from .base_attack import BaseAttack


class PGDEstimator(BaseEstimator, LossGradientsMixin):
    def __init__(
            self,
            model: torch.nn.Module,
            convertion: callable, 
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_max=1.0,
            clip_min=0.0,
            preprocessing=(0, 1)
    ):
        self.torch_model = model
        self.convertion = convertion

        self._input_shape = input_shape
        self._nb_classes = nb_classes

        self.clip_max = clip_max
        self.clip_min = clip_min
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._all_framework_preprocessing = preprocessing
        self.loss_computer = torch.nn.BCEWithLogitsLoss() if nb_classes == 2 else torch.nn.CrossEntropyLoss()

    @property
    def device(self):
        return self._device

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def nb_classes(self):
        return self._nb_classes

    @property
    def clip_values(self):
        return self.clip_min, self.clip_max

    @property
    def all_framework_preprocessing(self):
        return self._all_framework_preprocessing

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)
        if self.nb_classes == 2:
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)

        outputs = self.torch_model(self.convertion(x_tensor))
        loss = self.loss_computer(outputs, y_tensor)

        grads = torch.autograd.grad(loss, x_tensor)[0]
        return grads.detach().cpu().numpy()

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.torch_model(self.convertion(x_tensor))
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        return preds

    def fit(self, x, y, **kwargs):
        pass


class PGDAttack(BaseAttack):
    """PGD attack on model."""

    def __init__(
        self,
        model: torch.nn.Module,
        mean: tuple, 
        std: tuple, 
        height: int,
        width: int,
        nb_classes: int = 10,
        is_color: bool = True,
        max_iter: int = 5,
        eps: float = 0.05,
        eps_step: float = 0.01,
        verbose: bool = False,
        **kwargs
    ) -> None:
        super().__init__(model, mean, std, height, width)

        estimator = PGDEstimator(
            model=model,
            convertion=self._transform_output,
            input_shape=(3, height, width) if is_color else (height, width),
            nb_classes=nb_classes,
            clip_max=1.0,
            clip_min=0.0,
            preprocessing=(0, 1),
        )
        self.pgd_art = ProjectedGradientDescent(estimator=estimator, max_iter=max_iter, eps=eps, eps_step=eps_step, verbose=verbose)

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.device = x.device
        x = self._transform_input(x)
        x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()

        x = torch.from_numpy(self.pgd_art.generate(x_np, y_np))
        x_mean = x.mean(dim=1, keepdim=True) 
        x = x_mean.repeat(1, 3, 1, 1)    
        x = self._transform_output(x.to(self.device))
        return x.to(self.device), y
