from functools import partial
import torch
from .base_attack import BaseAttack
from art.attacks.evasion import SquareAttack as SqareARTAttack
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification import ClassifierMixin
import numpy as np


class SquareEstimator(BaseEstimator, NeuralNetworkMixin, ClassifierMixin):
    def __init__(
        self,
        model: torch.nn.Module,
        convertion: callable, 
        channels_first: bool = True,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_max=1.0,
        clip_min=0.0,
        preprocessing=(0, 1),
    ):
        self.convertion = convertion
        self._input_shape = input_shape
        self._nb_classes = nb_classes

        self._channels_first = channels_first
        self.torch_model = model

        self.clip_max = clip_max
        self.clip_min = clip_min
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._all_framework_preprocessing = preprocessing

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

    @property
    def channels_first(self):
        return self._channels_first

    def predict(self, x: np.ndarray, batch_size: int = 64, **kwargs) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.torch_model(self.convertion(x_tensor))
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        return preds

    def fit(self, x, y, **kwargs):
        pass

    def get_activations(self, x, layer, batch_size, **kwargs) -> np.ndarray:
        pass



class SquareAttack(BaseAttack):
    """Square attack on model."""

    def __init__(
        self,
        model: torch.nn.Module,
        mean: tuple, 
        std: tuple, 
        height: int,
        width: int,
        nb_classes: int = 10,
        is_color: bool = True,
        eps: float = 0.05,
        **kwargs
    ) -> None:
        super().__init__(model, mean, std, height, width)

        estimator = SquareEstimator(
            model=model,
            convertion=self._transform_output,
            input_shape=(3, height, width) if is_color else (height, width),
            nb_classes=nb_classes,
            clip_max=1.0,
            clip_min=0.0,
            preprocessing=(0, 1),
        )
        self.square_art = SqareARTAttack(estimator=estimator, eps=eps)

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.device = x.device
        x = self._transform_input(x)
        x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        return self._transform_output(torch.from_numpy(self.square_art.generate(x_np, y_np)).to(self.device)), y
