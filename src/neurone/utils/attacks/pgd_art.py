import torch
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators import BaseEstimator, LossGradientsMixin
import numpy as np
from .base_attack import BaseAttack


class PGDEstimator(BaseEstimator, LossGradientsMixin):
    def __init__(
            self,
            model: torch.nn.Module,
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_max=1.0,
            clip_min=0.0,
            preprocessing=(0, 1)
    ):
        self.model = model

        self._input_shape = input_shape
        self._nb_classes = nb_classes

        self.clip_max = clip_max
        self.clip_min = clip_min
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._all_framework_preprocessing = preprocessing
        self.loss_computer = torch.nn.CrossEntropyLoss()

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
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)

        outputs = self.model(x_tensor)
        loss = self.loss_computer(outputs, y_tensor)
        grads = torch.autograd.grad(loss, x_tensor, retain_graph=True, create_graph=True)
        return grads

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.model(x_tensor)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        return preds

    def fit(self, x, y, **kwargs):
        pass


class PGDAttack(BaseAttack):
    """CW linf attack on model."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        nb_classes: int,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        **pgd_params
    ) -> None:
        """
        Init pgd attack.

        Parameters
        ----------
        model: torch.nn.Module,
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

        estimator = PGDEstimator(
            model=model,
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_max=clip_max,
            clip_min=clip_min,
            preprocessing=(0, 1),
        )
        self.pgd_art = ProjectedGradientDescent(estimator=estimator, **pgd_params)

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
        return torch.from_numpy(self.pgd_art.generate(x, y)), y
