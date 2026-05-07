"""EarlyStopping class"""

from typing import Dict, List, Union
import numpy as np


class EarlyStopping:
    """EarlyStopping is called inside class Base
    Args
    ----
    track_metric:
        name of the metric that are used to track accuracy of model
    best_metric:
        best value of track_metric
    threshold:
        threshold for measuring the new optimum, to only focus on
        significant changes. Default: 1e-4.
    patience:
        Number of epochs with no improvement after which experiments
        will be terminated.
    """

    def __init__(
        self, track_metric: str = "loss", threshold: float = 1e-4, patience: int = 10
    ) -> None:
        self.track_metric = track_metric
        self.best_metric = np.inf if track_metric == "loss" else -np.inf
        self.threshold = threshold
        self.patience = patience
        self.counter = 0

    def check(self, loss: float, metric: Dict[str, Union[str, List[float]]]) -> bool:
        """
        Checks whether early stopping criterion is met.

        Args
        ----
        loss: The current loss value for validation dataset.
        metric:
            The current value of the tracked metric for validation dataset.
            It should be a dictionary containing the names and corresponding
            values of the metrics.

        Returns
        -------
        True if early stopping criterion is met, False otherwise.
        """
        if self.track_metric == "loss":
            value = loss
            improvement = self.best_metric - loss
        else:
            idx = metric["name"].index(self.track_metric)
            value = metric["value"][idx]
            improvement = value - self.best_metric

        if improvement > 0:
            self.best_metric = value

        if improvement > self.threshold:
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True

        return False
