"""
Scheduler class.
"""

# pylint: disable=unused-import
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    ReduceLROnPlateau,
)

# pylint: disable=unused-import
from transformers import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class Scheduler:
    """
    Scheduler
    """

    def __init__(
        self, type: str, track_metric: str, optimizer, **scheduler_config
    ) -> None:
        self.type = type
        self.track_metric = track_metric
        if self.type is None:
            self.scheduler = None
        else:
            self.scheduler = globals()[type](optimizer, **scheduler_config)

    def step(self, loss, metric):
        """
        This method is a wrapper for step method for every scheduler.
        """
        if self.scheduler:
            if self.track_metric is None:
                self.scheduler.step()
            elif self.track_metric == "loss":
                self.scheduler.step(loss)
            else:
                idx = metric["name"].index(self.track_metric)
                value = metric["value"][idx]
                self.scheduler.step(value)
