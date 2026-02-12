"""Utils for models"""

from collections import defaultdict
import torch
import pandas as pd
from tabulate import tabulate


def count_parameters(model, trainable=None):
    """
    Args:
        trainable=None: count all parameters
        trainable=True: counts trainable parameters
        trainable=False: counts non-trainable parameters
    """
    if trainable is None:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    elif trainable is True or trainable is False:
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad == trainable
        )
    else:
        raise ValueError(
            f"Unknown trainable: {trainable}. `trainable` must be from {None, True, False}"
        )

    return pytorch_total_params


def get_mode(model):
    """Get whether a model is in train or eval mode."""
    if model.training:
        return "Train"
    else:
        return "Eval"


def get_params(model, do_print=True):
    """
                      Parameters Trainable parameters (M) Frozen parameters (M) Mode
    encoder                63.79                     0.00                 63.79 Eval
    decoder                 3.31                     3.31                  0.00 Train
    segmentation_head       0.00                     0.00                  0.00 Eval
    """

    data = defaultdict(list)
    df = pd.DataFrame()
    for name, child in model.named_children():
        minn, maxx = min_max_params(model=child)
        data["Child"].append(name)
        data["Parameters (M)"].append(count_parameters(child, trainable=None) / 1e6)
        data["Trainable parameters (M)"].append(
            count_parameters(child, trainable=True) / 1e6
        )
        data["Frozen parameters (M)"].append(
            count_parameters(child, trainable=False) / 1e6
        )
        data["Min"].append(minn)
        data["Max"].append(maxx)
        data["Mode"].append(get_mode(child))

    minn, maxx = min_max_params(model=model)
    data["Child"].append("total")
    data["Parameters (M)"].append(count_parameters(model, trainable=None) / 1e6)
    data["Trainable parameters (M)"].append(
        count_parameters(model, trainable=True) / 1e6
    )
    data["Frozen parameters (M)"].append(count_parameters(model, trainable=False) / 1e6)
    data["Min"].append(minn)
    data["Max"].append(maxx)
    data["Mode"].append(get_mode(model))
    df = pd.DataFrame(data)
    if do_print:
        print(
            tabulate(
                df, headers="keys", tablefmt="psql", floatfmt=".4f", showindex=False
            )
        )
    return df


def unfreeze_model(*models):
    """Unfreeze all *models

    Note that `*models` allows the method to be used for both a single
    model and multiple models at once.
    """

    for model in models:
        for param in model.parameters():
            param.requires_grad = True


def freeze_model(*models):
    """Freeze all *models.

    Note that `*models` allows the method to be used for both a single
    model and multiple models at once.
    """
    for model in models:
        for param in model.parameters():
            param.requires_grad = False


def min_max_params(model):
    """Calculate the minimum and maximum values among all parameters in this
    PyTorch model to ensure that the weights are loaded correctly. As a rule,
    weights with random initialization have fixed min and max values.


    References:
        - https://pytorch.org/docs/stable/generated/torch.rand.html
        - https://discuss.pytorch.org/t/manually-change-assign-weights-of-a-neural-network/115444
    """
    minn = torch.tensor(float("inf"))
    maxx = torch.tensor(-float("inf"))
    # pylint: disable=unused-variable
    for _, param in model.named_parameters():
        minn = torch.min(minn, param.data.min())
        maxx = torch.max(maxx, param.data.max())

    return float(minn), float(maxx)
