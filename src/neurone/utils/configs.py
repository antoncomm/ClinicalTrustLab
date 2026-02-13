"""
The functions to configure config.
"""

import os.path
import copy
from collections.abc import MutableMapping
from functools import reduce
import operator
import json
from omegaconf import OmegaConf
import yaml
import albumentations as A
import cv2
import segmentation_models_pytorch as smp

# torch imports
import torch
from lion_pytorch import Lion

# Engines
from animus.torch.engine import CPUEngine, DDPEngine, DPEngine, GPUEngine, XLAEngine

# neurone imports
from neurone.train.metrics import Metric
from neurone.train.schedulers import Scheduler

from neurone.train.losses import (
    FocalLoss,
    GaussianFocalLoss,
)

E2E = {
    "cpu": CPUEngine,
    "gpu": GPUEngine,
    "dp": DPEngine,
    "ddp": DDPEngine,
    "xla": XLAEngine,
}


def get_config(config_path):
    """
    Get the config from a json file.
    """
    with open(config_path) as f:
        config = json.load(f)
    return config


def define_metrics(metrics, thresholds=None, **shared_kwargs):
    """
    Define the metrics to be used.
    """
    return Metric(metrics, thresholds=thresholds, **shared_kwargs)


def define_optimizer(optimizer_config, model):
    """
    Define the optimizer.
    """
    if optimizer_config["type"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_config["kwargs"])
    elif optimizer_config["type"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config["kwargs"])
    elif optimizer_config["type"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config["kwargs"])
    elif optimizer_config["type"] == "Lion":
        optimizer = Lion(model.parameters(), **optimizer_config["kwargs"])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")

    return optimizer


def define_criterion(criterion_config):
    """
    Define the criterion.
    """
    if criterion_config["type"] is None:
        return None
    elif criterion_config["type"] == "Focal":
        criterion = FocalLoss(**criterion_config["kwargs"])
    elif criterion_config["type"] == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss(**criterion_config["kwargs"])
    elif criterion_config["type"] == "Dice":
        criterion = smp.losses.DiceLoss(**criterion_config["kwargs"])
    elif criterion_config["type"] == "GaussianFocal":
        criterion = GaussianFocalLoss(**criterion_config["kwargs"])
    elif criterion_config["type"] == "BCEWithLogitsLoss":
        # FIXME
        if criterion_config["kwargs"].get("pos_weight") is not None:
            criterion_config["kwargs"]["pos_weight"] = torch.tensor(
                criterion_config["kwargs"]["pos_weight"]
            )
        criterion = torch.nn.BCEWithLogitsLoss(**criterion_config["kwargs"])
    else:
        raise Exception("Unknown criterion: %s" % criterion_config)

    return criterion


def define_scheduler(config, optimizer):
    scheduler = Scheduler(
        config["run"]["scheduler"]["type"] if config["mode"] != "test" else None,
        config["run"]["scheduler"].get("track_metric"),
        optimizer,
        **config["run"]["scheduler"].get("kwargs", {}),
    )
    return scheduler


def get_in_dict(dictionary, key_list):
    """
    "Get element from nested dictionary with list of keys"

    Parameters
    ----------
    dictionary: dict
        Dictionary to take values from
    key_list: list
        List of keys used to access element data.

    Returns
    -------
    Value of element
    """
    return reduce(operator.getitem, key_list, dictionary)


def set_in_dict(dictionary, key_list, value):
    """
    "Set value for element in nested dictionary with list of keys"

    Parameters
    ----------
    dictionary: dict
        Dictionary to set values in
    key_list: list
        List of keys used to set element
    value: Any
        Value to set in dictionary.
    """
    get_in_dict(dictionary, key_list[:-1])[key_list[-1]] = value


def save_config(config: dict, path: str, overwrite: bool = True):
    """
    Parameters
    ----------
    config:
        The configuration to be saved.
    path:
        The path where the configuration should be saved.
    overwrite:
        A flag indicating whether to overwrite the existing configuration file.
        Default is `True`.

    Raises
    ------
    ValueError
        If the file extension is not supported (.json, .yaml, or .yml).
    """
    if (not os.path.exists(path)) or overwrite:
        file_extension = os.path.splitext(path)[1]
        if file_extension == ".json":
            with open(path, "w", encoding="UTF-8") as f:
                json.dump(config, f, indent=4)
        elif file_extension in (".yaml", ".yml"):
            with open(path, "w", encoding="UTF-8") as f:
                OmegaConf.save(config, f)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")


def generate_gridsearch_config(
    config: dict, dict_of_params_to_replace: dict, sep: str = "@"
):
    """
    "Generate gridsearch config from basic and dict of tunable parameters"

    Parameters
    ----------
    config: dict
        base gridsearch config
    dict_of_params_to_replace: dict
        Dictionary of parameters to replace in base config
    sep: str
        Which separator used to separate keys in dictionary

    Returns
    -------
    New config, with parameters set
    """
    base_config = copy.deepcopy(config)
    for k, v in dict_of_params_to_replace.items():
        keys = k.split(sep)
        set_in_dict(base_config, keys, v)
    return base_config


def flatten_dict(
    dictionary: MutableMapping, parent_key: str = "", sep="@"
) -> MutableMapping:
    """
    "Recursive flatten given dictionary with separator"

    Parameters
    ----------
    dictionary: dict
        Dictionary to flatten
    parent_key: str
        Relative key path
    sep: str
        Separator to separate keys

    Returns
    -------
    Flattened dictionary
    """
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_gridsearch_config(grid_search_config, keysep="@"):
    """
    "get every tuneable parameter from grid-search config"

    Parameters
    ----------
    grid_search_config: dict
        grid search config
    keysep: str
        separator used to separate keys

    Returns
    -------
    Dictionary of parameters to check
    """
    flatten = flatten_dict(grid_search_config, sep=keysep)
    params = {}
    tunable_params = []
    for k, value in flatten.items():
        if k.split(keysep)[-1] == "gridsearch_tunable_params":
            tunable_params = value
        if isinstance(value, list):
            params[k] = value
    filtered_params = {}

    for k, value in params.items():
        if k.split(keysep)[-1] in tunable_params:
            filtered_params[k] = value
        else:
            pass
    return filtered_params


def setup_train(config, model):
    """
    Setup things for training such as optimizer etc.
    """
    criterion = define_criterion(config["run"]["criterion"])
    optimizer = define_optimizer(config["run"]["optimizer"], model)

    metric = define_metrics(
        config["run"]["metrics"],
        config["model"]["thresholds"],
        **config["run"].get("metrics_shared_kwargs", {}),
    )

    scheduler = define_scheduler(config, optimizer)

    return criterion, optimizer, scheduler, metric


def initialize_model(config_model, map_location="cpu"):
    """
    Instantiate the model and load weights if specified.

    Args:
        config_model (dict): Configuration for the model including its name, kwargs, and path to weights.
        map_location (str, optional): The device for mapping the loaded weights. Defaults to 'cpu'.

    Returns:
        torch.nn.Module: The initialized model.
    """
    # pylint: disable=import-outside-toplevel
    from neurone import models

    # Instantiate the model
    model_class = getattr(models, config_model["model_name"])
    model = model_class(**config_model["model_kwargs"])

    # Load weights if provided
    if config_model.get("weights_path") is not None:
        model.load_state_dict(
            torch.load(config_model["weights_path"], map_location=map_location)
        )

    return model


def setup_model(config_model, engine):
    """Initialize and prepare model"""
    model = initialize_model(config_model, map_location=engine.device)
    model = engine.prepare(model)

    return model


def define_extrapolation_mode(extrapolation_mode):
    if extrapolation_mode == "BORDER_CONSTANT":
        return cv2.BORDER_CONSTANT
    elif extrapolation_mode == "BORDER_REPLICATE":
        return cv2.BORDER_REPLICATE
    elif extrapolation_mode == "BORDER_WRAP":
        return cv2.BORDER_WRAP
    elif extrapolation_mode == "BORDER_REFLECT":
        return cv2.BORDER_REFLECT
    elif extrapolation_mode == "BORDER_REFLECT_101":
        return cv2.BORDER_REFLECT_101
    else:
        raise Exception(
            "\nExtrapolation mode should be one of the following: \n"
            + "BORDER_CONSTANT, BORDER_REPLICATE,BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101 \n"
            + "Got:\n%s" % extrapolation_mode
        )


def albumentations_from_config(aug_config):
    """"""
    border_mode = define_extrapolation_mode(aug_config["border_mode"])
    augs_list = [
        A.augmentations.HueSaturationValue(
            p=aug_config["p_hsv"],
            hue_shift_limit=5,
            sat_shift_limit=5,
            val_shift_limit=5,
        ),
        A.augmentations.GaussNoise(
            p=aug_config["p_noise"], var_limit=aug_config["noise_var"]
        ),
        A.augmentations.Rotate(
            limit=aug_config["rotate_angle"],
            p=aug_config["p_rotate"],
            border_mode=border_mode,
        ),
        A.augmentations.ShiftScaleRotate(
            shift_limit=aug_config["shift_factor"],
            scale_limit=0,
            rotate_limit=0,
            border_mode=border_mode,
            p=aug_config["p_shift"],
        ),
        A.augmentations.ShiftScaleRotate(
            shift_limit=0,
            scale_limit=aug_config["scale_factor"],
            rotate_limit=0,
            border_mode=border_mode,
            p=aug_config["p_scale"],
        ),
        A.augmentations.Perspective(
            scale=(0, aug_config["perspective_factor"]),
            p=aug_config["p_perspective"],
            interpolation=border_mode,
        ),
        A.augmentations.HorizontalFlip(p=aug_config["p_flip_hor"]),
        A.augmentations.VerticalFlip(p=aug_config["p_flip_vert"]),
    ]

    return augs_list


# pylint: disable=too-many-locals
def compose_transforms(transforms):
    """setup transforms"""
    # pylint: disable=import-outside-toplevel
    # pylint: disable=possibly-unused-variable
    from albumentations import HueSaturationValue, GaussNoise, Rotate
    from albumentations import ShiftScaleRotate, Perspective, HorizontalFlip
    from albumentations import Affine, VerticalFlip, RandomCrop, Resize, Compose
    from albumentations import Sharpen, AdvancedBlur, CLAHE, GridDistortion
    from albumentations import CenterCrop, BboxParams, Normalize, GridDropout
    from albumentations import RandomBrightnessContrast, CoarseDropout, PixelDropout
    from albumentations import RandomGridShuffle
    from albumentations.pytorch import ToTensorV2

    transforms_list = []
    for transform, kwargs in transforms["augmentations"].items():
        transforms_list.append(locals()[transform](**kwargs))
    if transforms.get("params", {}):
        params = {
            transforms["params"]["arg"]: locals()[transforms["params"]["type"]](
                **transforms["params"]["kwargs"]
            )
        }
        transforms_composed = Compose(transforms_list, **params)
    else:
        kwargs = transforms.get("kwargs", {})
        transforms_composed = Compose(transforms_list, **kwargs)

    return transforms_composed


def setup_config(config_path):
    """
    Setup json or yaml config.

    Args:
        config_path (str): path to config file

    Returns:
        config (dict): config dictionary
    """
    file_extension = os.path.splitext(config_path)[1]
    with open(config_path, "r", encoding="UTF-8") as config_file:
        if file_extension == ".json":
            config = json.load(config_file)
        elif file_extension == ".yaml" or file_extension == ".yml":
            config = yaml.safe_load(config_file)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    return config


def setup_engine(engine_name, tracker="tensorboard", project_dir="logs"):
    """
    Setup engine.

    Args:
        engine_name (str): name of the engine to use
        tracker (str): name of the tracker to use
        project_dir (str): name of the project directory to use

    Returns:
        engine object

    """
    engine = E2E[engine_name](log_with=tracker, project_dir=project_dir)
    return engine
