"""
Dataclasses for config files.
"""

from dataclasses import dataclass
from typing import Union


@dataclass
class Model:
    """
    Dataclass for model section.
    """

    runner: str
    model_name: str
    model_kwargs: dict
    weights_path: str
    thresholds: dict


@dataclass
class Data:
    """
    Dataclass for data section.
    """

    dataset: Union[str, list]
    dataset_dir: str
    ann_files: dict
    kwargs: dict
    ignore_classes: list
    class_map: dict
    transforms_train: dict
    transforms_test: dict
    transforms_inference: dict


@dataclass
class Run:
    """
    Dataclass for run section.
    """

    batch_size: int
    num_epochs: int
    log_every: int
    verbose: bool
    seed: int
    workers: int
    tracker: str
    checkpoint_every: int
    save_dir: str
    checkpoints_dir: str
    project_dir: str
    experiment_name: str
    overwrite: bool
    interpolation: int
    deterministic: bool
    engine_name: str
    cuda_visible_devices: Union[None, str]
    criterion: dict
    metrics_dir: str
    metrics: list
    prime_metric: int
    optimizer: dict
    scheduler: dict


@dataclass
class Config:
    """
    Dataclass for config.
    """

    model: Model
    data: Data
    run: Run
    mode: str
