"""
This module contains procedures for the training.
"""

from typing import Dict, Tuple
from copy import deepcopy
from torch.utils.data import Dataset
from ctl.utils.configs import compose_transforms
from ctl.data.datasets import *


def setup_data(config: Dict) -> Dict[str, Dataset]:
    """
    Compose datasets.

    Parameters
    ----------
    config: Dict
        Config file.

    Returns
    -------
    datasets: Dict[String: Dataset]
        Dict with train, valid, test datasets.
    """
    datasets = {}

    dataset_cfg = config["datasets"] 
    transforms_cfg = config.get("transforms", {"augmentations": {}, "params": {}})
    train_datasets, valid_datasets, test_datasets = [], [], []
    for dataset_config in dataset_cfg.values():
        dataset_config["attack_params"] = {}
        if config["attack"] and 'attack_dataset' in config["attack"]:
            dataset_config["attack_params"] = config["attack"]["attack_params"]
        train, valid, test = setup_one_dataset(dataset_config, transforms_cfg)
        if train is not None:
            train_datasets.append(train)
        if valid is not None:
            valid_datasets.append(valid)
        if test is not None:
            test_datasets.append(test)

    datasets["train"] = BaseConcatDataset(train_datasets) if train_datasets else None
    datasets["valid"] = BaseConcatDataset(valid_datasets) if valid_datasets else None
    datasets["test"] = BaseConcatDataset(test_datasets) if test_datasets else None

    return datasets


def setup_transforms(transforms_config: Dict):
    """Return transforms_train and transforms_test"""
    transforms_train = compose_transforms(transforms_config)
    augs_test = {}
    for aug, kwargs in transforms_config["augmentations"].items():
        if aug in ("Resize", "Normalize", "ToTensorV2"):
            augs_test[aug] = kwargs
    assert len(augs_test) <= 3
    transforms_test = {
        "augmentations": augs_test,
        "params": transforms_config.get("params", {}),
        "kwargs": transforms_config.get("kwargs", {}),
    }
    transforms_test = compose_transforms(transforms_test)

    return transforms_train, transforms_test


def setup_one_dataset(
    dataset_config: Dict, transforms_config: Dict
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup one dataset from config.

    Parameters
    ----------
    config: Dict
        Config file.

    Returns
    -------
    datasets: Tuple[Dataset]
        Tuple with train, valid, test datasets.
    """
    transforms_train, transforms_test = setup_transforms(transforms_config)

    dataset_class = globals().get(dataset_config.get("dataset"))
    if dataset_class is None:
        raise ValueError(
            "Dataset '{}' not found.".format(dataset_config.get("dataset"))
        )

    datasets = {}

    for dataset_type in "train", "valid", "test":
        datasets[dataset_type] = (
            dataset_class(
                root=dataset_config["dataset_dir"],
                transforms=(
                    transforms_train if dataset_type == "train" else transforms_test
                ),
                ignore_classes=dataset_config.get("ignore_classes", []),
                class_map=dataset_config.get("class_map", {}),
                annfile=dataset_config["ann_files"].get(dataset_type),
                dataset_type=dataset_type,
                attack_params=dataset_config["attack_params"],
                **dataset_config.get("kwargs", {}),
                **dataset_config.get(f"{dataset_type}_kwargs", {}),
            )
            if dataset_config["ann_files"].get(dataset_type)
            else None
        )

    return datasets["train"], datasets["valid"], datasets["test"]
