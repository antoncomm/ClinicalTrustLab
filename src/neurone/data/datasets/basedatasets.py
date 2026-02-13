"Base dataset classes."

import os
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from neurone.utils.general import get_kwarg
from neurone.data.utils import load_image
from neurone.train.augmentations import mask_crop


class BaseDataset(Dataset):
    """
    Base dataset class.

    Args:
        root (str): Root directory of dataset.
        annfile (str): Path to annotation file.
        transforms (albumentations.Compose): Transforms to apply to image.
        ignore_classes (list): List of classes to ignore.
        class_map (dict): Dictionary mapping classes to new classes.
    """

    def __init__(
        self,
        root: str,
        annfile: str,
        transforms=None,
        ignore_classes: List = None,
        class_map: Dict = None,
        **kwargs,
    ) -> None:
        self.root = root
        self.annfile = annfile
        self.transforms = transforms
        if ignore_classes is None:
            ignore_classes = []
        self.ignore_classes = ignore_classes
        if class_map is None:
            class_map = {}
        self.class_map = class_map
        self.classes = []
        self.num_classes = 0
        self.ratio = get_kwarg("ratio", kwargs, 1)

    def _setup_annotation(self) -> None:
        "Function to setup annotation."
        raise NotImplementedError

    def _load_image(self, path) -> np.array:
        """
        Loads an image from the dataset
        """
        image = load_image(path)
        return image

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> Tuple:
        raise NotImplementedError

    def collate_fn(self, samples) -> Tuple:
        "Function to collate samples into a batch."
        raise NotImplementedError


class BaseMammographyDataset(BaseDataset):
    """
    Base dataset for Mammography.
    """

    def __init__(
        self,
        root: str,
        annfile: str,
        transforms=None,
        ignore_classes: List = None,
        class_map: Dict = None,
        **kwargs,
    ) -> None:
        super().__init__(root, annfile, transforms, ignore_classes, class_map, **kwargs)
        self.mask_dir = kwargs.get("mask_dir", None)
        self.p_mask = kwargs.get(
            "p_mask", 0
        )  # 0 means mask with crop disabled, 1 -- works always

    def _setup_annotation(self) -> None:
        "Function to setup annotation."
        raise NotImplementedError

    def _load_image(self, path: str) -> np.array:
        image = load_image(path)
        if self.mask_dir is not None:
            path_to_mask = path.replace(self.root, self.mask_dir)
            root, _ = os.path.splitext(path_to_mask)  # root, ext
            path_to_mask = root + ".png"
            image = mask_crop({"image": image}, path_to_mask, self.p_mask)["image"]
        return image

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> Tuple:
        raise NotImplementedError

    def collate_fn(self, samples) -> Tuple:
        "Function to collate samples into a batch."
        raise NotImplementedError


class BaseClassificationDataset(BaseDataset):
    """
    Base classification dataset class.

    Examples:
        >>> from neurone.data.datasets import BaseClassificationDataset
        >>> dataset = BaseClassificationDataset(root="data/train",
        >>>                                     annfile="data/train.csv")
        >>> dataset[0]
    """

    def __init__(
        self,
        root: str,
        annfile: str,
        transforms=None,
        ignore_classes: List = None,
        class_map: Dict = None,
        **kwargs,
    ) -> None:
        super().__init__(root, annfile, transforms, ignore_classes, class_map, **kwargs)
        self.data = None
        self._setup_annotation()

    def _setup_annotation(self) -> None:
        data = pd.read_csv(self.annfile)

        data = data[~data["label"].isin(self.ignore_classes)]
        data["label"] = data["label"].map(self.class_map)

        self.data = data
        self.classes = sorted(data["label"].unique())
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        l = self.data.shape[0]

        if not (0 < self.ratio <= 1):
            raise ValueError(
                f"The ratio ({self.ratio}) should be in the range of 0 to 1"
            )

        if self.ratio != 1:
            l = int(l * self.ratio)
        return l

    def get_feature_values(self, names: List[Optional[str]]) -> Dict[str, List]:
        """
        Retrieve feature values for specified feature names across multiple datasets.

        Args:
            names (List[str]): A list of feature names for which values are to be retrieved.

        Note:
            It takes into account that self.ratio can be less than 1.
        """
        l = len(self)
        return {name: self.data[name].to_list()[:l] for name in names if name}

    def __getitem__(self, index) -> Tuple:
        sample = self.data.iloc[index, :]
        label = sample["label"]
        image = self._load_image(os.path.join(self.root, sample["image"]))

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.float32)
        # label_ohe = one_hot(label, num_classes=self.num_classes)
        return image, label

    def collate_fn(self, samples) -> Tuple:
        "Function to collate samples into a batch."
        images = []
        labels = []

        for sample in samples:
            images.append(sample[0])
            labels.append(sample[1])

        images = torch.stack(images, 0)
        targets = {
            "labels": torch.stack(labels, 0).unsqueeze(1),
        }

        return images, targets
