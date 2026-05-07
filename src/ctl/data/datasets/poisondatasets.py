"Poison classification datasets"

from .classification import MammographyClassification
from typing import Tuple
import torch
from math import floor
import torch.nn.functional as F
import random


class BadNetPoisonMammographyClassification(MammographyClassification):
    """
    BadNet poison classification dataset for Mammography.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attack_params = kwargs["attack_params"]
        self.p = self.attack_params["p"]
        self.patch_size = self.attack_params["patch_size"]
        self.value = self.attack_params["value"]
        self.value = self.attack_params["value"]
        self.dataset_type = kwargs["dataset_type"]
        self.prob = torch.bernoulli(torch.full(size=(len(self),), fill_value=self.p))

    def __getitem__(self, index) -> Tuple:
        image, label = super().__getitem__(index)

        if self.dataset_type == "train":
            pois = self.prob[index]
            if pois:
                return self._make_patch(image), self.value
            return image, label
        elif self.dataset_type == "valid":
            return image, label
        elif self.dataset_type == "test":
            return self._make_patch(image), self.value
        
        raise ValueError(
            "Dataset_type should be train, valid ot test."
        )


    def _make_patch(self, x: torch.Tensor) -> torch.Tensor:
        patch_size = floor(min(self.patch_size * x.shape[1], self.patch_size * x.shape[2]))
        x_pois = x.clone()
        x_pois[:, 0:patch_size, 0:patch_size] = (torch.arange(patch_size**2) % 2).reshape(patch_size, patch_size)
        return x_pois


class WaNetPoisonMammographyClassification(MammographyClassification):
    """
    WaNet poison classification dataset for Mammography.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attack_params = kwargs["attack_params"]
        self.p = self.attack_params["p"]
        self.p_noise = self.attack_params["p_noise"]
        self.value = self.attack_params["value"]
        self.k = self.attack_params["k"]
        self.s = self.attack_params["s"]
        self.dataset_type = kwargs["dataset_type"]
        self.prob = torch.rand((len(self),))

    def __getitem__(self, index) -> Tuple:
        image, label = super().__getitem__(index)
        pois = self.prob[index]

        if self.dataset_type == "train":
            image_pois = self._warp_x(image, pois)
            label_pois = label
            if pois < self.p:
                label_pois = self.value
            return image_pois, label_pois
        elif self.dataset_type == "valid":
            return image, label
        elif self.dataset_type == "test":
            return self._warp_x(image, 0), self.value
        
        raise ValueError(
            "Dataset_type should be train, valid ot test."
        )

    def _warp_x(self, x: torch.Tensor, p: float) -> torch.Tensor:
        """
        Apply a spatial warping or noise-based perturbation to an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.
        p : float
            A random probability value used to select the transformation mode.

        Returns
        -------
        torch.Tensor: The transformed image tensor.
        """
        self.grid = self._generate_wanet_grid(x.shape, self.k, self.s)
        x = x.to(self.device)
        if p < self.p:  # attack mode
            return F.grid_sample(x.unsqueeze(0), self.grid, align_corners=True, padding_mode="reflection").squeeze(0)
        elif p < self.p_noise + self.p:  # noise mode
            ins = torch.rand(1, x.shape[1], x.shape[2], 2, device=self.device) * 2 - 1
            ins[..., 0] /= x.shape[2]
            ins[..., 1] /= x.shape[1]

            grid_noise = torch.clamp(self.grid + ins, -1, 1).to(self.device)

            return F.grid_sample(x.unsqueeze(0), grid_noise, align_corners=True, padding_mode="reflection").squeeze(0)
        return x.clone()

    def _generate_wanet_grid(self, image_size: tuple, k: int = 4, s: float = 0.5) -> torch.Tensor:
        """
        Generate a smooth, randomized spatial warping grid (WaNet-style).

        Parameters
        ----------
        image_size : tuple
            Spatial size of the image.
        k : int
            Resolution of the low-dimensional noise grid before interpolation.
            Smaller values produce smoother deformations. Default is ``4``.
        s : float
            Scaling factor controlling the strength of the warp. Default is ``0.5``.

        Returns
        -------
        torch.Tensor: A normalized sampling grid of shape ``(1, H, W, 2)``.
        """
        # identity grid
        xs = torch.linspace(-1, 1, steps=image_size[1])
        ys = torch.linspace(-1, 1, steps=image_size[0])
        y, x = torch.meshgrid(ys, xs, indexing="ij")

        identity_grid = torch.stack((x, y), dim=2)[None]  # (1, H, W, 2)

        # low-res noise
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))

        noise_grid = F.interpolate(ins, size=image_size, mode="bicubic", align_corners=True).permute(
            0, 2, 3, 1
        )  # (1, H, W, 2)

        # scale independently for H and W
        noise_grid[..., 0] /= image_size[1]  # x-axis
        noise_grid[..., 1] /= image_size[0]  # y-axis

        grid = identity_grid + s * noise_grid
        grid = torch.clamp(grid, -1, 1)

        return grid


class LabelFlipMammographyClassification(MammographyClassification):
    """
    Labelflip classification dataset for Mammography.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attack_params = kwargs["attack_params"]
        self.p = self.attack_params["p"]
        self.num_classes = self.attack_params["num_classes"]
        self.dataset_type = kwargs["dataset_type"]
        self.prob = torch.bernoulli(torch.full(size=(len(self),), fill_value=self.p))

    def __getitem__(self, index) -> Tuple:
        image, label = super().__getitem__(index)
        if self.dataset_type == "train" and self.prob[index]:
            possible_labels = list(range(self.num_classes))
            possible_labels.remove(label)
            label = random.choice(possible_labels)
        return image, label
