import os
import cv2
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Any, Union, Callable, List, Optional, Tuple, Dict

from torch import Tensor
import torch

from neurone.utils.configs import (
    setup_config,
    setup_engine,
    compose_transforms,
    setup_engine,
)
from neurone.utils.visualization import torch_image_to_numpy

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
    GradCAMElementWise,
    GuidedBackpropReLUModel,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image,
)


class GradCAM:
    def __init__(self, config: Union[str, dict], model, engine_name: str = "gpu"):
        if type(config) is str:
            self.config = setup_config(config)
        elif type(config) is dict:
            self.config = config
        else:
            raise Exception(
                f"Type of config must be str or dict, but you have{type(config)}"
            )
        self.engine_name = engine_name
        self.engine = setup_engine(engine_name)
        self.model = model.to(self.engine.device)
        self.transforms = compose_transforms(
            self.config["data"]["transforms_inference"]
        )
        self.model.eval()

        if (model_name := self.config["model"]["model_name"]) == "EfficientNet":
            self.target_layers = [self.model.model.features[8]]  # for grad_cam
        else:
            raise ValueError(f"Unknown model_name type: {model_name}")

    def transform_images(self, images: Union[Any, List[Any]]) -> List[Tensor]:
        images = deepcopy(images)
        if type(images) is not list:
            images = [images]
        batched_images = []
        for image in images:
            if self.transforms is not None:
                transformed = self.transforms(image=image)["image"]
            else:
                transformed = image
            transformed = transformed.to(self.engine.device)
            batched_images.append(transformed)
        batched_images = torch.stack(batched_images, axis=0)
        return batched_images

    def generate_gradcams(self, images):
        images = self.transform_images(images)
        self.images = images
        # images.requires_grad = True

        cam_algorithm = globals()[self.config["grad_cam"]["method"]]

        targets = None  # FIXME: What does it affect?

        self.predictions = []
        grayscale_cams = []
        with cam_algorithm(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=(self.engine_name == "gpu"),
        ) as cam:
            for image in (pbar := tqdm(images)):
                pbar.set_description("Getting grayscale_cams")  # FIXME: add verbose
                grayscale_cams.append(
                    cam(
                        input_tensor=torch.stack([image], axis=0),
                        targets=targets,
                        # aug_smooth=True,
                        # eigen_smooth=True
                    )[0, :]
                )
                with torch.no_grad():
                    self.predictions.append(
                        self.model.do_threshold(
                            self.model(torch.stack([image], axis=0))
                        )
                    )
        grayscale_cams = np.array(grayscale_cams)
        return grayscale_cams

    def get_cam_images(self, grayscale_cams):
        cam_images = []
        for i in range(len(self.images)):
            grayscale_cam = grayscale_cams[i, :]
            rgb_img = torch_image_to_numpy(self.images[i])
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cam_images.append(cam_image)
        return cam_images
