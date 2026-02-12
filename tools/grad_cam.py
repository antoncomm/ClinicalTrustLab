"""Grad cam"""

import os
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from neurone.utils.configs import (
    setup_model,
    setup_engine,
)
from neurone.train.procedures import setup_data
from neurone.utils.visualization import visualize
from neurone.utils.grad_cam import GradCAM
from neurone.utils.config_class import Config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Config):
    """
    Main function that runs grad_cam.
    python tools/grad_cam.py 'run.cuda_visible_devices="<cuda_number>"' run.engine_name=<engine_name>

    FIXME
        1. Use dataloader
        2. Use get_data
        3. Add to cla test mode
    """
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)

    if cfg["run"]["cuda_visible_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["run"]["cuda_visible_devices"]

    dataset = setup_data(cfg)["valid"]

    print(f"{dataset.dataset.datasets[0].transforms = }")

    images = []
    labels = []
    os.makedirs(cfg["grad_cam"]["dir_to_res"], exist_ok=True)
    length = len(dataset)

    for i in (pbar := tqdm(range(length))):
        pbar.set_description("Loading images")
        im, label = dataset[i]
        label = int(label)
        images.append(im)
        labels.append(label)
        # visualize(save_obj=path_to_save, **{f'image_{label}': im})

    model = setup_model(
        config_model=cfg["model"], engine=setup_engine(cfg["run"]["engine_name"])
    )
    cam = GradCAM(model=model, config=cfg, engine_name=cfg["run"]["engine_name"])
    cam.transforms = None  # into account in dataset

    grayscale_cams = cam.generate_gradcams(images)
    cam_images = cam.get_cam_images(grayscale_cams=grayscale_cams)

    for i, cam_image in enumerate(pbar := tqdm(cam_images)):
        pbar.set_description("Saving images")
        prediction = cam.predictions[i].item()
        cam_output_path = os.path.join(
            cfg["grad_cam"]["dir_to_res"],
            f"cam_image_{i}_label={labels[i]}_prediction={prediction}.jpg",
        )
        images_kwargs = {
            f"image_{labels[i]}": images[i],
            f"cam_image_{prediction}": cam_image,
        }
        visualize(save_obj=cam_output_path, **images_kwargs)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
