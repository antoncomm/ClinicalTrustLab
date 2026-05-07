"""ctl augmentations"""

import numpy as np
import albumentations as A

# ctl imports
from ctl.data.utils import load_image


def mask_crop(inputs: dict, path_to_mask: str, p: float = 0.5) -> dict:
    """
    Crops an input image using a provided mask. Also adds the capability of a random chance to return the inputs as is.

    Parameters
    ----------
    inputs: The input image as an np.array in a dictionary with key "image".
    path_to_mask: Path to the image file to be used as a mask.
    p: Probability threshold to return the inputs as is, defaults to 0.5.

    Returns
    -------
    If np.random.rand() > p, it returns inputs unmodified. Otherwise, it performs a cropping operation on the input
    image based on the nonzero regions of the mask, applies the mask to the image and returns the masked and cropped image.
    """

    if np.random.rand() > p:
        return inputs
    mask = load_image(path_to_mask)  # (H, W, 3)
    y, x = np.nonzero(mask[:, :, 0])
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    crop = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    image = inputs["image"]
    del inputs["image"]
    image_and_mask = image * mask
    transformed = crop(image=image_and_mask, **inputs)
    return transformed
