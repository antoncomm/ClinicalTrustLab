"""
General utils.
"""

import os
import shutil as sh
from typing import Dict, List, Any
import subprocess
import warnings
import numpy as np
import cv2
import dicomsdl
import yaml
import torch
import torchvision.transforms.functional as fn


def get_hash_commit(branch: str) -> str:
    """Get the last hash from a branch or commit."""
    command = f"git rev-parse {branch}"

    if "origin/" not in branch:
        warnings.warn(
            "CalledProcessError: The branch name should start with 'origin/' to ensure it is tracking a remote branch."
        )

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        hash_commit = result.stdout.strip()
        return hash_commit
    except subprocess.CalledProcessError as e:
        warnings.warn(
            f"CalledProcessError: An error occurred while getting the last hash commit. Return code: {e.returncode}"
        )
        return branch


def save_patch_diff(branch: str, path_to_experiment: str):
    """
    Generate a git diff patch from the given commit or branch.

    Returns:
        str or None: The last hash commit from the branch or None if an error occurred.
    """

    # Get the hash commit from the branch or commit
    hash_commit = get_hash_commit(branch)

    # Define the output patch file name
    output_patch_file = os.path.join(path_to_experiment, "git_diff.patch")

    # Construct the command
    command = f"git diff {hash_commit} > {output_patch_file}"

    # Execute the command
    try:
        print(f"{command = }")
        _ = subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise f"An error occurred. Return code: {e.returncode}"

    return hash_commit


def makedir(dir_path):
    """
    Make directory by path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def makedir_overwrite(dir_path, overwrite):
    """Creates dir with specified path

    Parameters
    ----------
    dir_path: str
        path to dir to create
    overwrite: bool
        Specifies the behaviour for the case when the directory with dir_path exists
        If True, the dir will be deleted. If False, the dir will be left intact and the Exceptrion will be raised.
    """
    if os.path.exists(dir_path):
        if overwrite:
            sh.rmtree(dir_path)
        else:
            raise Exception(
                "The directory exists and overwrite mode is disabled: \n %s \n Aborting."
                % dir_path
            )
    os.makedirs(dir_path, exist_ok=True)


def load_yaml(path):
    """
    Safely loads contents from yaml file.

    Parameters
    ----------
    path: str
        path to yaml

    Returns
    -------
    contents: dict or list
        contents of yaml file
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(path, contents, overwrite=False):
    """
    Safely writes contents to the yaml file.

    Parameters
    ----------
    path: str
        path to yaml
    contents: dict or list
        contetns to write
    overwrite: bool
        Defines the behavior in case of already present file. If true, the file will be overwritten, if false, the exeption is raised.
    """

    if os.path.exists(path):
        raise FileExistsError("File %s exists and overwrite mode is disabled.")

    with open(path, "w+") as file:
        yaml.safe_dump(contents, file)


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def concat_resize(img_list, hw_dim=0, resize_to=None, interpolation=cv2.INTER_LINEAR):
    """
    Concatenation and resize of tensors with CHW format.

    Args:
        dim: 0 -> H, 1 -> W

    Returns:
        Concatanated image.
    """
    if resize_to is None:
        min_size = min(img.shape[hw_dim + 1] for img in img_list)
    else:
        min_size = resize_to

    img_list_resized = []
    for img in img_list:
        size = list(img.shape[1:])
        size[1 - hw_dim] = int(size[1 - hw_dim] * min_size / size[hw_dim])
        size[hw_dim] = min_size
        img_list_resized.append(fn.resize(img, size))
    return torch.cat(img_list_resized, hw_dim + 1)


def gridsearch_compare_logs(path_to_log1, path_to_log2):
    """
    Compare log files and return True, if second log is better by comparing both metric and loss. Returns False, if either log 1 or log 2 doesn't exist.

    Parameters
    ----------
    path_to_log1: str
        First log to compare with
    path_to_log2: str
        Second log to compare with

    Returns
    -------
    Bool, True, if second log is better. False, if path doesn't exist or if second isn't better.
    """
    if os.path.exists(path_to_log1) and os.path.exists(path_to_log2):
        string_1 = ""
        string_2 = ""
        metric_1 = 0
        metric_2 = 0
        with open(path_to_log1, "r") as f:
            for line in f:
                if line[:6] == " valid":
                    metric = line.split(" ")[8]
                    if metric > metric_1:
                        metric_1 = metric

        with open(path_to_log2, "r") as f:
            for line in f:
                if line[:6] == " valid":
                    metric = line.split(" ")[8]
                    if metric > metric_2:
                        metric_2 = metric

        if metric_1 < metric_2:
            return True
        else:
            return False
    else:
        raise ValueError("Logs don't exist at specified path")


def is_background(image, mean=225, std=14):
    """
    Check whether given image is background image and return True if that's true

    Parameters
    ----------
    image: np.array or torch.Tensor
        image to check
    mean: Int
        Mean threshold to filter background images
    std: Int
        Std threshold to filter background images
    """
    if image.mean() >= mean and image.std() <= std:
        return True
    else:
        return False


def is_background_batch(image, mean=225, std=14):
    """
    Check whether images in batch are background and return True if it is background and False otherwise

    Parameters
    ----------
    image: torch.Tensor
        [N_batch x C x W x H]
    mean: Int
        Mean threshold to filter background images
    std: Int
        Std threshold to filter background images
    """
    assert isinstance(image, torch.Tensor)
    image = image.float()
    image_mean = torch.mean(image, (1, 2, 3))
    image_std = torch.std(image, (1, 2, 3))
    return torch.where(
        (image_mean >= mean) & (image_std <= std), torch.tensor(), torch.tensor(0)
    ).bool()


def get_files_pathes(path, pattern, istype="file"):
    """
    Get pathes from input directory.
    """
    if istype == "file":
        files_list = [
            f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
        ]
    elif istype == "dir":
        files_list = [
            f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
        ]
    elif istype == "all":
        files_list = list(os.listdir(path))
    else:
        raise TypeError(f"Check availible values for istype. Now istype is {istype}")
    pathes = list(filter(pattern.match, files_list))
    return pathes


def get_kwarg(key, kwargs, default=None):
    """
    Function to get value from kwargs. If key is not present in kwargs,
    then default value is returned.

    Args:
        key: str
            Key to get value from kwargs
        kwargs: dict
            Dict with values
        default: any
            Default value to return if key is not present in kwargs

    Returns:
        Value from kwargs or default value.

    Example:
        >>> kwargs = {"key": "value"}
        >>> get_kwarg("key", kwargs)
        "value"
        >>> get_kwarg("key2", kwargs)
        None
        >>> get_kwarg("key2", kwargs, "default")
        "default"
    """
    if not kwargs.get(key) is None:
        return kwargs[key]
    return default


def list_to_dict(list_of_dicts: list) -> dict:
    """
    Merge list of dicts to one dict.

    Args:
        list_of_dicts: list
            List of dicts.

    Returns:
        new_dict
            Merged dict.

    Example:
        >>> list_of_dicts = [{"key": "value"}, {"key2": "value2"}]
        >>> list_to_dict(list_of_dicts)
        {"key": "value", "key2": "value2"}
    """
    new_dict = {}
    for dict_ in list_of_dicts:
        for key, value in dict_.items():
            if key not in new_dict:
                new_dict[key] = []
            new_dict[key].append(value)
    return new_dict


def concat_dictionaries(dicts: List[Dict[Any, List]]) -> Dict[Any, List]:
    """
    Concatenate a list of dictionaries with string keys and list values.

    Args:
        dicts: List of dictionaries to concatenate.

    Returns:
        Dict: A dictionary containing the concatenated lists.

    Examples:
        >>> dict1 = {"A": [1, 2, 3], "B": [4, 5, 6]}
        >>> dict2 = {"A": [7, 8], "C": [9, 10]}
        >>> dict3 = {"B": [11], "D": [12, 13]}

        >>> list_of_dicts = [dict1, dict2, dict3]
        >>> result = concat_dictionaries(list_of_dicts)
        >>> print(result)
    """
    concatenated_dict = {}

    for d in dicts:
        for key, value in d.items():
            if key in concatenated_dict:
                concatenated_dict[key].extend(value)
            else:
                concatenated_dict[key] = (
                    value.copy()
                )  # Use a copy to prevent modifying the original lists.

    return concatenated_dict


def transpose_dict(big_dict):
    """
    Transpose the values of a dictionary into a list of smaller dictionaries.

    Args:
    - big_dict (dict): a dictionary where each key maps to an iterable.

    Returns:
    - List[dict]: a list of smaller dictionaries.

    Examples:
        >>> big_dict_example = {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
            "c": [True, False, True]
        }

        >>> transpose_dict(big_dict_example)
    """
    num_items = len(next(iter(big_dict.values())))

    if not all(len(value) == num_items for value in big_dict.values()):
        raise ValueError("All values in the big_dict should have the same length.")

    return [
        {key: value[i] for key, value in big_dict.items()} for i in range(num_items)
    ]


def are_all_elements_similar(lst):
    """
    Checks if all elements in a list are the same.

    Args:
    lst (list): The input list to be checked.

    Returns:
    bool: True if all elements are the same, False otherwise.
    """
    # If the list is empty, there are no elements to compare, so return True
    if len(lst) == 0:
        return True

    # Get the first element in the list
    first_element = lst[0]

    # Compare all other elements to the first element
    for element in lst[1:]:
        if element != first_element:
            return False

    # If we've compared all elements and they are all the same, return True
    return True


def apply_windowing(
    image: np.ndarray, dicom: "dicomsdl.DataSet", index: int = 0
) -> np.ndarray:
    """
    Apply windowing to a DICOM image based on the specified Window Center and
    Window Width values along with the VOILUTFunction (if available).

    Args:
        image (np.ndarray): The image array to which the windowing should be applied.
        dicom ('pydicom.Dataset'): DICOM dataset containing the necessary tags.
        index (int): Index to specify which Window Center and Window Width values
                     to use in case they contain multiple values.

    Returns:
        np.ndarray: The resulting windowed image array.
    """
    if not dicom.WindowWidth and not dicom.WindowCenter:
        return image

    if dicom.PhotometricInterpretation not in ["MONOCHROME1", "MONOCHROME2"]:
        raise ValueError(
            "When performing a windowing operation only 'MONOCHROME1' and "
            "'MONOCHROME2' are allowed for (0028,0004) Photometric "
            "Interpretation"
        )

    # May be LINEAR (default), LINEAR_EXACT, SIGMOID or not present, VM 1
    voi_func = dicom.VOILUTFunction
    if voi_func is None:
        voi_func = "LINEAR"
    voi_func = str(voi_func).upper()
    # VR DS, VM 1-n
    center = dicom["WindowCenter"]
    if isinstance(center, list):
        center = center[index]

    width = dicom["WindowWidth"]
    if isinstance(width, list):
        width = width[index]

    # The output range depends on whether or not a modality LUT or rescale
    #   operation has been applied
    y_min: float
    y_max: float
    if dicom.ModalityLUTSequence:
        # Unsigned - see PS3.3 C.11.1.1.1
        y_min = 0
        item = dicom.ModalityLUTSequence[0]
        bit_depth = item.LUTDescriptor[2]
        y_max = 2**bit_depth - 1
    elif dicom.PixelRepresentation == 0:
        # Unsigned
        y_min = 0
        y_max = 2**dicom.BitsStored - 1
    else:
        # Signed
        y_min = -(2 ** (dicom.BitsStored - 1))
        y_max = 2 ** (dicom.BitsStored - 1) - 1

    slope = dicom.RescaleSlope
    intercept = dicom.RescaleIntercept
    if slope is not None and intercept is not None:
        # Otherwise its the actual data range
        y_min = y_min * dicom.RescaleSlope + dicom.RescaleIntercept
        y_max = y_max * dicom.RescaleSlope + dicom.RescaleIntercept

    y_range = y_max - y_min
    # image = image.astype("float64")

    if voi_func in ["LINEAR", "LINEAR_EXACT"]:
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == "LINEAR":
            if width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation"
                )
            center -= 0.5
            width -= 1
        elif width <= 0:
            raise ValueError(
                "The (0028,1051) Window Width must be greater than 0 "
                "for a 'LINEAR_EXACT' windowing operation"
            )

        below = image <= (center - width / 2)
        above = image > (center + width / 2)
        between = np.logical_and(~below, ~above)

        image[below] = y_min
        image[above] = y_max
        if between.any():
            image[between] = ((image[between] - center) / width + 0.5) * y_range + y_min
    elif voi_func == "SIGMOID":
        # PS3.3 C.11.2.1.3.1
        if width <= 0:
            raise ValueError(
                "The (0028,1051) Window Width must be greater than 0 "
                "for a 'SIGMOID' windowing operation"
            )

        image = y_range / (1 + np.exp(-4 * (image - center) / width)) + y_min
    else:
        raise ValueError(f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")

    return image


def data_dict_to_cpu(d: Dict) -> Dict:
    """Moves all tensors in the dictionary to CPU, leaving other types unchanged."""
    for key in d:
        if torch.is_tensor(d[key]):
            d[key] = d[key].detach().cpu()
    return d
