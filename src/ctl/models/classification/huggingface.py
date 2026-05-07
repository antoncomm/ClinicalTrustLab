"""
A wrapper class for Hugging Face classification models.
"""

import torch
from torch import nn
import transformers


class HFClassificationModel(nn.Module):
    """
    A wrapper class for any classification model from Hugging Face,
    compatible with `SimpleClassificationRunner`.
    """

    def __init__(self, class_name, **kwargs) -> None:
        """
        Initializes the Hugging Face classification model wrapper.
        `ignore_mismatched_sizes` argument is set to True by default.

        Parameters
        ----------
        class_name : str
            The name of the HF classification model class.
        **kwargs : dict
            Additional keyword arguments for `from_pretrained` HF method.
        """
        super().__init__()
        kwargs.setdefault("ignore_mismatched_sizes", True)
        self.model = getattr(transformers, class_name).from_pretrained(**kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass through the model.

        Parameters
        ----------
        inputs : torch.Tensor (B, C, H, W)
            Batch of images.

        Returns
        -------
        torch.Tensor (B, num_classes)
            Logits without any activation function applied.
        """
        return self.model(pixel_values=inputs).logits
