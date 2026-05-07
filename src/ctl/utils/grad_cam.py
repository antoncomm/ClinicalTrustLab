from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _model_on_cuda(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        return p.is_cuda
    return False


def get_gradcam(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_layers: Sequence[torch.nn.Module],
    *,
    targets: Optional[Sequence[object]] = None,
    eigen_smooth: bool = False,
    aug_smooth: bool = False,
) -> np.ndarray:
    """
    GradCAM++ for batched classification input.

    Returns:
      - Single-output logits (B,1): np.ndarray (B, H, W)
      - Multiclass logits (B, C):   np.ndarray (B, C, H, W)

    If `targets` is provided (len == B), returns np.ndarray (B, H, W).
    """
    model.eval()

    if x.dim() < 2:
        raise ValueError(f"Expected batched input, got shape={tuple(x.shape)}")

    # Grad-CAM needs gradients through the model
    if not x.requires_grad:
        x = x.requires_grad_(True)

    use_cuda = _model_on_cuda(model)

    with GradCAMPlusPlus(model=model, target_layers=list(target_layers)) as cam:
        # Explicit targets path: returns (B,H,W)
        if targets is not None:
            if len(targets) != x.shape[0]:
                raise ValueError("`targets` length must match batch size.")
            return cam(
                input_tensor=x,
                targets=list(targets),
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
            )

        # Infer number of outputs from a single forward
        with torch.enable_grad():
            logits = model(x)

        if logits.dim() != 2:
            raise ValueError(
                f"Expected logits shape (B,1) or (B,C). Got {tuple(logits.shape)}"
            )

        B, C = logits.shape

        # Single-output: CAM for the only output neuron => (B,H,W)
        if C == 1:
            inferred_targets = [ClassifierOutputTarget(0) for _ in range(B)]
            return cam(
                input_tensor=x,
                targets=inferred_targets,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
            )

        # Multiclass: CAM per class => (B,C,H,W)
        cams_per_class = []
        for cls in range(C):
            cls_targets = [ClassifierOutputTarget(cls) for _ in range(B)]
            cls_cam = cam(
                input_tensor=x,
                targets=cls_targets,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
            )  # (B,H,W)
            cams_per_class.append(cls_cam)

        return np.stack(cams_per_class, axis=1)  # (B,C,H,W)
