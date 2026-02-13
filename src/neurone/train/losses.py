import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to float32
        targets = targets.to(torch.float32)

        # Add an extra dimension to targets if shapes don't match for binary classification
        if inputs.shape != targets.shape:
            targets = targets.unsqueeze(-1)

        # Compute focal loss
        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class GaussianFocalLoss(nn.Module):
    """
    Focal loss for heatmaps.

    Parameters
    ----------
    alpha: float
        A balanced form for Focal loss.
    gamma: float
        The gamma for calculating the modulating factor.
    loss_weight:
        The weight for the Focal loss.
    """

    def __init__(self, alpha=2.0, gamma=4.0, loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.gamma)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return self.loss_weight * loss
