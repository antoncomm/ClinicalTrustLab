import torch
from torch import nn
from torchvision.models.efficientnet import efficientnet_b3, EfficientNet_B3_Weights


class EfficientNet(nn.Module):
    """
    EfficientNet model
    """

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()

        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    def forward(self, batch):
        return self.model(batch)
