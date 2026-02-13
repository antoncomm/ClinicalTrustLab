"""All models which are supported in neurone"""

__all__ = ["EfficientNet", "HFClassificationModel"]


from .classification.efficientnet import EfficientNet
from .classification.huggingface import HFClassificationModel
