"""All runners which are supported in neurone"""

__all__ = ["SimpleClassificationRunner", "MultiClassificationRunner"]

from .classification.simple import SimpleClassificationRunner
from .classification.multiclass import MultiClassificationRunner
