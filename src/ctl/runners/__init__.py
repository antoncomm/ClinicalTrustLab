"""All runners which are supported in ctl"""

__all__ = [
    "SimpleClassificationRunner",
    "AttackClassificationRunner",
    "MultiClassificationRunner",
    "AttackMultiClassificationRunner",
]

from .classification.multiclass import MultiClassificationRunner
from .classification.simple import SimpleClassificationRunner
from .classification.attack import AttackClassificationRunner, AttackMultiClassificationRunner
