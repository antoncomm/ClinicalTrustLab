"Classification datasets"

from .basedatasets import (
    BaseClassificationDataset,
    BaseMammographyDataset,
)


class MammographyClassification(BaseClassificationDataset, BaseMammographyDataset):
    """
    Classification dataset for Mammography
    """
