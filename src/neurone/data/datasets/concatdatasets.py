"Concat dataset classes."

from typing import Tuple, List, Dict
from torch.utils.data import ConcatDataset

# neurone imports
from neurone.utils.general import concat_dictionaries


class BaseConcatDataset(ConcatDataset):
    """BaseConcatDataset can be used for any task:
    classification, detection, segmentation, etc...
    """

    def get_feature_values(self, names: List[str]) -> Dict[str, List]:
        """
        Retrieve feature values for specified feature names across multiple datasets.

        Args:
            names (List[str]): A list of feature names for which values are to be retrieved.
        """

        data_arr = []
        for dataset in self.datasets:
            data_arr.append(dataset.get_feature_values(names))
        res = concat_dictionaries(data_arr)
        return res

    def collate_fn(self, samples) -> Tuple:
        """Function to collate samples into a batch, using the first dataset's collate_fn"""
        return self.datasets[0].collate_fn(samples)
