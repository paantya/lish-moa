# coding: utf-8
__author__ = "paantya: https://kaggle.com/paantya"
__forkfrom__ = "tiulpin: https://kaggle.com/tiulpin"


from typing import Tuple, Union, Dict

import torch
import numpy as np
from torch.utils.data import Dataset


class MoADatasetSingle(Dataset):
    def __init__(
            self,
            data,
            targets=None,
            mode='train'
    ):
        """

        Args:
        """

        self.mode = mode
        self.data = data
        self.targets = targets

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        data = self.data[idx]
        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = np.zeros((206,))

        sample = {'data': torch.tensor(data).float(),
                  'target': torch.tensor(target).float()
                  }

        return sample

    def __len__(self) -> int:
        return len(self.data)