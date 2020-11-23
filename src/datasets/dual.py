# coding: utf-8
__author__ = "paantya: https://kaggle.com/paantya"


from typing import Tuple, Union, Dict

import torch
import numpy as np
from torch.utils.data import Dataset


class MoADatasetDual(Dataset):
    def __init__(
            self,
            data,
            targets=None,
            targets1=None,
            mode='train'
    ):
        """

        Args:
        """

        self.mode = mode
        self.data = data
        self.targets = targets
        self.targets1 = targets1

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        data = self.data[idx]
        if self.targets is not None:
            target = self.targets[idx]
            target1 = self.targets1[idx]
        else:
            target = np.zeros((206,))
            target1 = np.zeros((402,))

        sample = {'data': torch.tensor(data).float(),
                  'target': torch.tensor(target).float(),
                  'target1': torch.tensor(target1).float()
                  }

        return sample

    def __len__(self) -> int:
        return len(self.data)