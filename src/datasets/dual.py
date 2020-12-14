# coding: utf-8
__author__ = "paantya: https://kaggle.com/paantya"


from typing import Tuple, Union, Dict

import torch
import numpy as np
from torch.utils.data import Dataset


class MoADatasetDual(Dataset):
    def __init__(
            self,
            x,
            y=None,
            y1=None,
            mode='train',
    ):
        """

        Args:
        """

        self.mode = mode
        self.data = x
        self.targets = y
        self.targets1 = y1

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        data = self.data[idx]
        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = np.zeros((206,))
        if self.targets1 is not None:
            target1 = self.targets1[idx]
        else:
            target1 = np.zeros((402,))

        sample = {'x': torch.tensor(data).float(),
                  'y': torch.tensor(target).float(),
                  'y1': torch.tensor(target1).float()
                  }

        return sample

    def __len__(self) -> int:
        return len(self.data)