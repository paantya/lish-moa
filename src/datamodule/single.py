# coding: utf-8
__author__ = "paantya: https://kaggle.com/paantya"


from typing import Tuple, Union, Dict

import torch
import numpy as np
import pytorch_lightning as pl
from src.datasets.single import MoADatasetSingle


class MoADataModuleSingle(pl.LightningDataModule):
    def __init__(self, hparams: Dict,
                 train_data, train_targets,
                 valid_data, valid_targets,
                 batch_size=128,
                 num_workers=0,
                 shuffle=False,
                 ):
        super().__init__()
        self.hparams = hparams
        self.train_data = train_data
        self.train_targets = train_targets
        self.valid_data = valid_data
        self.valid_targets = valid_targets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = MoADatasetSingle(data=self.train_data.iloc[:, 1:].values,
                                        targets=self.train_targets.iloc[:, 1:].values)
        self.valid_dataset = MoADatasetSingle(data=self.valid_data.iloc[:, 1:].values,
                                        targets=self.valid_targets.iloc[:, 1:].values)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return valid_loader

    def test_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return valid_loader