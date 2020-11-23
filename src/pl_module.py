# coding: utf-8
__author__ = "paantya: https://kaggle.com/paantya"

import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from hydra.utils import instantiate
from torch.utils.data import DataLoader

# from src.models.networks.linear import EncoderLinear, DecoderLinear
# from src.losses.xtanh import XTanhLoss
# from src.losses.r2_score import R2Loss
# from src.losses.log_cosh import LogCoshLoss
# from src.losses.xsigmoid import XSigmoidLoss
#


class LishMoaPL(pl.LightningModule):
    def __init__(self, hparams, model, dual=False):
        super().__init__()
        self.hparams = hparams
        self.net = model
        self.dual = dual
        self.lr = 0.1 if self.hparams.lr == 'auto' else self.hparams.lr
        self.batch_size = 128 if self.hparams.batch_size in ['auto', 'power', 'binsearch'] else self.hparams.batch_size

        self.example_input_array = torch.zeros(self.batch_size, self.hparams.model.input_height)# optional

        self.criterion = self.get_criterion()

    def forward(self, x, targets, targets1, *args, **kwargs):
        if not self.dual:
            return self.net(x, targets)
        else:
            return self.net(x, targets, targets1)


    def training_step(self, batch, batch_idx: int) -> dict:
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.criterion(x_hat, x)

        # Logging to TensorBoard by default
        self.log('_loss/train', loss, logger=True)
        return loss

    def show_img(self, model, n=10, title="run", loss=0):
        transform = torchvision.transforms.Compose(
            [instantiate(trnsfrm) for trnsfrm in self.hparams.transform.transforms[:2]])
        dataset = instantiate(self.hparams.dataset, train=False, transform=transform)
        #
        # dataset = instantiate(self.hparams.dataset, train=False)
        data_loader = instantiate(self.hparams.dataloader.val, dataset=dataset, batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.dataloader.num_workers)
        for images, _ in data_loader:
            # print('images.shape:', images.shape)
            plt.figure(figsize=(n, 3))
            plt.axis('off')

            y_hot = model.decoder(model.encoder(torch.reshape(images[:n], shape=(-1, 1, 1, 784))))
            img = torch.reshape(y_hot, shape=(-1, 1, 28, 28)).detach()
            imgf = torch.cat((images[:n], img), 0)

            plt.imshow(make_grid(imgf, nrow=n).permute((1, 2, 0)))
            plt.title(f"{title}")
            plt.show()
            return 0

    def validation_step(self, batch, batch_idx: int) -> dict:
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.criterion(x_hat, x)
        # if self.fplot and self.current_epoch % 10 == 0:
        #     self.show_img(self, n=10, title=f"epoch {self.current_epoch}", loss=loss)
        #     self.fplot = False
        self.log('_l1/val', self.l1_loss(x_hat, x), on_epoch=True, logger=True)
        self.log('_l2/val', self.l2_loss(x_hat, x), on_epoch=True, logger=True)
        return {
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs: torch.tensor):
        self.fplot = True
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # avg_loss_mae = torch.stack([x["val_mae"] for x in outputs]).mean()
        # avg_loss_mse = torch.stack([x["val_mse"] for x in outputs]).mean()

        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # "log": {
        #     f"val/avg_{self.hparams.criterion}": avg_loss,
        #     # f"val/avg_mae": avg_loss_mae,
        #     # f"val/avg_mse": avg_loss_mse,
        # }
        self.log('_loss/val', avg_loss)
        self.log('epoch/lr', self.optimizer.param_groups[0]['lr'])

    def test_step(self, batch, batch_idx: int) -> dict:
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        self.log('_l1/test', self.l1_loss(x_hat, x), on_epoch=True, logger=True)
        self.log('_l2/test', self.l2_loss(x_hat, x), on_epoch=True, logger=True)
        return {
            "test_loss": self.criterion(x_hat, x),
            # "val_mae": nn.L1Loss(y_hat, x),
            # "val_mse": nn.MSELoss(y_hat, x),
        }

    def test_epoch_end(self, outputs: torch.tensor) -> dict:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        # avg_loss_mae = torch.stack([x["val_mae"] for x in outputs]).mean()
        # avg_loss_mse = torch.stack([x["val_mse"] for x in outputs]).mean()

        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # "log": {
        #     f"val/avg_{self.hparams.criterion}": avg_loss,
        #     # f"val/avg_mae": avg_loss_mae,
        #     # f"val/avg_mse": avg_loss_mse,
        # }

        if self.current_epoch > 0:
            self.logger.log_metrics({'hp_metric': avg_loss}, 0)
        self.log('_loss/test', avg_loss)
        return avg_loss

    def configure_optimizers(self):
        self.optimizer = self.get_optimizer()
        self.scheduler = {'scheduler': self.get_scheduler(self.optimizer), 'interval': 'epoch', 'monitor': '_loss/val'}

        return [self.optimizer], [self.scheduler]

    # fabric

    def get_nets(self):
        encoder = instantiate(self.hparams.encoder)
        decoder = instantiate(self.hparams.decoder)
        return encoder, decoder

    def get_criterion(self):
        criterion = instantiate(self.hparams.criterion)
        return criterion
        # dict_criterion = {
        #     "mae":      nn.L1Loss,
        #     "l1":       nn.L1Loss,
        #
        #     "mse":      nn.MSELoss,
        #     "l2":       nn.MSELoss,
        #     "r2":       R2Loss,
        #
        #     "sl1":      nn.SmoothL1Loss,
        #     "huber":    nn.SmoothL1Loss,
        #
        #     "lcl":      LogCoshLoss,
        #     "xtl":      XTanhLoss,
        #     "xsl":      XSigmoidLoss,
        #
        #
        #     # "CE": nn.CrossEntropyLoss,
        #     # "BCE" : nn.BCELoss,
        #     # "BCElog" : nn.BCEWithLogitsLoss,
        #
        #     # "KL" : nn.KLDivLoss,
        #     # "nll" : nn.NLLLoss,
        # }
        # if self.hparams.criterion.name in dict_criterion:
        #     print(f"self.hparams.criterion.name: {self.hparams.criterion.name}")
        #     # return dict_criterion[self.hparams.criterion.name]
        # else:
        #     nl = '\n'
        #     raise NotImplementedError(
        #         f"Not a valid criterion configuration.{nl}"
        #         f"Criterion {self.hparams.criterion} in list criterions ({dict_criterion.keys()}): "
        #         f"{self.hparams.criterion in dict_criterion.keys()}{nl}"
        #     )

    def get_optimizer(self) -> object:
        optimizer = instantiate(self.hparams.optimizer, params=self.net.parameters(), lr=self.lr)
        return optimizer

    def get_scheduler(self, optimizer) -> object:
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
        return scheduler

    def train_dataloader(self):
        # REQUIRED

        transform = torchvision.transforms.Compose(
            [instantiate(trnsfrm) for trnsfrm in self.hparams.transform.transforms])
        dataset = instantiate(self.hparams.dataset, train=True, transform=transform)
        # dataset = instantiate(self.hparams.dataset, train=True)
        data_loader = instantiate(self.hparams.dataloader.train, dataset=dataset, batch_size=self.batch_size,
                                  num_workers=self.hparams.dataloader.num_workers)
        return data_loader

    def val_dataloader(self):
        # OPTIONAL
        transform = torchvision.transforms.Compose(
            [instantiate(trnsfrm) for trnsfrm in self.hparams.transform.transforms])
        dataset = instantiate(self.hparams.dataset, train=False, transform=transform)
        #
        # dataset = instantiate(self.hparams.dataset, train=False)
        data_loader = instantiate(self.hparams.dataloader.val, dataset=dataset, batch_size=self.batch_size,
                                  num_workers=self.hparams.dataloader.num_workers)
        return data_loader
        # DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
        #                   batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        # OPTIONAL

        transform = torchvision.transforms.Compose(
            [instantiate(trnsfrm) for trnsfrm in self.hparams.transform.transforms])
        dataset = instantiate(self.hparams.dataset, train=False, transform=transform)

        # dataset = instantiate(self.hparams.dataset, train=False)
        data_loader = instantiate(self.hparams.dataloader.test, dataset=dataset, batch_size=self.batch_size,
                                  num_workers=self.hparams.dataloader.num_workers)
        return data_loader