"""

models/lightning.py

"""


import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np

from argparse import ArgumentParser
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from imutils.ml.aug.loss import LabelSmoothingLoss
from imutils.ml.aug.image.masks import GridMask

from imutils.ml.models.vision_transformer import VisionTransformer

"""
Pour l'entrainement de notre modele on utilise pytorch-lightning pour faciliter l'implimentation et optimisation des fonctions d'entrainements.
"""

import torch.nn as nn
import torchvision.models as models
# from conf import *

def build_model():
    if args.model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

    elif args.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)

    #Modify the classifier for agriculture data
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs,512),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                            nn.Linear(512,4))
    
    if args.channels_last:
        model = model.to(args.device, memory_format=torch.channels_last)
    else:
        model = model.to(args.device)
        
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    return model




import pytorch_lightning as pl
import torch
import torchmetrics


class LoggedLitModule(pl.LightningModule):
    """LightningModule plus wandb features and simple training/val steps.
    By default, assumes that your training loop involves inputs (xs)
    fed to .forward to produce outputs (y_hats)
    that are compared to targets (ys)
    by self.loss and by metrics,
    where each batch == (xs, ys).
    This loss is fed to self.optimizer.
    If this is not true, overwrite _train_forward
    and optionally _val_forward and _test_forward.
    """

    def __init__(self,
                 criterion):
        super().__init__()

        self.loss = criterion
        self.train_metrics = torch.nn.ModuleList([])
        self.val_metrics = torch.nn.ModuleList([])
        self.test_metrics = torch.nn.ModuleList([])

    def training_step(self, xys, idx):
        xs, ys = xys
        y_hats = self._train_forward(xs)
        loss = self.loss(y_hats, ys)

        logging_scalars = {"loss": loss}
        for metric in self.training_metrics:
            self.log_metric(metric, logging_scalars, y_hats, ys)

        self.do_logging(xs, ys, idx, y_hats, logging_scalars)

        return {"loss": loss, "y_hats": y_hats}

    def validation_step(self, xys, idx):
        xs, ys = xys
        y_hats = self._val_forward(xs)
        loss = self.loss(y_hats, ys)

        logging_scalars = {"loss": loss}
        for metric in self.validation_metrics:
            self.log_metric(metric, logging_scalars, y_hats, ys)

        self.do_logging(xs, ys, idx, y_hats, logging_scalars, step="val")

        return {"loss": loss, "y_hats": y_hats}

    def test_step(self, xys, idx):
        xs, ys = xys
        y_hats = self._test_forward(xs)
        loss = self.loss(y_hats, ys)

        logging_scalars = {"loss": loss}
        for metric in self.test_metrics:
            self.log_metric(metric, logging_scalars, y_hats, ys)

        self.do_logging(xs, ys, idx, y_hats, logging_scalars, step="test")

        return {"loss": loss, "y_hats": y_hats}

    def do_logging(self, xs, ys, idx, y_hats, scalars, step="train"):
        self.log_dict(
            {step + "/" + name: value for name, value in scalars.items()})

    def on_pretrain_routine_start(self):
        print(self)

    # def log_metric(self, metric, logging_scalars, y_hats, ys):
    #     metric_str = metric.__class__.__name__.lower()
    #     value = metric(y_hats, ys)
    #     logging_scalars[metric_str] = value

    def _train_forward(self, xs):
        """Overwrite this method when module.forward doesn't produce y_hats."""
        return self.forward(xs)

    def _val_forward(self, xs):
        """Overwrite this method when training and val forward differ."""
        return self._train_forward(xs)

    def _test_forward(self, xs):
        """Overwrite this method when val and test forward differ."""
        return self._val_forward(xs)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_params)

    def optimizer(self, *args, **kwargs):
        error_msg = ("To use LoggedLitModule, you must set self.optimizer to a torch-style Optimizer"
                     + "and set self.optimizer_params to a dictionary of keyword arguments.")
        raise NotImplementedError(error_msg)


		
		
#################################################

class LitClassifierModel(LoggedLitModule): #pl.LightningModule):

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.criterion = LabelSmoothingLoss(kwargs['num_classes'], smoothing=0.1)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits, y)
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=False)
        self.log('train_acc', acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits, y)
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits, y)
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        return {'test_loss': loss, 'test_acc': acc}


    def test_epoch_end(self, outputs):
        # for test_epoch_end showcase, we record batch mse and batch std

        loss, acc = zip(*[(d['test_loss'], d['test_acc']) for d in outputs])

        avg_loss, avg_acc = np.mean(loss), np.mean(acc)
        std_loss, std_acc = np.std(loss), np.std(acc)

        result = {
            'loss': {'avg': avg_loss, 'std': std_loss},
            'acc': {'avg': avg_acc, 'std': std_acc}
            }
        self.log_dict(result)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                min_lr=1e-8,
                verbose=True
            ),
            'interval': 'step',
            'monitor': 'train_loss'
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--patch_size', default=4, type=int)
        parser.add_argument('--emb_dim', default=128, type=int)
        parser.add_argument('--mlp_dim', default=256, type=int)
        parser.add_argument('--num_heads', default=24, type=int)
        parser.add_argument('--num_layers', default=24, type=int)
        parser.add_argument('--attn_dropout_rate', default=0.0, type=float)
        parser.add_argument('--dropout_rate', default=.1, type=float)
        parser.add_argument('--resnet', action="store_true")
        return parser


class CIFAR10DataModule(pl.LightningDataModule):
    """CIFAR 10 DATASET
    """
    def __init__(self, data_dir: str = './', image_size: int = 512, batch_size: int = 128, num_workers: int = 12, val_size: float = 0.2, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size

        self.dims = (3, image_size, image_size)

    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            val_size = int(50000 * self.val_size)
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [50000 - val_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class CIFAR100DataModule(pl.LightningDataModule):
    """CIFAR 100 DATASET
    """

    def __init__(self, data_dir: str = './', image_size: int = 512, batch_size: int = 128, num_workers: int = 12, val_size: float = 0.2, **kwargs):
        super().__init__()
        self.data_dir = data_dir

        self.transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[n/255.
                    for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
        ])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size

        self.dims = (3, image_size, image_size)

    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            val_size = int(50000 * self.val_size)
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [50000 - val_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def load_from_checkpoint(model_class:type, lit_model:type, hparams_file: str, checkpoint_file:str):
    hparams = pl.core.saving.load_hparams_from_yaml(hparams_file)
    backbone = model_class(**hparams)
    model = lit_model.load_from_checkpoint(
                checkpoint_file,
                backbone=backbone
            )
    return model









