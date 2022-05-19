"""

models/pl/boring.py


Defines a BoringModel for testing purposes

source: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/bug_report/bug_report_model.py


Updated on: Sunday April 24th, 2022  
Updated by: Jacob A Rose


"""






import os

import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from typing import *

from imutils.ml.models.base import BaseLightningModule
from imutils.ml.utils.experiment_utils import resolve_config


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(BaseLightningModule):
	
    def __init__(self
				 cfg: DictConfig=None,
				 num_classes: int=None, 
				 loss_func: Union[Callable, str]=None,
				 sync_dist: bool=False,
				 *args, **kwargs):
        super().__init__(*args, **kwargs)
		
		self._setup(cfg=cfg,
				   num_classes=num_classes,
				   loss_func=loss_func,
				   sync_dist=sync_dist,
				   *args, **kwargs)
		
	def _setup(self,
			  cfg: DictConfig=None,
			  num_classes: int=None,
			  loss_func: Union[Callable, str]=None,
			  sync_dist: bool=False,
			  *args, **kwargs) -> None:

		cfg = resolve_config(cfg)
		self.cfg = cfg
		model_cfg = cfg.get("model_cfg", {})
		self.model_cfg = model_cfg or {}
		self.lr = cfg.hp.lr
		self.batch_size = cfg.hp.batch_size
		self.num_classes = num_classes or self.model_cfg.head.get("num_classes")
		
		self.sync_dist = sync_dist
		self.setup_loss(loss_func)
		self.setup_metrics()
		
		self.layer = torch.nn.Linear(32, self.num_classes)
		
		

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss,
				 sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss,
				 sync_dist=self.sync_dist)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss,
				 sync_dist=self.sync_dist)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=False,
    )

    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)


if __name__ == "__main__":
    run()
