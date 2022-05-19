"""

imutils/models/pl/classifier.py


Created on: Wednesday March 16th, 2022  
Created by: Jacob Alexander Rose  

"""


from icecream import ic
from rich import print as pp
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional, Callable
import hydra
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer

import numpy as np
import matplotlib.pyplot as plt
	
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from imutils.ml.utils.common import iterate_elements_in_batches, render_images
from imutils.ml.utils.metric_utils import get_scalar_metrics
from torchvision import models
# from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from imutils.ml.models.base import BaseModule, BaseLightningModule
from imutils.ml.models.backbones.backbone import build_model
from imutils.ml.utils.experiment_utils import resolve_config
# from imutils.ml.utils.model_utils import log_model_summary

from imutils.ml.utils.toolbox.nn.loss import LabelSmoothingLoss
from pytorch_lightning.utilities import rank_zero_only

# from imutils.ml import losses
# nn = losses.nn

__all__ = ["LitClassifier"]



class LitClassifier(BaseLightningModule): #pl.LightningModule):
	def __init__(self,
				 cfg: DictConfig=None,
				 model_cfg: DictConfig=None, 
				 name: str=None,
				 num_classes: int=None, 
				 loss_func: Union[Callable, str]=None,
				 sync_dist: bool=False,
				 log_images_freq: Optional[int]=None,
				 # pretrain : bool = True,
				 # self_supervised=False,
				 *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.save_hyperparameters(cfg, ignore=['loss_func'])
		self._setup(cfg=cfg,
				   model_cfg=model_cfg,
				   name=name,
				   num_classes=num_classes,
				   loss_func=loss_func,
				   sync_dist=sync_dist,
				   log_images_freq=log_images_freq,
				   *args, **kwargs)
		
	def _setup(self,
			  cfg: DictConfig=None,
			  model_cfg: DictConfig=None, 
			  name: str=None,
			  num_classes: int=None,
			  loss_func: Union[Callable, str]=None,
			  setup_backbone: bool=True,
			  setup_head: bool=True,
			  sync_dist: bool=False,
			  log_images_freq: Optional[int]=None,
			  *args, **kwargs) -> None:

		cfg = resolve_config(cfg)
		self.cfg = cfg
		model_cfg = cfg.get("model_cfg", {})
		self.model_cfg = model_cfg or {}
		self.lr = cfg.hp.lr
		self.batch_size = cfg.hp.batch_size
		self.num_classes = num_classes or self.model_cfg.head.get("num_classes")
		self.name = name or self.model_cfg.get("name")
		self.log_images_freq = log_images_freq
		
		self.sync_dist = sync_dist
		self.setup_loss(loss_func)
		self.setup_metrics()
		backbone = getattr(getattr(self, "net", None), "backbone", None)
		self.net = build_model(backbone_cfg=self.model_cfg.backbone,
							   head_cfg=self.model_cfg.head,
							   setup_backbone=setup_backbone,
							   setup_head=setup_head,
							   backbone=backbone)
		if self.cfg.train.freeze_backbone:
			self.freeze_up_to(layer=self.cfg.train.get("freeze_backbone_up_to"),
							  submodule="backbone",
							  verbose=False)	# def on_fit_start(self):

		
		if self.cfg.logging.log_model_summary:
			self.summarize_model(f"{self.name}/init")


	@classmethod
	def load_from_checkpoint(cls, 
							 checkpoint_path,
							 map_location=None,
							 cfg=None,
							 *args, **kwargs):
		model = super().load_from_checkpoint(checkpoint_path=checkpoint_path,
											 map_location=map_location)
											 # *args, **kwargs)
		
		model.source_cfg = model.cfg
		model.source_model_cfg = model.model_cfg
		if cfg is not None:
			model._setup(setup_backbone=False,
						cfg=cfg,
						*args, **kwargs)
		return model


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

	def step(self, batch, batch_idx: int=None) -> Dict[str, torch.Tensor]:
		"""
		TODO: remove the "x" from common method self.step() outputs & benchmark reduction in GPU memory leaks
		"""		
		if len(batch)>=3:
			x, y, metadata = batch[:3]
			image_idx = metadata.get("image_id")
		else:
			x, y = batch[:2]
			image_idx = torch.arange(0, len(x)) + batch_idx*self.batch_size


		logits = self(x)
		loss = self.loss(logits, y)
		return {"logits": logits, "loss": loss, "y": y, "x": x, "image_id": image_idx, "batch_idx": batch_idx}

	def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

		out = self.step(batch, batch_idx=batch_idx)
		return out #{k: out[k] for k in ["logits", "loss", "y", "batch_idx"]}

	def training_step_end(self, out):
		
		batch_size=self.batch_size #len(out["y"])
		loss = out["loss"].mean()
		
		self.train_metric.update(out["logits"], out["y"])
		
		self.log("train_loss", loss,
				 on_step=True, on_epoch=True,
				 prog_bar=True, batch_size=batch_size,
				 sync_dist=self.sync_dist
				)

		log_dict = {
			**self.train_metric
		}
		self.log_dict(
			log_dict,
			on_step=False, on_epoch=True,
			prog_bar=True, batch_size=batch_size
		)
		if isinstance(self.log_images_freq, int):
			if out["batch_idx"] % self.log_images_freq:
				self.render_image_predictions(
					outputs=out,
					batch_size=batch_size,
					n_elements_to_log=batch_size,
					log_name="train image predictions",
					logger=self.logger,
					global_step=self.trainer.global_step)

		return loss

	def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
		out = self.step(batch, batch_idx=batch_idx)
		return out #{k: out[k] for k in ["logits", "loss", "y", "batch_idx"]}
	
	def validation_step_end(self, out):
		batch_size=self.batch_size #len(out["y"])
		loss = out["loss"].mean()
		
		batch_idx = out.pop("batch_idx")

		self.val_metric.update(out["logits"], out["y"])
		log_dict = {
			"val_loss": loss,
			**self.val_metric
		}

		self.log_dict(log_dict,
					  on_step=False,
					  on_epoch=True,
					  prog_bar=True,
					  batch_size=batch_size)


	def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
		out = self.step(batch)
		return {k: out[k] for k in ["logits", "loss", "y"]}

	def test_step_end(self, out):
		batch_size=len(out["y"])
		self.test_metric(out["logits"], out["y"])
		self.log_dict(
			{
				"test_loss": out["loss"].mean(),
				**self.test_metric,
			},
			batch_size=batch_size
		)
		return {
			"y_true": out["y"],
			"logits": out["logits"],
			"val_loss": out["loss"].mean(),
		}

	def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
		if len(batch)==3:
			x, y, metadata = batch[:3]
			image_idx = metadata.get("image_id")
		else:
			x, y = batch[:2]
			image_idx = torch.arange(0, len(x)) + batch_idx*self.batch_size

		y_logit = self(x)
		return {"image_id":image_idx,
				"y_logit":y_logit}


	def configure_optimizers(
		self,
	) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
		"""
		Choose what optimizers and learning-rate schedulers to use in your optimization.
		Normally you'd need one. But in the case of GANs or similar you might have multiple.
		Return:
			Any of these 6 options.
			- Single optimizer.
			- List or Tuple - List of optimizers.
			- Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
			- Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
			  key whose value is a single LR scheduler or lr_dict.
			- Tuple of dictionaries as described, with an optional 'frequency' key.
			- None - Fit will run without any optimizer.
		"""
		# if hasattr(self.cfg.optim, "exclude_bn_bias") and \
				# self.cfg.optim.get("exclude_bn_bias", False):
		if self.cfg.optim.get("exclude_bn_bias", False):
			params = self.exclude_from_wt_decay(self.net,
												weight_decay=self.cfg.optim.optimizer.weight_decay)
		else:
			params = self.parameters()

		# pp(OmegaConf.to_container(self.cfg.optim.optimizer, resolve=True))
		opt = hydra.utils.instantiate(
			OmegaConf.to_container(self.cfg.optim.optimizer),
			params=params,
			weight_decay=self.cfg.optim.optimizer.weight_decay,
			_convert_="partial"
		)
		out = opt

		if self.cfg.optim.use_lr_scheduler:
			lr_scheduler = self.cfg.optim.lr_scheduler
			# pp(OmegaConf.to_container(lr_scheduler, resolve=True))
			scheduler = {"scheduler":hydra.utils.instantiate(lr_scheduler, optimizer=opt),
						 "name": "learning_rate",
						 "interval": "epoch",
						 "frequency":1}
			out = ([opt], [scheduler])
		return out

##############################
##############################

	# @staticmethod
	# @rank_zero_only
	def render_image_predictions(
		self,
		outputs: List[Any],
		batch_size: int,
		n_elements_to_log: int,
		log_name: str="image predictions",
		normalize_visualization: bool=True,
		logger=None,
		global_step: int=0,
		commit: bool=False
	) -> None:
		
		# images_feat_viz = []
		# integrated_gradients = IntegratedGradients(self.forward)
		# noise_tunnel = NoiseTunnel(integrated_gradients)
		
		classes = self.trainer.datamodule.train_dataset.classes
		
		
		images = []
		for output_element in iterate_elements_in_batches(
			outputs, batch_size, n_elements_to_log
		):  
			rendered_image = render_images(
				output_element["x"],
				autoshow=False,
				normalize=normalize_visualization)
			y_pred = classes[output_element['logits'].argmax()]
			y_true = classes[int(output_element['y'])]
			caption = f"y_pred: {y_pred}  [gt: {y_true}]"  # noqa	
			# attributions_ig_nt = noise_tunnel.attribute(output_element["image"].unsqueeze(0), nt_samples=50,
														# nt_type='smoothgrad_sq', target=output_element["y_true"],
														# internal_batch_size=50)
			images.append(
				wandb.Image(
					rendered_image,
					caption=caption,
				)
			)
		if logger is not None:
			logger.experiment.log(
				{log_name: images,
				 "global_step":global_step},
				commit=commit
			)


	# @rank_zero_only
	def render_image_predictions_table(
		self,
		outputs: List[Any],
		batch_size: int,
		n_elements_to_log: int,
		log_name: str="image_predictions",
		log_type: str="predictions_table",
		normalize_visualization: bool=True,
		logger=None,
		log_as_artifact: bool=False,
		global_step: int=0,
		commit: bool=False
	) -> None:
		
		if logger is None:
			return
		
		columns = ["image_id", "image", "y_pred", "y_true"]
		table = self._get_table(log_name=log_name, log_type=log_type, columns=columns)

		
		for output_element in iterate_elements_in_batches(
			outputs, batch_size, n_elements_to_log
		):  
			rendered_image = render_images(
				output_element["x"],
				autoshow=False,
				normalize=normalize_visualization)
			y_pred = output_element['logits'].argmax()
			y_true = output_element['y']
			image_id = output_element['image_id']
			img = wandb.Image(rendered_image)
			row = [image_id, img, y_pred, y_true]
			table.add_data(*row)
		
		if logger is not None:
			if log_as_artifact:
				artifact = self._get_artifact(log_name, log_type)
				if artifact is not None:
					artifact.add(table, "predictions")
				rank_zero_only(logger.experiment.log_artifact)(artifact)
				if log_name in self.artifacts.get(log_type, {}):
					self.artifacts[log_type][log_name] = None
			else:
				logger.experiment.log(
					{log_name: table,
					 "global_step":global_step},
					commit=commit
				)

	def _get_table(self, log_name, log_type, columns=None):
		
		table=None
		if log_type not in self.tables:
			self.tables[log_type] = {}
		if log_name in self.tables[log_type]:
			table = self.tables[log_type][log_name]
		if table is None:
			table = wandb.Table(columns=columns)
			self.tables[log_type][log_name] = table
		return table

	@rank_zero_only
	def _get_artifact(self, log_name, log_type):
		artifact=None
		if log_type not in self.artifacts:
			self.artifacts[log_type] = {}
		if log_name in self.artifacts[log_type]:
			artifact = self.artifacts[log_type][log_name]
		if artifact is None:
			artifact = wandb.Artifact(log_name, type=log_type)
			self.artifacts[log_type][log_name] = artifact
		return artifact


###########################
	
# [TODO] Uncomment & benchmark this
# source: https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#set-grads-to-none
	# def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
	#	 optimizer.zero_grad(set_to_none=True)