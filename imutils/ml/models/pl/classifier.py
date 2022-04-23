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
				   *args, **kwargs)
		
	def _setup(self,
			  cfg: DictConfig=None,
			  model_cfg: DictConfig=None, 
			  name: str=None,
			  num_classes: int=None,
			  loss_func: Union[Callable, str]=None,
			  setup_backbone: bool=True,
			  setup_head: bool=True,
			  *args, **kwargs) -> None:

		cfg = resolve_config(cfg)
		self.cfg = cfg
		model_cfg = cfg.get("model_cfg", {})
		self.model_cfg = model_cfg or {}
		self.lr = cfg.hp.lr
		self.batch_size = cfg.hp.batch_size
		self.num_classes = num_classes or self.model_cfg.head.get("num_classes")
		self.name = name or self.model_cfg.get("name")
		
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
							 


	def setup_loss(self,
				   loss_func: Optional[Union[Callable, str]]=None):
		if isinstance(loss_func, Callable):
			self.loss = loss_func
		else:
			self.loss = hydra.utils.instantiate(self.model_cfg.loss)


	def setup_metrics(self):
		
		self.train_metric = get_scalar_metrics(num_classes=self.num_classes,
											   average="macro",
											   prefix="train")
		self.val_metric = get_scalar_metrics(num_classes=self.num_classes,
											   average="macro",
											   prefix="val")
		self.test_metric = get_scalar_metrics(num_classes=self.num_classes,
											   average="macro",
											   prefix="test")
		
		self.artifacts = {}
		self.tables = {}

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

	def step(self, batch) -> Dict[str, torch.Tensor]:
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
		return {"logits": logits, "loss": loss, "y": y, "x": x, "image_id": image_idx}

	def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

		out = self.step(batch)
		out["batch_idx"] = batch_idx
		return out #{k: out[k] for k in ["logits", "loss", "y", "batch_idx"]}

	def training_step_end(self, out):
		self.train_metric(out["logits"], out["y"])
		
		batch_size=self.batch_size #len(out["y"])
		loss = out["loss"].mean()
		batch_idx = out.pop("batch_idx")
		if batch_idx <= self.cfg.logging.max_batches_to_log:
			# print("Running: self.render_image_predictions_table")
			self.render_image_predictions_table(
				outputs=out,
				batch_size=batch_size, #self.cfg.data.datamodule.batch_size,
				n_elements_to_log=self.cfg.logging.n_elements_to_log,
				log_name=f"train_batch",
				log_type="predictions_table",
				normalize_visualization=self.cfg.logging.normalize_visualization,
				logger=self.logger,
				global_step=self.global_step,
				commit=False)

			# self.render_image_predictions(
			# 	outputs=out,
			# 	batch_size=batch_size, #self.cfg.data.datamodule.batch_size,
			# 	n_elements_to_log=self.cfg.logging.n_elements_to_log,
			# 	log_name="train_image_predictions",
			# 	normalize_visualization=self.cfg.logging.normalize_visualization,
			# 	logger=self.logger,
			# 	global_step=self.global_step,
			# 	commit=False)
		
		log_dict = {
			"train_loss": loss,
			**self.train_metric
		}
		self.log_dict(
			log_dict,
			on_step=True,
			on_epoch=True,
			prog_bar=True,
			batch_size=batch_size
		)
		# self.print(f"training_step_end -> self.current_epoch: {self.current_epoch}, self.global_step: {self.global_step}, loss: {loss}")
		return loss

	def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
		out = self.step(batch)
		out["batch_idx"] = batch_idx
		return out #{k: out[k] for k in ["logits", "loss", "y", "batch_idx"]}
	
	def validation_step_end(self, out):
		self.val_metric(out["logits"], out["y"])
		batch_size=self.batch_size #len(out["y"])
		loss = out["loss"].mean()
		
		batch_idx = out.pop("batch_idx")
		if batch_idx <= self.cfg.logging.max_batches_to_log:
			# print("Running: self.render_image_predictions_table")
			self.render_image_predictions_table(
				outputs=out,
				batch_size=batch_size, #self.cfg.data.datamodule.batch_size,
				n_elements_to_log=self.cfg.logging.n_elements_to_log,
				log_name=f"val_batch",
				log_type="predictions_table",
				normalize_visualization=self.cfg.logging.normalize_visualization,
				logger=self.logger,
				global_step=self.global_step,
				commit=False)		
			
			
			# self.render_image_predictions(
			# 	outputs=out,
			# 	batch_size=batch_size, #self.cfg.data.datamodule.batch_size,
			# 	n_elements_to_log=self.cfg.logging.n_elements_to_log,
			# 	log_name="val_image_predictions",
			# 	normalize_visualization=self.cfg.logging.normalize_visualization,
			# 	logger=self.logger,
			# 	global_step=self.global_step,
			# 	commit=False)

		log_dict = {
			"val_loss": loss,
			**self.val_metric
		}

		self.log_dict(log_dict,
					  on_step=True, # False, #
					  on_epoch=True,
					  prog_bar=True,
					  batch_size=batch_size)
		# self.print(f"validation_step_end -> self.current_epoch: {self.current_epoch}, self.global_step: {self.global_step}, loss: {loss}")


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
			# "image": out["x"],
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

# 	def training_epoch_end(self, outputs: List[Any]) -> None:
# 		"""
		
# 		"""
		
# 		sch = self.lr_schedulers()
# 		sch.step()
		
		# info = {k: v.shape for k,v in outputs[0].items() if hasattr(v, "shape") else v}
		# print("self.training_epoch_end: ", f"device:{torch.cuda.current_device()}, len(outputs)={len(outputs)}, info: {info}")

		# losses = torch.stack([o["loss"] for o in outputs])
		# self.print(f"training_epoch_end -> self.current_epoch: {self.current_epoch}, self.global_step: {self.global_step}, losses: {losses}")
		# print(f"self.validation_epoch_end (torch.stack the losses): losses.shape = {losses.shape}")
		


# 	def validation_epoch_end(self, outputs: List[Any]) -> None:
# 		"""
		
# 		"""
# 		info = {k: v.shape for k,v in outputs[0].items()}
		# print("self.validation_epoch_end: ", f"device:{torch.cuda.current_device()}, len(outputs)={len(outputs)}, info: {info}")
		
		# losses = torch.stack([o["loss"] for o in outputs])
		# losses = []
		# y = []
		# logits = []
		# for o in outputs:
		# 	losses.append(o["loss"])
		# 	y.append(o["y"])
		# 	logits = [o["logits"]]
		# losses = torch.stack(losses)
		# y = torch.cat(y)
		# logits = torch.cat(logits)
		
		# self.print(f"validation_epoch_end -> self.current_epoch: {self.current_epoch}, self.global_step: {self.global_step}, losses: {losses}")


# 	def test_epoch_end(self, outputs: List[Any]) -> None:
# 		if "image" not in outputs:
# 			# print(f"Skipping test render_image_predictions due to missing 'image' key in epoch outputs.")
# 			return
		
# 		self.render_image_predictions(
# 			outputs=outputs,
# 			batch_size=self.cfg.data.datamodule.batch_size,
# 			n_elements_to_log=self.cfg.logging.n_elements_to_log,
# 			log_name="test_image_predictions",
# 			normalize_visualization=self.cfg.logging.normalize_visualization,
# 			logger=self.logger,
# 			global_step=self.global_step,
# 			commit=False)

	@staticmethod
	@rank_zero_only
	def render_image_predictions(
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
		
		images = []
		for output_element in iterate_elements_in_batches(
			outputs, batch_size, n_elements_to_log
		):  
			rendered_image = render_images(
				output_element["x"],
				autoshow=False,
				normalize=normalize_visualization)
			caption = f"y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y']}]"  # noqa	
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
			params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.cfg.optim.optimizer.weight_decay)
		else:
			params = self.parameters()

		# pp(OmegaConf.to_container(self.cfg.optim.optimizer))
		pp(OmegaConf.to_container(self.cfg.optim.optimizer, resolve=True))
		opt = hydra.utils.instantiate(
			OmegaConf.to_container(self.cfg.optim.optimizer),
			params=params,
			weight_decay=self.cfg.optim.optimizer.weight_decay,
			_convert_="partial"
		)

		out = opt
		if self.cfg.optim.use_lr_scheduler:
			lr_scheduler = self.cfg.optim.lr_scheduler
			pp(OmegaConf.to_container(lr_scheduler, resolve=True))
			scheduler = {"scheduler":hydra.utils.instantiate(lr_scheduler, optimizer=opt),
						 "name": "learning_rate",
						 "interval": "epoch",
						 "frequency":1}
			out = ([opt], [scheduler])
		return out

	@staticmethod
	def exclude_from_wt_decay(named_params: List[Tuple[str, torch.Tensor]],
							  weight_decay: float,
							  skip_list: Tuple[str]=("bias", "bn")
							 ) -> List[Dict[str, Any]]:
		"""
		Sort named_params into 2 groups: included & excluded from weight decay.
		Includes any params with a name that doesn't match any pattern in `skip_list`.
		
		Arguments:
			named_params: List[Tuple[str, torch.Tensor]]
			weight_decay: float,
			skip_list: Tuple[str]=("bias", "bn")):		
		"""
		params = []
		excluded_params = []

		for name, param in named_params:
			if not param.requires_grad:
				continue
			elif any(layer_name in name for layer_name in skip_list):
				excluded_params.append(param)
			else:
				params.append(param)

		return [
			{"params": params, "weight_decay": weight_decay},
			{
				"params": excluded_params,
				"weight_decay": 0.0,
			},
		]
	
# [TODO] Uncomment & benchmark this
# source: https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#set-grads-to-none
	# def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
	#	 optimizer.zero_grad(set_to_none=True)
	
	

# 	def init_metrics(self,
# 					 stage: str='train',
# 					 tag: Optional[str]=None):
# 		tag = tag or ""
# 		if not hasattr(self, "all_metrics"):
# 			self.all_metrics = {}
		
# 		if not hasattr(self,"num_classes") and hasattr(self.hparams, "num_classes"):
# 			self.num_classes = self.hparams.num_classes
		
# 		print(f"self.num_classes={self.num_classes}")
# 		if stage in ['train', 'all']:
# 			prefix=f'{tag}_train'.strip("_")
# 			self.metrics_train = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
# 			self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
# 			self.all_metrics['train'] = {"scalar":self.metrics_train,
# 										 "per_class":self.metrics_train_per_class}
			
# 		if stage in ['val', 'all']:
# 			prefix=f'{tag}_val'.strip("_")
# 			self.metrics_val = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
# 			self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
# 			self.all_metrics['val'] = {"scalar":self.metrics_val,
# 									   "per_class":self.metrics_val_per_class}
			
# 		if stage in ['test', 'all']:
# 			if isinstance(tag, str):
# 				prefix=tag
# 			else:
# 				prefix = "test"
# #			 prefix=f'{tag}_test'.strip("_")
# 			self.metrics_test = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
# 			self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix=prefix)
# 			self.all_metrics['test'] = {"scalar":self.metrics_test,
# 										"per_class":self.metrics_test_per_class}