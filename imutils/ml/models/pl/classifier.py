"""

imutils/models/pl/classifier.py


Created on: Wednesday March 16th, 2022  
Created by: Jacob Alexander Rose  

"""




from typing import Any, Dict, List, Sequence, Tuple, Union, Optional
import hydra
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer

import numpy as np
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from imutils.ml.utils.common import iterate_elements_in_batches, render_images

# from src.pl_modules import resnets
# from src.pl_modules import losses
from torchvision import models

from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay


class LitClassifier(pl.LightningModule):
	def __init__(self,
				 cfg: DictConfig, 
				 name: str,
				 num_classes: int, 
				 final_nl,
				 loss,
				 pretrain : bool = True,
				 self_supervised=False,
				 *args, **kwargs) -> None:

		super().__init__(*args, **kwargs)
		self.cfg = cfg
		self.save_hyperparameters(cfg)
		self.name = name

		self.pretrain = pretrain
		self.num_classes = num_classes
		self.self_supervised = self_supervised


		if self.name == "resnet18":
			self.net = models.resnet18(pretrained=False, num_classes=num_classes)
		elif self.name == "simclr_resnet18":
			self.net = models.resnet18(pretrained=False, num_classes=num_classes)
		if self.name == 'resnet50':
			self.net = models.resnet50(pretrained=True)
			n_features = self.net.fc.in_features
			fc = torch.nn.Linear(n_features, n_class)
			self.net.fc = fc
		else:
			raise NotImplementedError("Could not find network {}.".format(self.net))
		self.net.fc.weight.data.normal_(0, 0.005)
		self.net.fc.bias.data.fill_(0.1)

		self.loss = getattr(losses, loss)


			
		metric = torchmetrics.Accuracy()
		self.train_accuracy = metric.clone().cuda()
		self.val_accuracy = metric.clone().cuda()
		self.test_accuracy = metric.clone().cuda()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

	def step(self, x, y) -> Dict[str, torch.Tensor]:
		if self.self_supervised:
			z1, z2 = self.shared_step(x)
			loss = self.loss(z1, z2)
		else:
			logits = self(x)
			loss = self.loss(self.final_nl(logits, dim=-1), y)
		return {"logits": logits, "loss": loss, "y": y, "x": x}

	def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
		x, y = batch
		out = self.step(x, y)
		return out

	def training_step_end(self, out):
		self.train_accuracy(self.final_nl(out["logits"], dim=-1), out["y"])
		self.log_dict(
			{
				"train_acc": self.train_accuracy,
				"train_loss": out["loss"].mean(),
			},
			on_step=True,
			on_epoch=False
		)
		return out["loss"].mean()

	def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
		x, y = batch
		out = self.step(x, y)
		return out
	
	def validation_step_end(self, out):
		self.val_accuracy(self.final_nl(out["logits"], dim=-1), out["y"])
		self.log_dict(
			{
				"val_acc": self.val_accuracy,
				"val_loss": out["loss"].mean(),
			},
		)
		return {
			"image": out["x"],
			"y_true": out["y"],
			"logits": out["logits"],
			"val_loss": out["loss"].mean(),
		}

	def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
		x, y = batch
		out = self.step(x, y)
		return out

	def test_step_end(self, out):
		self.test_accuracy(self.final_nl(out["logits"], dim=-1), out["y"])
		self.log_dict(
			{
				"test_acc": self.test_accuracy,
				"test_loss": out["loss"].mean(),
			},
		)
		return {
			"image": out["x"],
			"y_true": out["y"],
			"logits": out["logits"],
			"val_loss": out["loss"].mean(),
		}


	def validation_epoch_end(self, outputs: List[Any]) -> None:
		"""
		
		"""
		self.render_image_predictions(
			outputs=outputs,
			batch_size=self.cfg.data.datamodule.batch_size.val,
			n_elements_to_log=self.cfg.logging.n_elements_to_log,
			log_name="val_image_predictions",
			normalize_visualization=self.cfg.logging.normalize_visualization,
			logger=self.logger,
			global_step=self.global_step)	


	def test_epoch_end(self, outputs: List[Any]) -> None:
		
		self.render_image_predictions(
			outputs=outputs,
			batch_size=self.cfg.data.datamodule.batch_size.test,
			n_elements_to_log=self.cfg.logging.n_elements_to_log,
			log_name="test_image_predictions",
			normalize_visualization=self.cfg.logging.normalize_visualization,
			logger=self.logger,
			global_step=self.global_step)


	@staticmethod
	def render_image_predictions(
		outputs: List[Any],
		batch_size: int,
		n_elements_to_log: int,
		log_name: str="image predictions",
		normalize_visualization: bool=True,
		logger=None,
		global_step: int=0
	) -> None:
		
		# images_feat_viz = []
		# integrated_gradients = IntegratedGradients(self.forward)
		# noise_tunnel = NoiseTunnel(integrated_gradients)
		
		images = []
		for output_element in iterate_elements_in_batches(
			outputs, batch_size, n_elements_to_log
		):  
			rendered_image = render_images(
				output_element["image"],
				autoshow=False,
				normalize=normalize_visualization)
			caption = f"y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"  # noqa	
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
				{log_name: images}, 
				step=global_step
			)


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
		if hasattr(self.cfg.optim.optimizer, "exclude_bn_bias") and \
				self.cfg.optim.optimizer.exclude_bn_bias:
			params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.cfg.optim.optimizer.weight_decay)
		else:
			params = self.parameters()

		opt = hydra.utils.instantiate(
			self.cfg.optim.optimizer, 
			params=params,
			weight_decay=self.cfg.optim.optimizer.weight_decay
		)
		
		if not self.cfg.optim.use_lr_scheduler:
			return opt

		# Handle schedulers if requested
		if torch.optim.lr_scheduler.warmup_steps:
			# Right now this is specific to SimCLR
			lr_scheduler = {
				"scheduler": torch.optim.lr_scheduler.LambdaLR(
					opt,
					linear_warmup_decay(
						self.cfg.optim.lr_scheduler.warmup_steps,
						self.cfg.optim.lr_scheduler.total_steps,
						cosine=True),
				),
				"interval": "step",
				"frequency": 1,
			}
		else:
			lr_scheduler = self.cfg.optim.lr_scheduler
		scheduler = hydra.utils.instantiate(lr_scheduler, optimizer=opt)
		return [opt], [scheduler]

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

	def init_metrics(self,
					 stage: str='train',
					 tag: Optional[str]=None):
		tag = tag or ""
		if not hasattr(self, "all_metrics"):
			self.all_metrics = {}
		
		if not hasattr(self,"num_classes") and hasattr(self.hparams, "num_classes"):
			self.num_classes = self.hparams.num_classes
		
		print(f"self.num_classes={self.num_classes}")
		if stage in ['train', 'all']:
			prefix=f'{tag}_train'.strip("_")
			self.metrics_train = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
			self.metrics_train_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='train')
			self.all_metrics['train'] = {"scalar":self.metrics_train,
										 "per_class":self.metrics_train_per_class}
			
		if stage in ['val', 'all']:
			prefix=f'{tag}_val'.strip("_")
			self.metrics_val = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
			self.metrics_val_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix='val')
			self.all_metrics['val'] = {"scalar":self.metrics_val,
									   "per_class":self.metrics_val_per_class}
			
		if stage in ['test', 'all']:
			if isinstance(tag, str):
				prefix=tag
			else:
				prefix = "test"
#			 prefix=f'{tag}_test'.strip("_")
			self.metrics_test = get_scalar_metrics(num_classes=self.num_classes, average='macro', prefix=prefix)
			self.metrics_test_per_class = get_per_class_metrics(num_classes=self.num_classes, prefix=prefix)
			self.all_metrics['test'] = {"scalar":self.metrics_test,
										"per_class":self.metrics_test_per_class}