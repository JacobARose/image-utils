"""

imutils/ml/aug/image/images.py

Created on: Wednesday March 16th, 2022  
Created by: Jacob Alexander Rose  

"""


import argparse
from rich import print as pp
import numpy as np
from omegaconf import OmegaConf
import os
from torch import nn
import torch
from typing import *

from torchvision import transforms as T
from albumentations.augmentations import transforms as AT

DEFAULT_CFG_PATH = os.path.join(os.path.dirname(__file__), "default_image_transform_config.yaml")
DEFAULT_CFG = OmegaConf.load(DEFAULT_CFG_PATH)

to_tensor = T.ToTensor()

__all__ = ["Preprocess", "BatchTransform", "get_default_transforms"]



class Preprocess(nn.Module):

	def __init__(self, mode="train", resize=None):
		super().__init__()
		self.mode = mode
		self.resize = resize		
		self.resize_func = T.Resize(self.resize)
	
	@torch.no_grad()  # disable gradients for effiency
	def forward(self, x) -> torch.Tensor:
		# x_tmp: np.ndarray = np.array(x)  # HxWxC
		# x_out: Tensor = to_tensor(x_tmp, keepdim=True)  # CxHxW
		if self.resize:
			x = self.resize_func(x)

		return x #_out.float()# / 255.0




class BatchTransform(nn.Module):
	"""Module to perform data augmentation using Kornia on torch tensors."""

	def __init__(self,
				 mode: str="train",
				 random_resize_crop=None,
				 center_crop=None,
				 apply_color_jitter: bool = False,
				 normalize = (
					 [0,0,0],
					 [1,1,1]
				 )
				) -> None:
		super().__init__()
		self.mode = mode
		self.random_resize_crop = random_resize_crop
		self.center_crop = center_crop
		self._apply_color_jitter = apply_color_jitter
		self.normalize = normalize
		
		self.build_transforms(mode=mode)

		
	def add_train_transforms(self, transforms=None):
		
		transforms = transforms or []
		# if mode == "train":
		transforms.append(T.RandomPerspective())
		if type(self.random_resize_crop) == int:
			transforms.append(T.RandomResizedCrop(self.random_resize_crop))
		transforms.extend([
			T.RandomHorizontalFlip(),
			T.RandomVerticalFlip()
		])
		return transforms

	def add_test_transforms(self, transforms=None):
		
		transforms = transforms or []
		if type(self.center_crop) == int:
			transforms.append(T.CenterCrop(self.center_crop))
		return transforms


	def build_transforms(self,
						 mode: str = "train"):
		transforms = []
		if mode == "train":
			transforms = self.add_train_transforms(transforms=transforms)
		elif mode in ["val", "test"]:
			transforms = self.add_test_transforms(transforms=transforms)

		transforms.extend([
			# T.ToTensor(),
			T.Normalize(*self.normalize)
		])

		self.transforms = nn.Sequential(*transforms)
		self.jitter = AT.ColorJitter(brightness=0.2,
									 contrast=0.2,
									 saturation=0.2,
									 hue=0.2,
									 always_apply=False,
									 p=0.5)

	@torch.no_grad()  # disable gradients for effiency
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_out = self.transforms(x)  # BxCxHxW
		if self._apply_color_jitter:
			x_out = self.jitter(x_out)
		return x_out
	


def get_default_transforms(
		mode: str="train",
		config = dict(
			preprocess={
				'train': {'resize': 512},
				'val': {'resize': 256}},

			batch_transform={
				'train': {'random_resize_crop': 224}, 
				'val': {'center_crop': 224}},
			normalize=(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		)
	) -> Tuple[Callable]:
	
	config = OmegaConf.merge(DEFAULT_CFG, config)
	
	preprocess_transforms = Preprocess(mode=mode, 
									   resize=config["preprocess"][mode]["resize"])
	
	batch_transforms = BatchTransform(mode=mode,
									   random_resize_crop=None, #config["batch_transform"]["val"]["random_resize_crop"],
									   center_crop=None, #config["batch_transform"]["val"]["center_crop"],
									   apply_color_jitter = False,
									   normalize = normalize)
	
	return preprocess_transforms, batch_transforms



if __name__=="__main_":
	
	config = DEFAULT_CFG
	
	train_preprocess_transforms, train_batch_transforms = get_default_transforms(mode="train",
																				 **config)

	val_preprocess_transforms, val_batch_transforms = get_default_transforms(mode="val",
																			 **config)