"""

imutils/ml/aug/image/images.py

Created on: Wednesday March 16th, 2022  
Created by: Jacob Alexander Rose  

"""


import argparse
import cv2
from rich import print as pp
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig
import os
from torch import nn
import torch
from typing import *

from torchvision import transforms as T
import albumentations as A
from albumentations.augmentations import transforms as AT
import hydra

# DEFAULT_CFG_PATH = os.path.join(os.path.dirname(__file__), "default_image_transform_config.yaml")

DEFAULT_CFG_PATH = os.path.join(Path(__file__).parent.parent.parent, "conf", "aug", "default_image_aug.yaml")
DEFAULT_CFG = OmegaConf.load(DEFAULT_CFG_PATH)

to_tensor = T.ToTensor()

__all__ = ["instantiate_transforms", "Preprocess", "BatchTransform", "get_default_transforms"]






def functional_to_grayscale(img: np.ndarray, num_output_channels: int=3):
	"""Convert image to grayscale version of image.
	Args:
		img (np.ndarray): Image to be converted to grayscale.
	Returns:
		CV Image:  Grayscale version of the image.
					if num_output_channels == 1 : returned image is single channel
					if num_output_channels == 3 : returned image is 3 channel with r == g == b
	"""
	# if not _is_numpy_image(img):
	#	 raise TypeError('img should be CV Image. Got {}'.format(type(img)))

	if num_output_channels == 1:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	elif num_output_channels == 3:
		img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
	else:
		raise ValueError('num_output_channels should be either 1 or 3')

	return img


class Grayscale(AT.ImageOnlyTransform):
	"""Convert image to grayscale.
	Args:
		num_output_channels (int): (1 or 3) number of channels desired for output image
	Returns:
		CV Image: Grayscale version of the input.
		- If num_output_channels == 1 : returned image is single channel
		- If num_output_channels == 3 : returned image is 3 channel with r == g == b
	"""

	def __init__(self, num_output_channels=3, always_apply: bool = True, p: float = 1.0):
		super().__init__(always_apply=always_apply, p=p)
		self.num_output_channels = num_output_channels
		# replay mode params
		self.deterministic = False
		self.save_key = "replay"
		self.params: Dict[Any, Any] = {}
		self.replay_mode = False
		self.applied_in_replay = False

		
	def get_transform_init_args_names(self):
		return (
		"num_output_channels",
		)
		
		
	def apply(self, img=None,  **kwargs):
		"""
		Args:
			img (CV Image): Image to be converted to grayscale.
		Returns:
			CV Image: Randomly grayscaled image.
		"""
		img = img if "image" not in kwargs else kwargs["image"]
		return functional_to_grayscale(img, num_output_channels=self.num_output_channels)
		# return {"image":
		# 			functional_to_grayscale(img, num_output_channels=self.num_output_channels)
		# 	   }




def adjust_gamma(img, gamma, gain=1):
	"""Perform gamma correction on an image.
	Also known as Power Law Transform. Intensities in RGB mode are adjusted
	based on the following equation:
		I_out = 255 * gain * ((I_in / 255) ** gamma)
	See https://en.wikipedia.org/wiki/Gamma_correction for more details.
	Args:
		img (np.ndarray): CV Image to be adjusted.
		gamma (float): Non negative real number. gamma larger than 1 make the
			shadows darker, while gamma smaller than 1 make dark regions
			lighter.
		gain (float): The constant multiplier.
	"""
	if not _is_numpy_image(img):
		raise TypeError('img should be CV Image. Got {}'.format(type(img)))

	if gamma < 0:
		raise ValueError('Gamma should be a non-negative real number')

	im = img.astype(np.float32)
	im = 255. * gain * np.power(im / 255., gamma)
	im = im.clip(min=0., max=255.)
	return im.astype(img.dtype)













def instantiate_transforms(cfg: List[DictConfig],
						   to_grayscale: bool=False,
						   num_output_channels: int=3,
						   verbose: bool=False) -> List[Callable]:
	"""
	Compose a series of albumentations image transformations specified entirely within a config.
	
	Each augmentation's python class is specified with a _target_ key, followed by any kwargs.
	
	"""
	transforms = []
	
	if to_grayscale:
		transforms.append(Grayscale(num_output_channels=num_output_channels))


	for name, transform_step in cfg.items():
		if verbose: print(name)
		transforms.append(
			hydra.utils.instantiate(transform_step)
		)
		
	if verbose:
		pp(transforms)
	return A.Compose(transforms)














class Preprocess(nn.Module):

	def __init__(self, mode="train", resize=None, to_tensor: bool=True):
		super().__init__()
		self.mode = mode
		self.resize = resize
		self.to_tensor = to_tensor
		self.resize_func = T.Resize(self.resize)
	
	@torch.no_grad()  # disable gradients for effiency
	def forward(self, x) -> torch.Tensor:
		# x_tmp: np.ndarray = np.array(x)  # HxWxC
		if self.to_tensor:
			x: Tensor = to_tensor(x)  # CxHxW
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
				 random_flips: bool=True,
				 normalize = (
					 [0,0,0],
					 [1,1,1]
				 ),
				 skip_augmentations: bool=False
				) -> None:
		super().__init__()
		self.mode = mode
		self.random_resize_crop = random_resize_crop
		self.center_crop = center_crop
		self._apply_color_jitter = apply_color_jitter
		self.normalize = normalize
		self.random_flips = random_flips
		self.skip_augmentations = skip_augmentations
		self.build_transforms(mode=mode)

		
	def add_train_transforms(self, transforms=None):
		
		transforms = transforms or []
		
		if self.skip_augmentations:
			if self.random_resize_crop:
				transforms.append(T.CenterCrop(self.random_resize_crop))
		else:
			transforms.append(T.RandomPerspective())
			if type(self.random_resize_crop) == int:
				transforms.append(T.RandomResizedCrop(self.random_resize_crop))
			if self.random_flips:
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
			
		print(f"self.normalize: {self.normalize}")

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
			x_out = self.jitter(image=x_out)
		return x_out
	


def get_default_transforms(
		mode: str="train",
		compose: bool=True,
		config = dict(
			preprocess={
				'train': {'resize': 512},
				'val': {'resize': 256},
				'test': {'resize': 256}},

			batch_transform={
				'train': {'random_resize_crop': 224}, 
				'val': {'center_crop': 224},
				'test': {'center_crop': 224}},
			normalize=(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			),
			apply_color_transform=False,
			random_flips=True,
			skip_augmentations=False
		)
	) -> Tuple[Callable]:
	
	config = OmegaConf.merge(DEFAULT_CFG, config)
	
	if config["preprocess"][mode].get("resize", None):
		preprocess_transforms = Preprocess(mode=mode, 
										   resize=config["preprocess"][mode]["resize"])
	else:
		preprocess_transforms = T.ToTensor()
	
	if mode == "train":
		random_resize_crop = config["batch_transform"]["train"]["random_resize_crop"]
		center_crop = None
	else:
		random_resize_crop = None
		center_crop = config["batch_transform"][mode]["center_crop"]
	apply_color_jitter = config.get("apply_color_transform", False)
	random_flips = config.get("random_flips", True)
	skip_augmentations = config.get("skip_augmentations", False)
	normalize = config.get("normalize", 
						   (
		[0,0,0],
		[1,1,1]
	)
						  )
	
	batch_transforms = BatchTransform(mode=mode,
									   random_resize_crop=random_resize_crop,
									   center_crop=center_crop,
									   apply_color_jitter = apply_color_jitter,
									   random_flips = random_flips,
									   normalize = normalize,
									   skip_augmentations=skip_augmentations)
	
	transforms = (preprocess_transforms, 
				  batch_transforms)
	if compose:
		transforms = T.Compose(transforms)
	return transforms



if __name__=="__main_":
	
	config = DEFAULT_CFG
	
	train_preprocess_transforms, train_batch_transforms = get_default_transforms(mode="train",
																				 **config)

	val_preprocess_transforms, val_batch_transforms = get_default_transforms(mode="val",
																			 **config)