"""

lightning_hydra_classifiers/models/layers/pool_layers.py

Description: Defines custom pytorch layers

Created on: Thursday October 14th, 2021
Author: Jacob A Rose


"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Union
import timm
import glob
# import hydra
from collections import OrderedDict


__all__ = ["Flatten", "AdaptiveConcatPool2d", "build_global_pool"]


class Flatten(nn.Module):
	def forward(self,x):
		"""
		Assumes the batch dim is dim=0, thus flattens all dims above 0 into dim 1.
		"""
		return torch.flatten(x, start_dim=1)
	

class AdaptiveConcatPool2d(nn.Module):
	"Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

	def __init__(self, sz: Optional[int] = None):
		super(AdaptiveConcatPool2d, self).__init__()
		"Output will be 2*sz or 2 if sz is None"
		self.output_size = sz or 1
		self.ap = nn.AdaptiveAvgPool2d(self.output_size)
		self.mp = nn.AdaptiveMaxPool2d(self.output_size)

	def forward(self, x):
		"""
		Compute (1) the maxpool(x), and (2) the averagepool(x), then concatenate their outputs.
		"""
		return torch.cat([self.mp(x), self.ap(x)], 1)
	

def build_global_pool(pool_type: str="avg",
					  pool_size: int=1, 
					  dropout_p: float=0.3,
					  feature_size: Optional[int]=0):

#	 head = OrderedDict()
	if pool_type == 'avg':
		global_pool = nn.AdaptiveAvgPool2d(pool_size)
		return global_pool, feature_size
		
	elif pool_type == 'avgdrop':
		global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(pool_size),
									nn.Dropout2d(p=dropout_p, inplace=False))
		return global_pool, feature_size
	elif pool_type == 'avgmax':
		feature_size = feature_size * 2
		global_pool = AdaptiveConcatPool2d(pool_size)
		return global_pool, feature_size
	elif pool_type == 'avgmaxdrop':
		feature_size = feature_size * 2
		global_pool = nn.Sequential(AdaptiveConcatPool2d(pool_size),
									nn.Dropout2d(p=dropout_p, inplace=False))
		return global_pool, feature_size

	elif pool_type == 'max':
		global_pool = nn.AdaptiveMaxPool2d(pool_size)
		return global_pool, feature_size
	
	else:
		raise NotImplementedError(f"pool_type={pool_type} invalid.")


