"""
imutils/ml/data/datamodule.py

Created on: Wednesday March 16th, 2022  
Created by: Jacob Alexander Rose  


- Update (Wednesday March 24th, 2022)
	- Refactored to make more modular BaseDataset and BaseDataModule, which Herbarium2022Dataset and Herbarium2022DataModule inherit, respectively.
- Update (Wednesday April 6th, 2022)
	- Wonky refactor to add ExtantLeavesDataset and ExtantLeavesDataModule to definitions. Involed some blurring of abstractions -- will need to refactor the base class AbstractCatalogDataset.
"""

import dataclasses
from dataclasses import dataclass, asdict, replace
import matplotlib.pyplot as plt
from icecream import ic
import jpeg4py as jpeg
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from rich import print as pp
import multiprocessing as mproc
import pytorch_lightning as pl
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torchvision
from pytorch_lightning.utilities import rank_zero_only
from typing import *


# from imutils.big.make_train_val_splits import main as make_train_val_splits
from imutils.big.make_train_val_splits import main as make_train_val_splits

from imutils.big.split_catalog_utils import (check_already_built,
											   read_encoded_splits,
											   find_data_splits_dir)


# from imutils.big.transforms.image import (Preprocess,
from imutils.ml.aug.image.images import (instantiate_transforms,
										 DEFAULT_CFG as DEFAULT_TRANSFORM_CFG)

from imutils.ml.utils import label_utils, taxonomy_utils
from imutils.ml.data.dataset import Herbarium2022Dataset, ExtantLeavesDataset

__all__ = ["Herbarium2022DataModule",
		   "ExtantLeavesDataModule"]

import torch

def tensor_to_image(x: torch.Tensor) -> np.ndarray:
	if x.ndim==3:
		return x.permute(1,2,0)
	return x.permute(0, 2, 3, 1)





@dataclass
class DataModuleConfig:

	catalog_dir: str=None
	label_col: str="family"
	shuffle: bool=True
	seed:int=14
	batch_size: int=128
	num_workers: int=4
	pin_memory: bool=True
	persistent_workers: Optional[bool]=False
	transform_cfg: Optional["Config"]=None
	to_grayscale: bool=False
	num_channels: int=3
	remove_transforms: bool=False



@dataclass
class ExtantLeavesDataModuleConfig(DataModuleConfig):

	catalog_dir: str="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/extant_leaves_family_3_512"
	label_col: str="family"
	splits: Tuple[float]=(0.5,0.2,0.3)


@dataclass
class FossilLeavesDataModuleConfig(DataModuleConfig):

	catalog_dir: str="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/Fossil_family_3_512"
	label_col: str="family"
	splits: Tuple[float]=(0.5,0.2,0.3)


@dataclass
class PNASDataModuleConfig(DataModuleConfig):

	catalog_dir: str="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/PNAS_family_100_512"
	label_col: str="family"
	splits: Tuple[float]=(0.5,0.2,0.3)


@dataclass
class Herbarium2022DataModuleConfig(DataModuleConfig):

	catalog_dir: str="/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize-512/catalogs" #/splits/train_size-0.8"
	label_col: str="scientificName"
	train_size: float=0.8


############################
############################
	



class BaseDataModule(pl.LightningDataModule):
	dataset_cls = None #Herbarium2022Dataset
	train_dataset = None
	val_dataset = None
	test_dataset = None
	
	train_transform = None
	val_transform = None
	test_transform = None
	
	default_cfg: DataModuleConfig = DataModuleConfig()
	transform_cfg = None #DEFAULT_TRANSFORM_CFG


	@classmethod
	def from_cfg(cls,
				 cfg: Union[DataModuleConfig, DictConfig],
				 **kwargs):
		if isinstance(cfg, DataModuleConfig):
			cfg = dataclasses.asdict(cfg)
		elif isinstance(cfg, DictConfig):
			cfg = OmegaConf.to_container(cfg, resolve=True)
		
		cfg = dataclasses.replace(cls.default_cfg, **cfg)

		return cls(**cfg)


	def get_cfg(self, as_dict: bool=False) -> DataModuleConfig:
		cfg = replace(self.default_cfg, 
					  catalog_dir=self.catalog_dir,
					  label_col=self.label_col,
					  splits=self.splits,
					  shuffle=self.shuffle,
					  seed=self.seed,
					  batch_size=self.batch_size,
					  num_workers=self.num_workers,
					  pin_memory=self.pin_memory,
					  persistent_workers=self.persistent_workers,
					  transform_cfg=self.transform_cfg,
					  remove_transforms=self.remove_transforms)
		if as_dict:
			return asdict(cfg)
		return cfg


	def prepare_data(self):
		pass

	def setup(self, stage="fit"):
		subsets=[]
		if stage in ["train", "fit", "all", None]:
			self.train_dataset = self.dataset_cls(catalog_dir=self.catalog_dir,
													  subset="train",
													  label_col=self.label_col,
													  train_size=getattr(self, "train_size", None),
													  splits=getattr(self, "splits", None),
													  shuffle=self.shuffle,
													  seed=self.seed,
													  transform=self.train_transform)
			# self.setup_taxonomy_table(df=self.train_dataset.df,
			# 						  smallest_taxon_col=self.label_col)
			subsets.append("train")
		if stage in ["val", "fit", "all", None]:
			self.val_dataset = self.dataset_cls(catalog_dir=self.catalog_dir,
												subset="val",
												label_col=self.label_col,
												train_size=getattr(self, "train_size", None),
												splits=getattr(self, "splits", None),
												shuffle=self.shuffle,
												seed=self.seed,
												transform=self.val_transform)
			subsets.append("val")
		if stage in ["test", "all", None]:
			self.test_dataset = self.dataset_cls(catalog_dir=self.catalog_dir,
												 subset="test",
												 label_col=self.label_col,
												 train_size=getattr(self, "train_size", None),
												 splits=getattr(self, "splits", None),
												 shuffle=self.shuffle,
												 seed=self.seed,
												 transform=self.test_transform)
			subsets.append("test")
			
		for s in subsets:
			self.get_dataset_size(subset=s,
								  verbose=True)
			
		self.set_image_reader(self.image_reader)
		self.setup_taxonomy_table(
			df=self.train_dataset.df,
			smallest_taxon_col=getattr(self, "smallest_taxon_col", None)
		)


	def setup_transforms(self,
						 transform_cfg: dict=None,
						 train_transform=None,
						 val_transform=None,
						 test_transform=None,
						 remove_transforms: bool=False):
		if transform_cfg is None and self.transform_cfg is None:
			remove_transforms = True
		transform_cfg = transform_cfg or {}
		self.remove_transforms = remove_transforms
		if self.remove_transforms:
			for subset in ["train", "val", "test"]:
				setattr(self, f"{subset}_transform", None)
				if self.get_dataset(subset) is not None:
					self.get_dataset(subset).transform = None
			return
		else:
			self.transform_cfg = OmegaConf.merge(self.transform_cfg, transform_cfg)
			# print("self.transform_cfg:"); pp(self.transform_cfg)
			self.train_transform = (
				instantiate_transforms(cfg=self.transform_cfg.train, to_grayscale=self.to_grayscale, num_output_channels=self.num_channels, verbose=False)
				if train_transform is None else train_transform
			)
			self.val_transform = (
				instantiate_transforms(cfg=self.transform_cfg.val, to_grayscale=self.to_grayscale, num_output_channels=self.num_channels, verbose=False)
				if val_transform is None else val_transform
			)
			self.test_transform = (
				instantiate_transforms(cfg=self.transform_cfg.test, to_grayscale=self.to_grayscale, num_output_channels=self.num_channels, verbose=False)
				if test_transform is None else test_transform
			)
		for subset in ["train", "val", "test"]:
			if self.get_dataset(subset) is not None:
				# Replace the existing transforms on any already setup datasets
				self.get_dataset(subset).transform = getattr(self, f"{subset}_transform")

	def set_image_reader(self,
						 reader: Callable) -> None:
		"""
		Pass in a callable that reads image data from disk,
		which is assigned to each of this datamodule's datasets, respectively.
		"""
		for data in [self.train_dataset, self.val_dataset, self.test_dataset]:
			if data is None:
				continue
			data.set_image_reader(reader)


	def setup_taxonomy_table(self, 
							 df: pd.DataFrame=None,
							 smallest_taxon_col: str="Species",
							 taxonomy: taxonomy_utils.TaxonomyLookupTable=None):
		if isinstance(taxonomy, taxonomy_utils.TaxonomyLookupTable):
			self.taxonomy = taxonomy
		else:
			self.taxonomy = taxonomy_utils.TaxonomyLookupTable(df=df,
															   smallest_taxon_col=smallest_taxon_col)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=True,
			pin_memory=self.pin_memory,
			persistent_workers=self.persistent_workers
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,#*2,
			num_workers=self.num_workers,
			shuffle=False,
			pin_memory=self.pin_memory,
			persistent_workers=self.persistent_workers
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,#*2,
			num_workers=self.num_workers,
			shuffle=False,
			pin_memory=self.pin_memory
		)

	def get_dataloader(self,
					   subset:str="train"):
		if subset == "train":
			return self.train_dataloader()
		elif subset == "val":
			return self.val_dataloader()
		elif subset == "test":
			return self.test_dataloader()
		else:
			return None

	def get_dataset(self,
					   subset:str="train"):
		if subset == "train":
			return self.train_dataset
		elif subset == "val":
			return self.val_dataset
		elif subset == "test":
			return self.test_dataset
		else:
			return None

	@property
	def num_classes(self) -> int:
		assert self.train_dataset and self.val_dataset
		return max(self.train_dataset.num_classes, self.val_dataset.num_classes)

	def num_samples(self, 
					subset: str="train"):
		return len(self.get_dataset(subset=subset))
	
	def num_batches(self, 
					subset: str="train"):
		return len(self.get_dataloader(subset=subset))
	
	def get_dataset_size(self, 
						subset: str="train",
						verbose: bool=False):
		num_samples = self.num_samples(subset) # len(datamodule.get_dataset(subset=subset))
		num_batches = self.num_batches(subset) # len(datamodule.get_dataloader(subset=subset))
		if verbose:
			# print(f"{subset} --> (num_samples: {num_samples:,}), (num_batches: {num_batches:,})")
			rank_zero_only(ic)(subset, num_samples, num_batches, self.num_classes, self.batch_size)
		return num_samples, num_batches



	def show_batch(self, batch_idx: int=0, nrow: int=4, figsize=(10, 10)):
		def _to_vis(data):
			return tensor_to_image(torchvision.utils.make_grid(data, nrow=nrow, normalize=True))
		transform_cfg = self.transform_cfg

		bsz = self.batch_size
		indices = list(range(batch_idx*bsz, (batch_idx+1)*bsz))

		self.setup_transforms(remove_transforms=True)
		imgs = [self.train_dataset[i][0] for i in indices]
		self.setup_transforms(transform_cfg=self.transform_cfg)
		imgs_aug = [self.train_dataset[i][0] for i in indices]
		
		fig, ax = plt.subplots(1,2, figsize = (2*figsize[0], figsize[1]))
		ax[0].set_title("image")
		ax[1].set_title("aug image")
		ax[1].set_yticklabels([])
		ax[1].set_facecolor('#eafff5')
		ax[0].imshow(_to_vis(imgs))
		ax[1].imshow(_to_vis(imgs_aug))
		plt.tight_layout(w_pad=0.05)
		


##############################
##############################



class Herbarium2022DataModule(BaseDataModule):
	catalog_dir: str="/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize-512/catalogs" #/splits/train_size-0.8"
	dataset_cls = Herbarium2022Dataset
	# transform_cfg = DEFAULT_TRANSFORM_CFG
	default_cfg: Herbarium2022DataModuleConfig = Herbarium2022DataModuleConfig(catalog_dir=catalog_dir)

	def __init__(self,
				 catalog_dir: Optional[str]=None,
				 label_col="scientificName",
				 train_size=0.8,
				 smallest_taxon_col: str="Species",
				 shuffle: bool=True,
				 seed=14,
				 batch_size: int = 128,
				 num_workers: int = None,
				 pin_memory: bool=True,
				 persistent_workers: Optional[bool]=False,
				 train_transform=None,
				 val_transform=None,
				 test_transform=None,
				 transform_cfg=None,
				 to_grayscale: bool=False,
				 num_channels: int=3,
				 remove_transforms: bool=False,
				 image_reader: Callable="default", #Image.open,
				 **kwargs
	):
		super().__init__()
		
		self.catalog_dir = catalog_dir or self.catalog_dir
		self.label_col = label_col
		self.train_size = train_size
		self.shuffle = shuffle
		self.seed = seed
		self.batch_size = batch_size
		self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()
		self.pin_memory = pin_memory
		self.persistent_workers = persistent_workers
		self.image_reader = image_reader
		self.to_grayscale = to_grayscale
		self.num_channels = num_channels
		self.smallest_taxon_col = smallest_taxon_col

		self.setup_transforms(transform_cfg=transform_cfg,
							  train_transform=train_transform,
							  val_transform=val_transform,
							  test_transform=test_transform,
							  remove_transforms=remove_transforms)
		self.setup()
		self.cfg = self.get_cfg()
		self.kwargs = kwargs





	def get_cfg(self, as_dict: bool=False) -> ExtantLeavesDataModuleConfig:
		cfg = replace(self.default_cfg, 
					  catalog_dir=self.catalog_dir,
					  label_col=self.label_col,
					  train_size=self.train_size,
					  shuffle=self.shuffle,
					  seed=self.seed,
					  batch_size=self.batch_size,
					  num_workers=self.num_workers,
					  pin_memory=self.pin_memory,
					  persistent_workers=self.persistent_workers,
					  transform_cfg=self.transform_cfg,
					  to_grayscale=self.to_grayscale,
					  num_channels=self.num_channels,
					  remove_transforms=self.remove_transforms)
		if as_dict:
			return asdict(cfg)
		return cfg



#################
#################

from imutils.big import make_train_val_test_splits as leavesdb_utils
from imutils.big.make_train_val_test_splits import main as make_train_val_test_splits


class ExtantLeavesDataModule(BaseDataModule):
	catalog_dir: str="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/extant_leaves_family_3_512/splits/splits=(0.5,0.2,0.3)"
	dataset_cls = ExtantLeavesDataset	
	# transform_cfg = DEFAULT_TRANSFORM_CFG	
	default_cfg: ExtantLeavesDataModuleConfig = ExtantLeavesDataModuleConfig(catalog_dir=catalog_dir)

	def __init__(self,
				 catalog_dir: Optional[str]=None,
				 label_col="family",
				 splits: Tuple[float]=(0.5,0.2,0.3),
				 smallest_taxon_col: str="Species",
				 shuffle: bool=True,
				 seed=14,
				 batch_size: int = 128,
				 num_workers: int = None,
				 pin_memory: bool=True,
				 persistent_workers: Optional[bool]=False,
				 train_transform=None,
				 val_transform=None,
				 test_transform=None,
				 transform_cfg=None,
				 to_grayscale: bool=False,
				 num_channels: int=3,
				 remove_transforms: bool=False,
				 image_reader: Callable="default", #Image.open,
				 **kwargs
	):
		super().__init__()
		
		self.catalog_dir = catalog_dir or self.catalog_dir
		self.label_col = label_col
		self.splits = splits
		self.shuffle = shuffle
		self.seed = seed
		self.batch_size = batch_size
		self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()
		self.pin_memory = pin_memory
		self.persistent_workers = persistent_workers
		self.image_reader = image_reader
		self.to_grayscale = to_grayscale
		self.num_channels = num_channels
		self.smallest_taxon_col = smallest_taxon_col

		self.setup_transforms(transform_cfg=transform_cfg,
							  train_transform=train_transform,
							  val_transform=val_transform,
							  test_transform=test_transform,
							  remove_transforms=remove_transforms)
		self.cfg = self.get_cfg()
		self.setup() 
		self.kwargs = kwargs


# 	@classmethod
# 	def from_cfg(cls,
# 				 cfg: Union[ExtantLeavesDataModuleConfig, DictConfig],
# 				 **kwargs):
# 		if isinstance(cfg, ExtantLeavesDatasetConfig):
# 			cfg = asdict(cfg)
# 		elif isinstance(cfg, DictConfig):
# 			cfg = OmegaConf.to_container(cfg, resolve=True)
# 		cfg = replace(self.default_cfg, **cfg)
# 		return cls(**cfg)


	def get_cfg(self, as_dict: bool=False) -> ExtantLeavesDataModuleConfig:
		cfg = replace(self.default_cfg, 
					  catalog_dir=self.catalog_dir,
					  label_col=self.label_col,
					  splits=self.splits,
					  shuffle=self.shuffle,
					  seed=self.seed,
					  batch_size=self.batch_size,
					  num_workers=self.num_workers,
					  pin_memory=self.pin_memory,
					  persistent_workers=self.persistent_workers,
					  transform_cfg=self.transform_cfg,
					  remove_transforms=self.remove_transforms)
		if as_dict:
			return asdict(cfg)
		return cfg


@dataclass
class AutoDataModule:
	"""
	
	
	"""
	
	# def __init__(self)
	
	@classmethod
	def from_config(self,
						  name: str):
		if "Herbarium" in name:
			["Herbarium2022DataModule",
			 "ExtantLeavesDataModule"]

















