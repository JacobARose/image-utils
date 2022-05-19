"""
imutils/ml/data/datamodule.py

Created on: Wednesday March 16th, 2022  
Created by: Jacob Alexander Rose  


- Update (Wednesday March 24th, 2022)
	- Refactored to make more modular BaseDataset and BaseDataModule, which Herbarium2022Dataset and Herbarium2022DataModule inherit, respectively.
- Update (Wednesday April 6th, 2022)
	- Wonky refactor to add ExtantLeavesDataset and ExtantLeavesDataModule to definitions. Involed some blurring of abstractions -- will need to refactor the base class AbstractCatalogDataset.
"""

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
										 Preprocess,
										 BatchTransform,
										 get_default_transforms,
										 DEFAULT_CFG as DEFAULT_TRANSFORM_CFG)

from imutils.ml.utils import label_utils, taxonomy_utils

__all__ = ["Herbarium2022DataModule", 
		   "Herbarium2022Dataset",
		   "ExtantLeavesDataset", 
		   "ExtantLeavesDataModule",
		   "get_default_transforms"]

import torch

def tensor_to_image(x: torch.Tensor) -> np.ndarray:
	if x.ndim==3:
		return x.permute(1,2,0)
	return x.permute(0, 2, 3, 1)


def read_jpeg(path):
	return jpeg.JPEG(path).decode()

def read_pil(path):
	return Image.open(path)

def read_torchvision_img(path):
	img=torchvision.io.read_image(path)
	return img


default_reader = read_jpeg


IMAGE_READERS = {
	"jpeg.JPEG":read_jpeg,
	"PIL":read_pil,
	"torchvision":read_torchvision_img,
	"default":default_reader
}




@dataclass
class ExtantLeavesDatasetConfig:
	catalog_dir: str="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/extant_leaves_family_3_512" #/splits/splits=(0.5,0.2,0.3)"
	subset: str="train"
	label_col: str="family"
	x_col: str="path"
	y_col: str="y"
	id_col: str="catalog_number"
	smallest_taxon_col: str="Species"
	splits: Tuple[float]=(0.5,0.2,0.3)
	shuffle: bool=True
	seed: int=14


@dataclass
class ExtantLeavesDataModuleConfig:

	catalog_dir: str="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/extant_leaves_family_3_512" #/splits/splits=(0.5,0.2,0.3)"
	label_col: str="family"
	splits: Tuple[float]=(0.5,0.2,0.3)
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


############################
############################

@dataclass
class Herbarium2022DatasetConfig:
	catalog_dir: str="/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize-512/catalogs" #/splits/train_size-0.8"
	# catalog_dir: str="/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize-512/catalogs"
	subset: str="train"
	label_col: str="scientificName"
	x_col: str="path"
	y_col: str="y"
	id_col: str="image_id"
	smallest_taxon_col: str="Species"
	train_size: float=0.8
	shuffle: bool=True
	seed: int=14


@dataclass
class Herbarium2022DataModuleConfig:

	catalog_dir: str="/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize-512/catalogs" #/splits/train_size-0.8"
	label_col: str="scientificName"
	train_size: float=0.8
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










class AbstractCatalogDataset(Dataset):

	def __init__(self,
				 label_col: str="scientificName",
				 x_col: str="path",
				 y_col: str="y",
				 id_col: str="image_id",
				 smallest_taxon_col: str="Species"):
		super().__init__()
		"""
		TBD: document these function kwargs
		
		"""
		self.label_col = label_col
		self.x_col = x_col
		self.y_col = y_col
		self.id_col = id_col
		self.smallest_taxon_col = smallest_taxon_col
	
	@property
	def splits_dir(self) -> Path:
		return find_data_splits_dir(source_dir=self.catalog_dir,
									train_size=self.train_size)
	
	@property
	def split_file_path(self) -> Path:
		"""
		Should return the path of this subset's on-disk csv catalog file. I emphasize should.
		"""
		return self.splits_dir / f"{self.subset}_metadata.csv"
	
	
	@property
	def already_built(self) -> bool:
		"""
		[TODO] Make abstract, move implementation to subclasses
		"""
		return check_already_built(self.splits_dir, label_col=self.label_col)

	
	def prepare_metadata(self):
		"""
		[TODO] Make abstract, move implementation to subclasses
		"""
		if not self.already_built:		
			data = make_train_val_splits(source_dir=self.catalog_dir,
										 save_dir=self.splits_dir,
										 label_col=self.label_col,
										 train_size=self.train_size,
										 seed=self.seed)

			return data


	def get_data_subset(self,
						subset: str="train") -> Tuple["LabelEncoder", pd.DataFrame]:
		"""
		Read the selected data subset into a pd.DataFrame
		
		Returns a Tuple containing an sklearn LabelEncoder and a pd.DataFrame of the subset's data catalog
		"""
		data = read_encoded_splits(source_dir=self.splits_dir,
								   include=[subset],
								   label_col=self.label_col)
		encoder = data["label_encoder"]
		data = data["subsets"][subset]
		
		return encoder, data

	
	def setup(self):
		"""
		Assigns the following instance attributes:
		
			::self.label_encoder
			::self.df
			::self.paths
			::self.targets
			::self.num_classes
		"""
		data = self.prepare_metadata()
		if data is None:
			encoder, data = self.get_data_subset(subset=self.subset)
		else:
			encoder = data["label_encoder"]
			data = data["subsets"][self.subset]

		if isinstance(encoder, preprocessing.LabelEncoder):
			"""
			Auto wraps any sklearn LabelEncoder in our custom class.
			(Added 2022-03-25 - untested)
			"""
			encoder = label_utils.LabelEncoder.from_sklearn(encoder)

		setattr(data, "label_encoder", encoder)
		self.label_encoder = encoder
		self.df = data
		
		if self.shuffle:
			self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=False)

		if self.id_col in self.df.columns:
			self.image_ids = self.df[self.id_col]
		self.paths = self.df[self.x_col]

		if self.is_supervised:
			# self.imgs = 
			self.targets = self.df[self.y_col]
			self.num_classes = len(set(self.df[self.y_col]))
		else:
			self.targets = None
			self.num_classes = -1 # Or 0?
	
		# if getattr(self, "subset") == "train":
		# self.setup_taxonomy_table(self.df,
		# 						  smallest_taxon_col=self.smallest_taxon_col)
	

	def get_decoded_targets(self, with_image_ids: bool=True) -> Tuple[str, np.ndarray]:
		assert self.is_supervised
		return self.label_encoder.inv_transform(self.targets)

	@property
	def classes(self):
		return self.label_encoder.classes

	@property
	def class2idx(self):
		return self.label_encoder.class2idx

	def setup_taxonomy_table(self, 
							 df: pd.DataFrame=None,
							 smallest_taxon_col: str="Species",
							 taxonomy: taxonomy_utils.TaxonomyLookupTable=None):
		if isinstance(taxonomy, taxonomy_utils.TaxonomyLookupTable):
			self.taxonomy = taxonomy
		else:
			self.taxonomy = taxonomy_utils.TaxonomyLookupTable(df=df,
															   smallest_taxon_col=smallest_taxon_col)



class BaseDataset(AbstractCatalogDataset):
	catalog_dir: str = os.path.abspath("./data")
	
	def __init__(self,
				 catalog_dir: Optional[str]=None,
				 subset: str="train",
				 label_col: str="scientificName",
				 x_col: str="path",
				 y_col: str="y",
				 id_col: str="image_id",
				 smallest_taxon_col: str="Species",
				 train_size: float=0.8,
				 shuffle: bool=True,
				 seed: int=14,
				 image_reader: Union[Callable,str]="default",
				 preprocess: Callable=None,
				 transform: Callable=None,
				 output_image_type = torch.Tensor,
				 output_image_range: Tuple[Any]=(0,1)):
		"""
		
		Arguments:
		
			catalog_dir: Optional[str]=None,
				
			subset: str="train",
			
			label_col: str="scientificName",
				Column containing the fully decoded str labels
			train_size: float=0.7,
			shuffle: bool=True,
			seed: int=14,
			image_reader: Union[Callable,str]="default", #Image.open,
			preprocess: Callable=None,
			transform: Callable=None
		
		"""
		super().__init__(
			label_col=label_col,
			x_col=x_col,
			y_col=y_col,
			id_col=id_col,
			smallest_taxon_col=smallest_taxon_col)
		
		self.catalog_dir = catalog_dir or self.catalog_dir
		self.train_size = train_size
		self.shuffle = shuffle
		self.seed = seed
		self.subset = subset
		self.is_supervised = bool(subset != "test")
		self.set_image_reader(image_reader)
		self.preprocess = preprocess
		self.transform = transform
		self.output_image_type = output_image_type
		self.output_image_range = output_image_range
		# self.setup()
		# self.cfg = self.get_cfg()

	@classmethod
	def from_cfg(cls,
				 cfg: DictConfig,
				 **kwargs):
		cfg = OmegaConf.merge(cfg, kwargs)
		return cls(**cfg)


	def get_cfg(self,
				 cfg: DictConfig=None,
				 **kwargs):
		cfg=cfg or {}
		default_cfg = DictConfig(dict(			
			catalog_dir=self.catalog_dir or None,
			subset=self.subset or "train",
			label_col=self.label_col or "scientificName",
			train_size=self.train_size or 0.7,
			shuffle=self.shuffle,
			seed=self.seed or 14))
		
		cfg = OmegaConf.merge(default_cfg, cfg, kwargs)
		return cfg

	def __len__(self):
		return len(self.df)
	
	def set_image_reader(self,
						 reader: Union[Callable,str]) -> None:
		if isinstance(reader, str):
			if reader not in IMAGE_READERS:
				print(f"specified image_reader is invalid, using default jpeg.JPEG")
			reader = IMAGE_READERS.get(reader, IMAGE_READERS["default"])
		elif not isinstance(reader, Callable):
			raise InvalidArgument
		
		self.reader = reader
		
	def parse_output_image(self, img: Any):
		if isinstance(img, self.output_image_type):
			return img
		if self.output_image_type == torch.Tensor:
			if isinstance(img, np.ndarray):
				if img.dtype == "uint8":
					img = img.astype("float32")
				if np.allclose(img.max(), 255.0):
					img = img / 255.0
				return torch.from_numpy(img).permute(2,0,1)
			elif isinstance(img, PIL.Image.Image):
				return T.ToTensor()(img)
		elif self.output_image_type == np.ndarray:
			if isinstance(img, torch.Tensor):
				img = img.permute(1,2,0).numpy()
			elif isinstance(img, PIL.Image.Image):
				img = np.array(img)
			
			if self.output_image_range == (0,1):
				if img.dtype == "uint8":
					img = img.astype("float32")
				if np.allclose(img.max(), 255.0):
					img = img / 255.0
			return img

		else:
			raise Exception(f"Warning, parse_output_image received unexpected image of type {type(img)=}")

	def parse_sample(self, index: int):
		return self.df.iloc[index, :]
		
	def fetch_item(self, index: int) -> Tuple[str]:
		"""
		Returns identically-structured namedtuple as __getitem__, with the following differences:
			- PIL Image (or raw bytes) as returned by self.reader function w/o any transforms
				vs.
			  torch.Tensor after all transformssx
			- target text label vs, target int label
			- image path
			- image catalog_number
		
		"""
		sample = self.parse_sample(index)
		path = getattr(sample, self.x_col)
		image_id = getattr(sample, self.id_col)
		
		image = self.reader(path)
		
		metadata={
			"path":path,
			"image_id":image_id
				  # "catalog_number":catalog_number
				 }
		label = -1
		if self.is_supervised:
			label = getattr(sample, self.y_col, -1)
		return image, label, metadata
		
	def __getitem__(self, index: int):
		
		image, label, metadata = self.fetch_item(index)
		
		# if self.preprocess is not None:
		# 	image = self.preprocess(image)
		
		if self.transform is not None:
			image = self.transform(image = image)
			image = image["image"]
		image = self.parse_output_image(image)
		
		return image, label, metadata






class BaseDataModule(pl.LightningDataModule):
	dataset_cls = None #Herbarium2022Dataset
	train_dataset = None
	val_dataset = None
	test_dataset = None
	
	train_transform = None
	val_transform = None
	test_transform = None
	
	transform_cfg = DEFAULT_TRANSFORM_CFG


	def prepare_data(self):
		pass

	def setup(self, stage=None):
		raise NotImplementedError


	def setup_transforms(self,
						 transform_cfg: dict=None,
						 train_transform=None,
						 val_transform=None,
						 test_transform=None,
						 remove_transforms: bool=False):
		transform_cfg = transform_cfg or {}
		self.transform_cfg = OmegaConf.merge(self.transform_cfg, transform_cfg)
		self.remove_transforms = remove_transforms
		if self.remove_transforms:
			for subset in ["train", "val", "test"]:
				setattr(self, f"{subset}_transform", None)
				if self.get_dataset(subset) is not None:
					self.get_dataset(subset).transform = None
			return
		else:
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


class Herbarium2022Dataset(BaseDataset):
	catalog_dir: str="/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize-512/catalogs" #/splits/train_size-0.8"
	default_cfg: Herbarium2022DatasetConfig = Herbarium2022DatasetConfig(catalog_dir=catalog_dir)


	def __init__(self,
				 catalog_dir: Optional[str]=None,
				 subset: str="train",
				 label_col: str="scientificName",
				 x_col: str="path",
				 y_col: str="y",
				 id_col: str="image_id",
				 smallest_taxon_col: str="Species",
				 train_size: float=0.8,
				 shuffle: bool=True,
				 seed: int=14,
				 image_reader: Union[Callable,str]="default", #Image.open,
				 preprocess: Callable=None,
				 transform: Callable=None,
				 output_image_type = torch.Tensor):

		"""
		
		Arguments:
		
			catalog_dir: Optional[str]=None,
				
			subset: str="train",
			
			label_col: str="family",
				Column containing the fully decoded str labels
			splits: float=(0.5,0.2,0.3),
			shuffle: bool=True,
			seed: int=14,
			image_reader: Union[Callable,str]="default", #Image.open,
			preprocess: Callable=None,
			transform: Callable=None
		
		"""
		# self.x_col = "path"
		# self.y_col = "y"
		# self.id_col = "catalog_number"

		super().__init__(catalog_dir=catalog_dir,
						 subset=subset,
						 label_col=label_col,
						 x_col=x_col,
						 y_col=y_col,
						 id_col=id_col,
						 smallest_taxon_col=smallest_taxon_col,
						 train_size=train_size,
						 shuffle=shuffle,
						 seed=seed,
						 image_reader=image_reader,
						 preprocess=preprocess,
						 transform=transform,
						 output_image_type=output_image_type)

		self.is_supervised = bool(subset != "test")
		self.setup()
		self.cfg = self.get_cfg()

	@classmethod
	def from_cfg(cls,
				 cfg: Union[Herbarium2022DatasetConfig, DictConfig],
				 **kwargs):
		if isinstance(cfg, Herbarium2022DatasetConfig):
			cfg = asdict(cfg)
		elif isinstance(cfg, DictConfig):
			cfg = OmegaConf.to_container(cfg, resolve=True)
		
		cfg = replace(self.default_cfg, **cfg)

		return cls(**cfg)


	def get_cfg(self, as_dict: bool=False) -> Herbarium2022DatasetConfig:
		cfg = replace(self.default_cfg, 
					  catalog_dir=self.catalog_dir,
					  subset=self.subset,
					  label_col=self.label_col,
					  x_col=self.x_col,
					  y_col=self.y_col,
					  id_col=self.id_col,
					  smallest_taxon_col=self.smallest_taxon_col,
					  train_size=self.train_size,
					  shuffle=self.shuffle,
					  seed=self.seed)
		if as_dict:
			return asdict(cfg)
		return cfg




class Herbarium2022DataModule(BaseDataModule):
	catalog_dir: str="/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize-512/catalogs" #/splits/train_size-0.8"
	dataset_cls = Herbarium2022Dataset
	transform_cfg = DEFAULT_TRANSFORM_CFG
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



	def setup(self, stage="fit"):
		subsets=[]
		if stage in ["train", "fit", "all", None]:
			self.train_dataset = self.dataset_cls(catalog_dir=self.catalog_dir,
													  subset="train",
													  label_col=self.label_col,
													  train_size=self.train_size,
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
													train_size=self.train_size,
													shuffle=self.shuffle,
													seed=self.seed,
													transform=self.val_transform)
			subsets.append("val")
		if stage in ["test", "all", None]:
			self.test_dataset = self.dataset_cls(catalog_dir=self.catalog_dir,
													 subset="test",
													 label_col=self.label_col,
													 train_size=self.train_size,
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
			smallest_taxon_col=self.smallest_taxon_col)


	@classmethod
	def from_cfg(cls,
				 cfg: Union[ExtantLeavesDataModuleConfig, DictConfig],
				 **kwargs):
		if isinstance(cfg, ExtantLeavesDatasetConfig):
			cfg = asdict(cfg)
		elif isinstance(cfg, DictConfig):
			cfg = OmegaConf.to_container(cfg, resolve=True)
		
		cfg = replace(self.default_cfg, **cfg)

		return cls(**cfg)


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


class ExtantLeavesDataset(BaseDataset):
	catalog_dir: str = "/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/Extant_Leaves_family_10_512/splits/splits=(0.5,0.2,0.3)"
	default_cfg: ExtantLeavesDatasetConfig = ExtantLeavesDatasetConfig(catalog_dir=catalog_dir)

	def __init__(self,
				 catalog_dir: Optional[str]=None,
				 subset: str="train",
				 label_col: str="family",
				 x_col: str="path",
				 y_col: str="y",
				 id_col: str="catalog_number",
				 smallest_taxon_col: str="Species",
				 splits: float=(0.5,0.2,0.3),
				 shuffle: bool=True,
				 seed: int=14,
				 image_reader: Union[Callable,str]="default", #Image.open,
				 preprocess: Callable=None,
				 transform: Callable=None,
				 output_image_type = torch.Tensor):
		"""
		
		Arguments:
		
			catalog_dir: Optional[str]=None,
				
			subset: str="train",
			
			label_col: str="family",
				Column containing the fully decoded str labels
			splits: float=(0.5,0.2,0.3),
			shuffle: bool=True,
			seed: int=14,
			image_reader: Union[Callable,str]="default", #Image.open,
			preprocess: Callable=None,
			transform: Callable=None
		
		"""
		# self.x_col = "path"
		# self.y_col = "y"
		# self.id_col = "catalog_number"
		self.splits = splits
		self.smallest_taxon_col="Species"
		super().__init__(catalog_dir=catalog_dir,
						 subset=subset,
						 label_col=label_col,
						 x_col=x_col,
						 y_col=y_col,
						 id_col=id_col,
						 smallest_taxon_col=smallest_taxon_col,
						 train_size=None,
						 shuffle=shuffle,
						 seed=seed,
						 image_reader=image_reader,
						 preprocess=preprocess,
						 transform=transform,
						 output_image_type=output_image_type)
		# super().__init__()
		# self.x_col = "path"
		# self.y_col = "y"
		# self.id_col = "catalog_number"

		# self.catalog_dir = catalog_dir or self.catalog_dir
		# self.label_col = label_col
		# self.splits = splits
		# self.shuffle = shuffle
		# self.seed = seed
		# self.subset = subset
		self.is_supervised = bool(subset != "test")
		# self.set_image_reader(image_reader)
		# self.preprocess = preprocess
		# self.transform = transform
		self.setup()
		self.cfg = self.get_cfg()

				
	@classmethod
	def from_cfg(cls,
				 cfg: Union[ExtantLeavesDatasetConfig, DictConfig],
				 **kwargs):
		if isinstance(cfg, ExtantLeavesDatasetConfig):
			cfg = asdict(cfg)
		elif isinstance(cfg, DictConfig):
			cfg = OmegaConf.to_container(cfg, resolve=True)
		
		cfg = replace(self.default_cfg, **cfg)

		return cls(**cfg)


	def get_cfg(self, as_dict: bool=False) -> ExtantLeavesDatasetConfig:
		cfg = replace(self.default_cfg, 
					  catalog_dir=self.catalog_dir,
					  subset=self.subset,
					  label_col=self.label_col,
					  x_col=self.x_col,
					  y_col=self.y_col,
					  id_col=self.id_col,
					  smallest_taxon_col=self.smallest_taxon_col,
					  splits=self.splits,
					  shuffle=self.shuffle,
					  seed=self.seed)
		if as_dict:
			return asdict(cfg)
		return cfg


	@property
	def splits_dir(self) -> Path:
		return leavesdb_utils.find_data_splits_dir(source_dir=self.catalog_dir,
												   splits=self.splits)
	
	@property
	def split_file_path(self) -> Path:
		"""
		Should return the path of this subset's on-disk csv catalog file. I emphasize should.
		"""
		return self.splits_dir / f"{self.subset}_metadata.csv"
	
	
	@property
	def already_built(self) -> bool:
		return leavesdb_utils.check_already_built(self.splits_dir, label_col=self.label_col)

	
	def prepare_metadata(self):

		if not self.already_built:
			
			data = make_train_val_test_splits(source_dir=self.catalog_dir,
										 splits_dir=self.splits_dir,
										 label_col=self.label_col,
										 splits=args.splits,
										 seed=self.seed)
			return data


	def get_data_subset(self,
						subset: str="train") -> Tuple["LabelEncoder", pd.DataFrame]:
		"""
		Read the selected data subset into a pd.DataFrame
		
		Returns a Tuple containing an sklearn LabelEncoder and a pd.DataFrame of the subset's data catalog
		"""
		data = leavesdb_utils.read_encoded_splits(source_dir=self.splits_dir,
												  include=[subset],
												  label_col=self.label_col,
                                                  index_col=0)
		encoder = data["label_encoder"]
		data = data["subsets"][subset]
		
		return encoder, data

	
# 	def setup(self):
# 		"""
# 		Assigns the following instance attributes:
		
# 			::self.label_encoder
# 			::self.df
# 			::self.paths
# 			::self.targets
# 			::self.num_classes
# 		"""
# 		data = self.prepare_metadata()
# 		if data is None:
# 			encoder, data = self.get_data_subset(subset=self.subset)
# 		else:
# 			encoder = data["label_encoder"]
# 			data = data["subsets"][self.subset]

# 		if isinstance(encoder, preprocessing.LabelEncoder):
# 			"""
# 			Auto wraps any sklearn LabelEncoder in our custom class.
# 			(Added 2022-03-25 - untested)
# 			"""
# 			encoder = label_utils.LabelEncoder.from_sklearn(encoder)

# 		setattr(data, "label_encoder", encoder)
# 		self.label_encoder = encoder
# 		self.df = data
		
# 		if self.shuffle:
# 			self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=False)

# 		if self.id_col in self.df.columns:
# 			self.image_ids = self.df[self.id_col]
# 		self.paths = self.df[self.x_col]

# 		if self.is_supervised:
# 			# self.imgs = 
# 			self.targets = self.df[self.y_col]
# 			self.num_classes = len(set(self.df[self.y_col]))
# 		else:
# 			self.targets = None
# 			self.num_classes = -1 # Or 0?


class ExtantLeavesDataModule(BaseDataModule):
	catalog_dir: str="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/extant_leaves_family_3_512/splits/splits=(0.5,0.2,0.3)"
	dataset_cls = ExtantLeavesDataset	
	transform_cfg = DEFAULT_TRANSFORM_CFG	
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


	@classmethod
	def from_cfg(cls,
				 cfg: Union[ExtantLeavesDataModuleConfig, DictConfig],
				 **kwargs):
		if isinstance(cfg, ExtantLeavesDatasetConfig):
			cfg = asdict(cfg)
		elif isinstance(cfg, DictConfig):
			cfg = OmegaConf.to_container(cfg, resolve=True)
		
		cfg = replace(self.default_cfg, **cfg)

		return cls(**cfg)


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



	def setup(self, stage=None):
		subsets=[]
		if stage in ["train", "fit", "all", None]:
			self.train_dataset = self.dataset_cls(catalog_dir=self.catalog_dir,
													  subset="train",
													  label_col=self.label_col,
													  splits=self.splits,
													  shuffle=self.shuffle,
													  seed=self.seed,
													  transform=self.train_transform)
			# self.setup_taxonomy_table(df=self.train_dataset.df,
			# 						  smallest_taxon_col=self.smallest_taxon_col)
			subsets.append("train")
		if stage in ["val", "fit", "all", None]:
			self.val_dataset = self.dataset_cls(catalog_dir=self.catalog_dir,
													subset="val",
													label_col=self.label_col,
													splits=self.splits,
													shuffle=self.shuffle,
													seed=self.seed,
													transform=self.val_transform)
			subsets.append("val")
		if stage in ["test", "all", None]:
			self.test_dataset = self.dataset_cls(catalog_dir=self.catalog_dir,
													 subset="test",
													 label_col=self.label_col,
													 splits=self.splits,
													 shuffle=self.shuffle,
													 seed=self.seed,
													 transform=self.test_transform)
			subsets.append("test")
			
		# for s in subsets:
		# 	self.get_dataset_size(subset=s,
		# 						  verbose=True)
			
		self.set_image_reader(self.image_reader)
		
		self.setup_taxonomy_table(df=self.train_dataset.df,
								  smallest_taxon_col=self.smallest_taxon_col)




	
	
	
	
	# catalog_dir: str="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1/Fossil_family_10_512/splits/splits=(0.5,0.2,0.3)"


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

















