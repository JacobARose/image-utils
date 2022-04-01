"""
imutils/ml/data/datamodule.py

Created on: Wednesday March 16th, 2022  
Created by: Jacob Alexander Rose  


- Update (Wednesday March 24th, 2022)
	- Refactored to make more modular BaseDataset and BaseDataModule, which Herbarium2022Dataset and Herbarium2022DataModule inherit, respectively.

"""

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
from typing import *


# from imutils.big.make_train_val_splits import main as make_train_val_splits
from imutils.big.make_train_val_splits import (check_already_built,
											   read_encoded_splits,
											   find_data_splits_dir,
											   main as make_train_val_splits)

# from imutils.big.transforms.image import (Preprocess,
from imutils.ml.aug.image.images import (Preprocess,
										 BatchTransform,
										 get_default_transforms,
										 DEFAULT_CFG as DEFAULT_TRANSFORM_CFG)

from imutils.ml.utils import label_utils

__all__ = ["Herbarium2022DataModule", "Herbarium2022Dataset", "get_default_transforms"]



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



class AbstractCatalogDataset(Dataset):

	def __init__(self):
		super().__init__()
		"""
		TBD: document these function kwargs
		
		"""
		self.x_col = "path"
		self.y_col = "y"
		self.id_col = "image_id"
	
	
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
		return check_already_built(self.splits_dir)

	
	def prepare_metadata(self):

		if not self.already_built:
			
			data = make_train_val_splits(source_dir=self.catalog_dir,
										 splits_dir=self.splits_dir,
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
											include=[subset])
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
	

	def get_decoded_targets(self, with_image_ids: bool=True) -> Tuple[str, np.ndarray]:
		assert self.is_supervised
		return self.label_encoder.inv_transform(self.targets)
	

class BaseDataset(AbstractCatalogDataset):
	catalog_dir: str = os.path.abspath("./data")
	
	def __init__(self,
				 catalog_dir: Optional[str]=None,
				 subset: str="train",
				 label_col: str="scientificName",
				 train_size: float=0.7,
				 shuffle: bool=True,
				 seed: int=14,
				 image_reader: Union[Callable,str]="default", #Image.open,
				 preprocess: Callable=None,
				 transform: Callable=None):
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
		super().__init__()
		
		self.catalog_dir = catalog_dir or self.catalog_dir

		self.label_col = label_col
		self.train_size = train_size
		self.shuffle = shuffle
		self.seed = seed
		self.subset = subset
		self.is_supervised = bool(subset != "test")

		self.set_image_reader(image_reader)
		self.preprocess = preprocess
		self.transform = transform
		self.setup()
				
	@classmethod
	def from_cfg(cls,
				 cfg: DictConfig,
				 **kwargs):
		cfg = OmegaConf.merge(cfg, kwargs)
		return cls(**cfg)
		
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
		
		if self.preprocess is not None:
			image = self.preprocess(image)
		
		if self.transform is not None:
			image = self.transform(image)
		
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

	def __init__(self,
				 catalog_dir: Optional[str]=None,
				 label_col="scientificName",
				 train_size=0.7,
				 shuffle: bool=True,
				 seed=14,
				 batch_size: int = 128,
				 num_workers: int = None,
				 pin_memory: bool=True,
				 train_transform=None,
				 val_transform=None,
				 test_transform=None,
				 transform_cfg=None,
				 remove_transforms: bool=False,
				 image_reader: Callable="default", #Image.open,
				 **kwargs
	):
		super().__init__()
		
		self.catalog_dir = catalog_dir
		
		self.label_col = label_col
		self.train_size = train_size
		self.shuffle = shuffle
		self.seed = seed
		self.batch_size = batch_size
		self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()
		self.pin_memory = pin_memory
		self.image_reader = image_reader

		self.setup_transforms(transform_cfg=transform_cfg,
							  train_transform=train_transform,
							  val_transform=val_transform,
							  test_transform=test_transform,
							  remove_transforms=remove_transforms)
		self.cfg = self.get_cfg()
		self.kwargs = kwargs
		

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
			label_col=self.label_col or "scientificName",
			train_size=self.train_size or 0.7,
			shuffle=self.shuffle,
			seed=self.seed or 14,
			batch_size = self.batch_size or 128,
			num_workers = self.num_workers or None,
			pin_memory=self.pin_memory,
			transform_cfg=self.transform_cfg,
			remove_transforms=self.remove_transforms,
		))
		
		cfg = OmegaConf.merge(default_cfg, cfg, kwargs)
		return cfg


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
			self.train_transform = None
			self.val_transform = None
			self.test_transform = None
		else:
			print("self.transform_cfg:"); pp(self.transform_cfg)
			self.train_transform = (
				get_default_transforms(mode="train", config=self.transform_cfg)
				if train_transform is None else train_transform
			)
			self.val_transform = (
				get_default_transforms(mode="val", config=self.transform_cfg)
				if val_transform is None else val_transform
			)
			self.test_transform = (
				get_default_transforms(mode="test", config=self.transform_cfg)
				if test_transform is None else test_transform
			)


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


	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=True,
			pin_memory=self.pin_memory
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,#*2,
			num_workers=self.num_workers,
			shuffle=False,
			pin_memory=self.pin_memory
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
			ic(subset, num_samples, num_batches, self.num_classes, self.batch_size)
		return num_samples, num_batches


	



	def show_batch(self, win_size=(10, 10)):
		def _to_vis(data):
			return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

		# get a batch from the training set: try with `val_datlaoader` :)
		imgs, labels = next(iter(self.train_dataloader()))
		imgs_aug = self.transform(imgs)  # apply transforms
		# use matplotlib to visualize
		plt.figure(figsize=win_size)
		plt.imshow(_to_vis(imgs))
		plt.figure(figsize=win_size)
		plt.imshow(_to_vis(imgs_aug))
		


		
		
		


class Herbarium2022Dataset(BaseDataset):
	catalog_dir: str = os.path.abspath("./data")



class Herbarium2022DataModule(BaseDataModule):
	dataset_cls = Herbarium2022Dataset	
	transform_cfg = DEFAULT_TRANSFORM_CFG

	def setup(self, stage=None):
		subsets=[]
		if stage in ["train", "fit", "all", None]:
			self.train_dataset = Herbarium2022Dataset(catalog_dir=self.catalog_dir,
													  subset="train",
													  label_col=self.label_col,
													  train_size=self.train_size,
													  shuffle=self.shuffle,
													  seed=self.seed,
													  transform=self.train_transform)
			subsets.append("train")
		if stage in ["val", "fit", "all", None]:
			self.val_dataset = Herbarium2022Dataset(catalog_dir=self.catalog_dir,
													subset="val",
													label_col=self.label_col,
													train_size=self.train_size,
													shuffle=self.shuffle,
													seed=self.seed,
													transform=self.val_transform)
			subsets.append("val")
		if stage in ["test", "all", None]:
			self.test_dataset = Herbarium2022Dataset(catalog_dir=self.catalog_dir,
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

