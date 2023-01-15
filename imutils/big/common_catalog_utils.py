"""

imutils/big/common_catalog_utils.py

Created On: Tuesday April 5th, 2022  
Created By: Jacob A Rose


"""

import collections
from functools import cached_property
from omegaconf import DictConfig
import os
import pandas as pd
from pathlib import Path
from typing import *
from dataclasses import dataclass, asdict
from imutils.ml.utils.etl_utils import ETL

import torchvision
from imutils.utils import torchdata
from imutils.ml.utils import template_utils
from imutils.ml.utils.label_utils import LabelEncoder

from imutils import catalog_registry

log = template_utils.get_logger(__name__)

__all__ = ['PathSchema', 'SampleSchema', 'Batch', 'CSVDatasetConfig', 'CSVDataset', "ImageFileDataset", "ImageFileDatasetConfig", "DataETL"]

# __all__ += ["DataSplitter"]


@dataclass 
class PathSchema:
	path_schema: str = Path("{family}_{genus}_{species}_{collection}_{catalog_number}")
		
	def __init__(self,
				 path_schema,
				 sep: str="_"):

		self.sep = sep
		self.schema_parts: List[str] = path_schema.split(sep)
		self.maxsplit: int = len(self.schema_parts) - 2
	
	def parse(self, path: Union[Path, str], sep: str="_"):
	
		parts = Path(path).stem.split(sep, maxsplit=self.maxsplit)
		if len(parts) == 5:
			family, genus, species, collection, catalog_number = parts
		elif len(parts) == 4:
			family, genus, species, catalog_number = parts
			collection = catalog_number.split("_")[0]
		else:
			print(f'len(parts)={len(parts)}, parts={parts}, path={path}')

		return family, genus, species, collection, catalog_number
	
	def split(self, sep):
		return self.schema_parts


@dataclass
class SampleSchema:
	path : Union[str, Path] = None
	family : str = None
	genus : str = None
	species : str = None
	collection : str = None
	catalog_number : str = None

	@classmethod
	def keys(cls):
		return list(cls.__dataclass_fields__.keys())
		
	def __getitem__(self, index: int):
		return getattr(self, self.keys()[index])



# Batch = namedtuple("Batch", ("image", "target", "path", "catalog_number"))
totensor: Callable = torchvision.transforms.ToTensor()
toPIL: Callable = torchvision.transforms.ToPILImage("RGB")



class DataETL(ETL):
	
	@classmethod
	def export_dataset_state(cls,
							 output_dir: Union[str, Path],
							 df: pd.DataFrame=None,
							 encoder: LabelEncoder=None,
							 dataset_name: Optional[str]="dataset",
							 config: "CSVDatasetConfig"=None
							 ) -> None:
		
		paths = {ftype: str(output_dir / str(dataset_name + ext)) for ftype, ext in cls.data_file_ext_maps.items()}
		
		output_dir = Path(output_dir)
		if isinstance(df, pd.DataFrame):
			cls.df2csv(df = df,
					   path = paths["df"])
			if config:
				config.data_path = paths["df"]
		if isinstance(encoder, LabelEncoder):
			cls.labels2json(encoder=encoder,
							path = paths["encoder"])
			if config:
				config.label_encoder_path = paths["encoder"]
		if isinstance(config, CSVDatasetConfig):
			config.save(path = paths["config"])
#			 cls.config2yaml(config=config,
#							 path = paths["config"])
			
			
	@classmethod
	def import_dataset_state(cls,
							 data_dir: Optional[Union[str, Path]]=None,
							 config_path: Optional[Union[Path, str]]=None,
							 **kwargs
							) -> Tuple["CSVDataset", "CSVDatasetConfig"]:
		if (
			(not os.path.exists(str(data_dir))) 
			and (not os.path.exists(config_path))
		):
			raise ValueError("Either data_dir or config_path must be existing paths")
		
		if os.path.isdir(str(data_dir)):
			data_dir = Path(data_dir)
		paths = {}
		
		
		if not os.path.isfile(str(config_path)):
			config_path = data_dir / "CSVDataset-config.yaml"		
		assert os.path.isfile(str(config_path)), f"Couldn't find config: {config_path=}"
		paths['config'] = config_path
#			 config = cls.config_from_yaml(path = paths["config"])
		config = CSVDatasetConfig.load(path = paths["config"])
		if hasattr(config, "data_path"):
			paths["df"] = str(config.data_path)
		if hasattr(config, "label_encoder_path"):
			paths["encoder"] = str(config.label_encoder_path)
		data_dir = Path(os.path.dirname(config_path))
			
		for ftype, ext in cls.data_file_ext_maps.items():
			if ftype not in paths:
				paths[ftype] = str(list(data_dir.glob("*" + ext))[0])
				
				
		config.data_path = str(paths["df"])
		config.label_encoder_path = str(paths["encoder"])
		label_encoder = None
		if os.path.isfile(paths["encoder"]):
			# import label encodings json file if it exists
			label_encoder = cls.labels_from_json(path = paths["encoder"])
			
		# import dataset samples from a csv file as a CustomDataset/CSVDataset object
		dataset = CSVDataset.from_config(config,
										 eager_encode_targets=False) # True)
		dataset.setup(samples_df=dataset.samples_df,
					  label_encoder=label_encoder,
					  fit_targets=True)
		
		return dataset, config






@dataclass
class BaseConfig:

	def save(self,
			 path: Union[str, Path]) -> None:
		"""
		Save current config object's info into a yaml file.
		"""
		
		cfg = asdict(self)
#		 cfg = DictConfig({k: getattr(self,k) for k in self.keys()})
		ETL.config2yaml(cfg, path)
	
	@classmethod
	def load(cls,
			 path: Union[str, Path]) -> "DatasetConfig":
		"""
		Load current config object's info from a yaml file.
		"""
		cfg = ETL.config_from_yaml(path)

#		 keys = cls.__dataclass_fields__.keys()
		cfg = cls(**{k: cfg[k] for k in cls.keys()})
		return cfg
	
	@classmethod
	def keys(cls):
		"""
		Return the default dataclass fields of this type of config as strings.
		"""
		return cls.__dataclass_fields__.keys()
	
	def __repr__(self):
		out = f"{type(self)}" + "\n"
		out += "\n".join([f"{k}: {getattr(self, k)}" for k in self.keys()])
#		 out += f"\nroot_dir: {self.root_dir}"
#		 out += "\nsubset_dirs: \n\t" + '\n\t'.join(self.subset_dirs)
		return out

	
@dataclass
class DatasetConfig(BaseConfig):
	base_dataset_name: str = "" # "Extant_Leaves"
	class_type: str = "family"
	threshold: Optional[int] = 10
	resolution: int = 512
	version: str = "v1_0"
	path_schema: str = "{family}_{genus}_{species}_{collection}_{catalog_number}"
	
	def __post_init__(self):
		assert self.version in self.available_versions
	
	@property
	def available_versions(self) -> List[str]:
		return list(catalog_registry.available_datasets.versions.keys())

	@property
	def full_name(self) -> str:
		name = []
		if len(self.base_dataset_name):
			name.append(self.base_dataset_name)
		if self.threshold:
			name.extend([str(self.class_type), str(self.threshold)])
		name.append(str(self.resolution))
		return "_".join(name)
#		 name  = self.base_dataset_name
#		 if self.threshold:
#			 name += f"_{self.class_type}_{self.threshold}"
#		 name += f"_{self.resolution}"
#		 return name

	
class ImageFileDatasetConfig(DatasetConfig):	
	@property
	def root_dir(self):
		return catalog_registry.available_datasets.get(self.full_name, version=self.version)
	
	def is_valid_subset(self, subset: str):
		for s in ("train", "val", "test", "train_images", "test_images:"):
			if s in subset:
				return True
		return False
	
	@property
	def subsets(self):
		if isinstance(self.root_dir, list):
			return []
		return [s for s in os.listdir(self.root_dir) if self.is_valid_subset(s)]
	
	@property
	def subset_dirs(self):
		return [os.path.join(self.root_dir, subset) for subset in self.subsets]

	def locate_files(self) -> Dict[str, List[Path]]:
		return ETL.locate_files(self.root_dir)

	@cached_property
	def num_samples(self):
#		 subset_dirs = {Path(subset_dir).stem: Path(subset_dir) for subset_dir in self.subset_dirs}
		files = {subset: f for subset, f in self.locate_files().items() if self.is_valid_subset(subset)}
		return {subset: len(list(f)) for subset, f in files.items()}
	
	def __repr__(self):
		out = super().__repr__()
		out += f"\nroot_dir: {self.root_dir}"
		out += "\nsubsets: "
		for i, subset in enumerate(self.subsets):
			out += '\n\t' + f"{subset}:"
			out += '\n\t\t' + f"subdir: {self.subset_dirs[i]}"
			out += '\n\t\t' + f"subset_num_samples: {self.num_samples[subset]}"
		return out

@dataclass
class CSVDatasetConfig(BaseConfig):
	"""
	Represents a single data subset, or the set of "all" data.
	
	"""
	full_name: str = None
	data_path: str = None
	label_encoder_path: Optional[str] = None
	subset_key: str = "all"
	
	def update(self, **kwargs) -> None:
		if "subset_key" in kwargs:
			self.subset_key = kwargs["subset_key"]
		if "num_samples" in kwargs:
			self.num_samples = {self.subset_key: kwargs["num_samples"]}
	
	@cached_property
	def num_samples(self) -> Dict[str,int]:
		return {self.subset_key: len(self.locate_files())}

	def __repr__(self):
		out = super().__repr__()
		out += '\n' + f"num_samples: {self.num_samples[self.subset_key]}"
		return out

	def locate_files(self) -> pd.DataFrame:
		return ETL.df_from_csv(self.data_path)
	
	def load_label_encoder(self) -> Union[None, LabelEncoder]:
		if os.path.exists(str(self.label_encoder_path)):
			return ETL.labels_from_json(str(self.label_encoder_path))
		return

	@classmethod
	def export_dataset_state(cls,
							 df: pd.DataFrame,
							 output_dir: Union[str, Path],
							 config: DictConfig=None,
							 encoder: LabelEncoder=None,
							 dataset_name: Optional[str]="dataset"
							 ) -> None:
		ETL.export_dataset_state(output_dir=output_dir,
									 df=df,
									 config=config,
									 encoder=encoder,
									 dataset_name=dataset_name)
			
	@classmethod
	def import_dataset_state(cls,
							 data_dir: Optional[Union[str, Path]]=None,
							 config_path: Optional[Union[Path, str]]=None,
							) -> Tuple["CSVDataset", "CSVDatasetConfig"]:

		return ETL.import_dataset_state(data_dir=data_dir,
											config_path=config_path)

##############





class CustomDataset(torchdata.datasets.Files): # (CommonDataset):

	def __init__(self,
				 files: List[Path]=None,
				 samples_df: pd.DataFrame=None,
				 path_schema: Path = "{family}_{genus}_{species}_{collection}_{catalog_number}",
				 batch_fields: List[str] = ["image","target", "path", "catalog_number"],
				 eager_encode_targets: bool = False,
				 config: Optional[BaseConfig]=None,
				 transform=None,
				 target_transform=None):
		files = files or []
		super().__init__(files=files)
		self.path_schema = PathSchema(path_schema)
		self.Batch = collections.namedtuple("Batch", batch_fields, defaults = (None,)*len(batch_fields))
		
		self.x_col = "path"
		self.y_col = "family"
		self.id_col = "catalog_number"
		self.eager_encode_targets = eager_encode_targets
		self.config = config or {}
		self.transform = transform
		self.target_transform = target_transform
		self.setup(samples_df=samples_df)
		
		
	def fetch_item(self, index: int) -> Tuple[str]:
		sample = self.parse_sample(index)
		image = Image.open(sample.path)
		return self.Batch(image=image,
						  target=getattr(sample, self.y_col),
						  path=getattr(sample, self.x_col),
						  catalog_number=getattr(sample, self.id_col))


	def __getitem__(self, index: int):
		
		item = self.fetch_item(index)
		image, target, path, catalog_number = item.image, item.target, item.path, item.catalog_number
		target = self.label_encoder.class2idx[target]
		
		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return self.Batch(image=image,
						  target=target,
						  path=path,
						  catalog_number=catalog_number)
		
	def setup(self,
			  samples_df: pd.DataFrame=None,
			  label_encoder: LabelEncoder=None,
			  fit_targets: bool=True):
		"""
		Running setup() should result in the Dataset having assigned values for:
			self.samples
			self.targets
			self.samples_df
			self.label_encoder
		
		"""
		if samples_df is not None:
			self.samples_df = samples_df.convert_dtypes()
		self.samples = [self.parse_sample(idx) for idx in range((len(self)))]
		self.targets = [sample[1] for sample in self.samples]
		self.samples_df = pd.DataFrame(self.samples).convert_dtypes()
		
		self.label_encoder = label_encoder or LabelEncoder()
		if fit_targets:
			self.label_encoder.fit(self.targets)
			
		if self.eager_encode_targets:
			self.targets = self.label_encoder.encode(self.targets).tolist()
		
	@classmethod
	def from_config(cls, config: DatasetConfig, subset_keys: List[str]=None) -> "CustomDataset":
		pass
		
	def parse_sample(self, index: int):
		pass
	
	@property
	def classes(self):
		return self.label_encoder.classes
	
	def __repr__(self):
		disp = f"""<{str(type(self)).strip("'>").split('.')[1]}>:"""
		disp += '\n\t' + self.config.__repr__().replace('\n','\n\t')
		return disp

	
	@classmethod
	def get_files_from_samples(cls,
							   samples: Union[pd.DataFrame, List],
							   x_col: Optional[str]="path"):
		if isinstance(samples, pd.DataFrame):
			if x_col in samples.columns:
				files = list(samples[x_col].values)
			else:
				files = list(samples.iloc[:,0].values)
		elif isinstance(samples, list):
			files = [s[0] for s in self.samples]
			
		return files
	
	def intersection(self, other, suffixes=("_x","_y")):
		samples_df = self.samples_df
		other_df = other.samples_df
		
		intersection = samples_df.merge(other_df, how='inner', on=self.id_col, suffixes=suffixes)
		return intersection
	
	def __add__(self, other):
	
		intersection = self.intersection(other)[self.id_col].tolist()
		samples_df = self.samples_df
		
		left_union = samples_df[samples_df[self.id_col].apply(lambda x: x in intersection)]
		
		return left_union
	
	def __sub__(self, other):
	
		intersection = self.intersection(other)[self.id_col].tolist()
		samples_df = self.samples_df
		
		remainder = samples_df[samples_df[self.id_col].apply(lambda x: x not in intersection)]
		
		return remainder
	
	def filter(self, indices, subset_key: Optional[str]="all"):
		out = type(self)(samples_df = self.samples_df.iloc[indices,:],
						 config = deepcopy(self.config))
		out.config.update(subset_key=subset_key,
						  num_samples=len(out))
		return out
	
	def get_unsupervised(self):
		return UnsupervisedDatasetWrapper(self)



	
class ImageFileDataset(CustomDataset):
	
	@classmethod
	def from_config(cls, config: DatasetConfig, subset_keys: List[str]=None) -> "CustomDataset":
		files = config.locate_files()
		if isinstance(subset_keys, list):
			files = {k: files[k] for k in subset_keys}
		if len(files.keys())==1: 
			files = files[subset_keys[0]]
		new = cls(files=files,
				  path_schema=config.path_schema)
		new.config = config
		return new
	
	def parse_sample(self, index: int):
		path = self.files[index]
		family, genus, species, collection, catalog_number = self.path_schema.parse(path)

		return SampleSchema(path=path,
							family=family,
							genus=genus,
							species=species,
							collection=collection,
							catalog_number=catalog_number)




class CSVDataset(CustomDataset):
	
	@classmethod
	def from_config(cls,
					config: DatasetConfig, 
					subset_keys: List[str]=None,
					eager_encode_targets: bool=False) -> Union[Dict[str, "CSVDataset"], "CSVDataset"]:
		
		files_df = config.locate_files()
		if subset_keys is None:
			subset_keys = ['all']
		if isinstance(subset_keys, list) and isinstance(files_df, dict):
			files_df = {k: files_df[k] for k in subset_keys}
			new = {k: cls(samples_df=files_df[k],  \
						  eager_encode_targets=eager_encode_targets) for k in subset_keys}
			for k in subset_keys:
				new[k].config = deepcopy(config)
				new[k].config.subset_key = k

		if len(subset_keys)==1:
			if isinstance(files_df, dict):
				files_df = files_df[subset_keys[0]]
			new = cls(samples_df=files_df, 
					  eager_encode_targets=eager_encode_targets)
			new.config = config
			new.config.subset_key = subset_keys[0]
		return new
	
	def setup(self,
			  samples_df: pd.DataFrame=None,
			  label_encoder: LabelEncoder=None,
			  fit_targets: bool=True):
		
		if samples_df is not None:
			self.samples_df = samples_df.convert_dtypes()
		self.files = self.samples_df[self.x_col].apply(lambda x: Path(x)).tolist()
		super().setup(samples_df=self.samples_df,
					  label_encoder=label_encoder,
					  fit_targets=fit_targets)


	def parse_sample(self, index: int):
		
		row = self.samples_df.iloc[index,:].tolist()
		path, family, genus, species, collection, catalog_number = row
		return SampleSchema(path=path,
							 family=family,
							 genus=genus,
							 species=species,
							 collection=collection,
							 catalog_number=catalog_number)




##################
##################







def export_dataset_catalog_configuration(
	output_dir: str = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0",
	base_dataset_name="Extant_Leaves",
	threshold=100,
	resolution=512,
	version: str = "v1_0",
	path_schema: str = "{family}_{genus}_{species}_{collection}_{catalog_number}",
):
	"""
	Produces an output csv catalog containing all available metadata about an on-disk image dataset.
	
	
	Refactored on (2022-04-05)
	
	"""

	image_file_config = ImageFileDatasetConfig(
		base_dataset_name=base_dataset_name,
		class_type="family",
		threshold=threshold,
		resolution=resolution,
		version=version,
		path_schema=path_schema,
	)

	out_dir = os.path.join(output_dir, image_file_config.full_name)
	os.makedirs(out_dir, exist_ok=True)

	csv_out_path = os.path.join(out_dir, f"{image_file_config.full_name}-full_dataset.csv")
	image_file_config_out_path = os.path.join(out_dir, "ImageFileDataset-config.yaml")
	csv_config_out_path = os.path.join(out_dir, "CSVDataset-config.yaml")

	dataset = ImageFileDataset.from_config(image_file_config, subset_keys=["all"])
	ETL.df2csv(dataset.samples_df, path=csv_out_path)
	image_file_config.save(image_file_config_out_path)

	csv_config = CSVDatasetConfig(
		full_name=image_file_config.full_name, data_path=csv_out_path, subset_key="all"
	)

	csv_config.save(csv_config_out_path)

	print(f"[FINISHED] DATASET FULL NAME: {csv_config.full_name}")
	print(f"Newly created dataset assets located at:  {out_dir}")

	return dataset, image_file_config, csv_config