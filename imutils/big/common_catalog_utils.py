"""

imutils/big/common_catalog_utils.py

Created On: Tuesday April 5th, 2022  
Created By: Jacob A Rose


"""

from functools import cached_property
from omegaconf import DictConfig
import os
import pandas as pd
from pathlib import Path
from typing import *
from dataclasses import dataclass, asdict
from imutils.ml.utils.etl_utils import ETL

import torchvision
from imutils.ml.utils import template_utils
from imutils.ml.utils.label_utils import LabelEncoder

from imutils import catalog_registry

log = template_utils.get_logger(__name__)

__all__ = ['PathSchema', 'SampleSchema', 'Batch', 'CSVDatasetConfig', 'CSVDataset', "DataETL"]

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
#             cls.config2yaml(config=config,
#                             path = paths["config"])
            
            
    @classmethod
    def import_dataset_state(cls,
                             data_dir: Optional[Union[str, Path]]=None,
                             config_path: Optional[Union[Path, str]]=None,
                            ) -> Tuple["CSVDataset", "CSVDatasetConfig"]:
        if (not os.path.exists(str(data_dir))) and (not os.path.exists(config_path)):
            raise ValueError("Either data_dir or config_path must be existing paths")
        
        if os.path.isdir(str(data_dir)):
            data_dir = Path(data_dir)
        paths = {}
        
        # import config yaml file
        if os.path.isfile(str(config_path)):
            paths['config'] = config_path
#             config = cls.config_from_yaml(path = paths["config"])
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