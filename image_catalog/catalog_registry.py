"""

image-utils/image_catalog/catalog_registry.py

Importable registry of directories containing different datasets and versions mounted on data_cifs.

## Example 1: Search
-------

from imutils.catalog_registry import available_datasets

>> catalog_registry.available_datasets.search("General_Fossil_family_10")

{'General_Fossil_family_10_512': '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/512/10/jpg',
 'General_Fossil_family_10_1024': '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1024/10/jpg',
 'General_Fossil_family_10_1536': '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1536/10/jpg',
 'General_Fossil_family_10_2048': '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/2048/10/jpg'}


---------------------------
## Example 2: Query an image dataset dir and construct image path & label vectors
-------

img_root_dir = catalog_registry.available_datasets.get("General_Fossil_family_10_512")

class_names = sorted(os.listdir(img_root_dir))

path_list = []; label_list = []; label_str2int = {}

for i, n in enumerate(class_names):
    i_paths = sorted(os.listdir(
        os.path.join(img_root_dir,n)))
    i_paths = [os.path.join(img_root_dir, n, fname) for fname in i_paths]
    i_labels = [i]*len(i_paths)
    path_list.extend(i_paths)
    label_list.extend(i_labels)
    label_str2int[n] = i

name_list = [class_names[l] for l in label_list]

data = [(i, l) for i, l in zip(path_list, label_list)]

---------------------------
## Example 3: Plot images grouped into class tabs
-------


class_counts = Counter(name_list)
tab_labels = [f"{l}: {class_counts[l]}" for l in name_list]


ipyplot.plot_class_tabs(path_list, tab_labels)

---------------------------


2. Previously:
image-utils/imutils/catalog_registry.py
---------------------------
1. Previously:
lightning_hydra_classifiers/data/utils/catalog_registry.py

Author: Jacob A Rose
Created: Sunday August 1st, 2021


Currently covers:
	- leavesdb v0_3
	- leavesdb v1_0
	- leavesdb v1_1
Work In Progress:
	- leavesdb v1_1 -- Blurred Text (Added 2022-04-03)
----------------------------
TODO:

[] Implement file-system agnostic dataset registration for
	[] 1. automatically downloading and caching from GCS to local system
	[] 2. Deploying catalogs from central data_cifs repository to GCS for future querying (step 1.)
[] 3. Create a `test_catalog_registry.py` script with pytest fixtures to efficiently describe every combination of possible inputs to AvailableDataset's main methods, including
	* a. AvailableDatasets.search
	* b. AvailableDatasets.query_tags
	* c. AvailableDatasets.get
	* d. AvailableDatasets.get_latest
	* e. AvailableDatasets.add_dataset


-- (Sunday 2022-06-12) Added the following 40 (!) dataset variations to v1_1:

	Fossil_family_3_512
	Fossil_family_3_1024
	Fossil_family_3_1536
	Fossil_family_3_2048
	
	Fossil_family_10_512
	Fossil_family_10_1024
	Fossil_family_10_1536
	Fossil_family_10_2048

	Fossil_family_20_512
	Fossil_family_20_1024
	Fossil_family_20_1536
	Fossil_family_20_2048

	Fossil_family_50_512
	Fossil_family_50_1024
	Fossil_family_50_1536
	Fossil_family_50_2048


	General_Fossil_family_10_512
	General_Fossil_family_10_1024
	General_Fossil_family_10_1536
	General_Fossil_family_10_2048
	Florissant_Fossil_family_10_512
	Florissant_Fossil_family_10_1024
	Florissant_Fossil_family_10_1536
	Florissant_Fossil_family_10_2048


	General_Fossil_family_20_512
	General_Fossil_family_20_1024
	General_Fossil_family_20_1536
	General_Fossil_family_20_2048
	Florissant_Fossil_family_20_512
	Florissant_Fossil_family_20_1024
	Florissant_Fossil_family_20_1536
	Florissant_Fossil_family_20_2048

	
	General_Fossil_family_50_512
	General_Fossil_family_50_1024
	General_Fossil_family_50_1536
	General_Fossil_family_50_2048
	Florissant_Fossil_family_50_512
	Florissant_Fossil_family_50_1024
	Florissant_Fossil_family_50_1536
	Florissant_Fossil_family_50_2048

--pnas-extant --extant-w-pnas --original





--------------

* [TODO] Add a way to switch between querying for image data dirs and the CSV dataset files (Added 2022-06-13).





python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/catalog_registry.py"

python "./lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/catalog_registry.py"



"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/processed_edit18Jan22/catalog_files_th/extant_family_10"



Examples:
	---------

	from imutils.catalog_registry import available_datasets

	available_datasets.get(tag='Fossil_2048', version='v1_0')
	>>['/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/2048/full/jpg',
	   '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/2048/full/jpg']

	available_datasets.get(tag='Extant_Leaves_family_20_1024', version='v1_0')
	>>'/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/20/jpg'


	print(available_datasets().tags)
	print(available_datasets())


	*shortcut*
	from lightning_hydra_classifiers import catalog_registry, available_datasets

"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import rich
from rich import print as pp
import yaml

from imutils.ml.utils.template_utils import get_logger
import logging
log = get_logger(name=__name__, level=logging.INFO)


# __all__ = [
# 	"leavesdb_catalogv0_3",
# 	"leavesdb_catalogv1_0",
# 	"leavesdb_catalogv1_1",
# 	"available_datasets",
# 	"AvailableDatasets"
# ]

##-----------------------

@dataclass
class BaseCatalog:

	#	 datasets = ["PNAS", "Extant", "Fossil"]

	def keys(self):
		return self.__dict__.keys()

	def __getitem__(self, index):
		return self.__dict__[index]

	@property
	def datasets(self):
		out = {"PNAS": self.PNAS, "Extant": self.Extant, "Fossil": self.Fossil}
		if len(self.Herbarium2022):
			out.update({"Herbarium2022": self.Herbarium2022})
		return out

	@property
	def tags(self):
		out = {}
		for k, v in self.datasets.items():
			out[k] = []
			for tag in v.keys():
				out[k].append(tag)
		return out

	def __repr__(self):
		out = []
		for k, v in self.datasets.items():
			if k != "all":
				out.append((k, len(v)))
		out = pd.DataFrame(out)
		out = out.rename(columns={0: "Base Dataset Name", 1: "# of Variations"}).set_index(
			"Base Dataset Name"
		)

		return out.__repr__()

	#	 def __repr__(self):
	#		 out = ".[italic red]".join(str(type(self)).replace("\'>","").split(".")[-2:]) + "[/]" + "\n"
	#		 # out = f"{out!r}"
	#		 out += rich.pretty.pretty_repr(self.datasets)
	#		 return out

	def __rich__(self):

		out = ".[italic red]".join(str(type(self)).replace("'>", "").split(".")[-2:]) + "[/]" + "\n"
		out += rich.pretty.pretty_repr(self.datasets)
		return out

	@property
	def PNAS(self):
		return {k: self[k] for k in self.keys() if k.startswith("PNAS")}

	@property
	def Extant(self):
		return {k: self[k] for k in self.keys() if k.startswith("Extant")}

	@property
	def Fossil(self):
		return {k: self[k] for k in self.keys() if k.startswith("Fossil")}
	
	@property
	def Herbarium2022(self):
		return {k: self[k] for k in self.keys() if k.startswith("Herbarium2022")}

##-----------------------
	
@dataclass
class LeavesdbCatalogv0_3(BaseCatalog):

	PNAS_family_100_original: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100"
	)
	PNAS_family_4_original: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_4"
	)

	PNAS_family_100_512: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_512"
	)
	PNAS_family_100_1024: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1024"
	)
	PNAS_family_100_1536: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1536"
	)
	PNAS_family_100_2048: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_2048"
	)
	################################S
	################################
	Extant_Leaves_family_10_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/512"
	Extant_Leaves_family_10_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/1024"
	Extant_Leaves_family_10_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/1536"
	Extant_Leaves_family_10_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_10/2048"
	Extant_Leaves_family_20_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/512"
	Extant_Leaves_family_20_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/1024"
	Extant_Leaves_family_20_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/1536"
	Extant_Leaves_family_20_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_20/2048"
	Extant_Leaves_family_50_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/512"
	Extant_Leaves_family_50_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/1024"
	Extant_Leaves_family_50_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/1536"
	Extant_Leaves_family_50_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_50/2048"
	Extant_Leaves_family_100_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/512"
	Extant_Leaves_family_100_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/1024"
	Extant_Leaves_family_100_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/1536"
	Extant_Leaves_family_100_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/extant_family_100/2048"
	################################
	################################
	Wilf_Fossil_512: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Wilf_Fossil",
	)
	Wilf_Fossil_1024: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Wilf_Fossil",
	)
	Wilf_Fossil_1536: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Wilf_Fossil",
	)
	Wilf_Fossil_2048: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Wilf_Fossil",
	)
	Florissant_Fossil_512: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Florissant_Fossil",
	)
	Florissant_Fossil_1024: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Florissant_Fossil",
	)
	Florissant_Fossil_1536: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Florissant_Fossil",
	)
	Florissant_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Florissant_Fossil"
	################################
	################################
	Fossil_512: List[str] = None
	Fossil_1024: List[str] = None
	Fossil_1536: List[str] = None
	Fossil_2048: List[str] = None

	def __post_init__(self):

		self.Fossil_512: List[str] = [
			"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Wilf_Fossil",
			"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_512/Florissant_Fossil",
		]
		self.Fossil_1024: List[str] = [
			"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Wilf_Fossil",
			"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1024/Florissant_Fossil",
		]
		self.Fossil_1536: List[str] = [
			"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Wilf_Fossil",
			"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_1536/Florissant_Fossil",
		]
		self.Fossil_2048: List[str] = [
			"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Wilf_Fossil",
			"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/Fossil/ccrop_2048/Florissant_Fossil",
		]

	@property
	def datasets(self):
		return {"PNAS": self.PNAS, "Extant": self.Extant, "Fossil": self.Fossil}

	def __repr__(self):
		return super().__repr__()

##-----------------------

@dataclass
class LeavesdbCatalogv1_0(BaseCatalog):

	Extant_Leaves_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/original/full/jpg"
	General_Fossil_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/original/full/jpg"
	Florissant_Fossil_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/original/full/jpg"
	PNAS_family_100_original: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100"
	)
	PNAS_original: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_1/PNAS"
	)

	PNAS_family_100_512: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_512"
	)
	PNAS_family_100_1024: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1024"
	)
	PNAS_family_100_1536: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1536"
	)
	PNAS_family_100_2048: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_2048"
	)
	# ################################S
	# ################################
	Extant_Leaves_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/full/jpg"
	Extant_Leaves_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/full/jpg"
	Extant_Leaves_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/full/jpg"
	Extant_Leaves_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/full/jpg"

	Extant_Leaves_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/3/jpg"
	Extant_Leaves_family_3_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/3/jpg"
	Extant_Leaves_family_3_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/3/jpg"
	Extant_Leaves_family_3_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/3/jpg"
	Extant_Leaves_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/3/jpg"

	Extant_Leaves_family_10_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/10/jpg"
	Extant_Leaves_family_10_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/10/jpg"
	Extant_Leaves_family_10_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/10/jpg"
	Extant_Leaves_family_20_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/20/jpg"
	Extant_Leaves_family_20_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/20/jpg"
	Extant_Leaves_family_20_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/20/jpg"
	Extant_Leaves_family_20_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/20/jpg"
	Extant_Leaves_family_10_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/50/jpg"
	Extant_Leaves_family_50_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/50/jpg"
	Extant_Leaves_family_50_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/50/jpg"
	Extant_Leaves_family_50_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/50/jpg"
	Extant_Leaves_family_100_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/512/100/jpg"
	Extant_Leaves_family_100_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/100/jpg"
	Extant_Leaves_family_100_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1536/100/jpg"
	Extant_Leaves_family_100_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/2048/100/jpg"

	# ################################
	# ################################

	General_Fossil_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/512/full/jpg"
	General_Fossil_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/1024/full/jpg"
	General_Fossil_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/1536/full/jpg"
	General_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/2048/full/jpg"
	Florissant_Fossil_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/512/full/jpg"
	Florissant_Fossil_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/1024/full/jpg"
	Florissant_Fossil_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/1536/full/jpg"
	Florissant_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/2048/full/jpg"

	General_Fossil_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/512/3/jpg"
	General_Fossil_family_3_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/1024/3/jpg"
	General_Fossil_family_3_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/1536/3/jpg"
	General_Fossil_family_3_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/2048/3/jpg"
	Florissant_Fossil_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/512/3/jpg"
	Florissant_Fossil_family_3_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/1024/3/jpg"
	Florissant_Fossil_family_3_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/1536/3/jpg"
	Florissant_Fossil_family_3_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/2048/3/jpg"

	# ################################
	# ################################
	original: List[str] = None
	Fossil_512: List[str] = None
	Fossil_1024: List[str] = None
	Fossil_1536: List[str] = None
	Fossil_2048: List[str] = None
	Fossil_family_3_512: List[str] = None
	Fossil_family_3_1024: List[str] = None
	Fossil_family_3_1536: List[str] = None
	Fossil_family_3_2048: List[str] = None

	def __post_init__(self):

		self.original: List[str] = [
			self.Extant_Leaves_original,
			self.General_Fossil_original,
			self.Florissant_Fossil_original,
		]
		# Exclude PNAS_family_100_original b/c it's not really useful for concatenation.

		self.Fossil_original: List[str] = [
			self.General_Fossil_original,
			self.Florissant_Fossil_original,
		]

		self.Fossil_512: List[str] = [self.General_Fossil_512, self.Florissant_Fossil_512]
		self.Fossil_1024: List[str] = [self.General_Fossil_1024, self.Florissant_Fossil_1024]
		self.Fossil_1536: List[str] = [self.General_Fossil_1536, self.Florissant_Fossil_1536]
		self.Fossil_2048: List[str] = [self.General_Fossil_2048, self.Florissant_Fossil_2048]
		self.Fossil_family_3_512: List[str] = [
			self.General_Fossil_family_3_512,
			self.Florissant_Fossil_family_3_512,
		]
		self.Fossil_family_3_1024: List[str] = [
			self.General_Fossil_family_3_1024,
			self.Florissant_Fossil_family_3_1024,
		]
		self.Fossil_family_3_1536: List[str] = [
			self.General_Fossil_family_3_1536,
			self.Florissant_Fossil_family_3_1536,
		]
		self.Fossil_family_3_2048: List[str] = [
			self.General_Fossil_family_3_2048,
			self.Florissant_Fossil_family_3_2048,
		]

	@property
	def datasets(self):
		return {
			"PNAS": self.PNAS,
			"Extant": self.Extant,
			"Fossil": self.Fossil,
			"all": self.original,
		}

	@property
	def tags(self):
		out = {}
		for k, v in self.datasets.items():
			if isinstance(v, list):
				out[k] = v
			else:
				out[k] = []
				for tag in v.keys():
					out[k].append(tag)
		return out

	def __repr__(self):
		return super().__repr__()

##-----------------------

@dataclass
class LeavesdbCatalogv1_1(BaseCatalog):
	"""
	Leavesdb-v1.1

	Released Dec 2021/Jan 2022 as a direct descendent of leavesdb-v1.0-patch.2
	"""

	Extant_Leaves_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/original/full/jpg"
	General_Fossil_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/original/full/jpg"
	Florissant_Fossil_original: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/original/full/jpg"
	PNAS_family_100_original: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100"
	)
	PNAS_original: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_1/PNAS"
	)

	PNAS_family_100_512: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_512"
	)
	PNAS_family_100_1024: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1024"
	)
	PNAS_family_100_1536: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1536"
	)
	PNAS_family_100_2048: str = (
		"/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_2048"
	)
	# ################################S
	# ################################
	Extant_Leaves_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/512/full/jpg"
	Extant_Leaves_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1024/full/jpg"
	Extant_Leaves_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1536/full/jpg"
	Extant_Leaves_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/2048/full/jpg"

	Extant_Leaves_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/512/3/jpg"
	Extant_Leaves_family_3_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1024/3/jpg"
	Extant_Leaves_family_3_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1536/3/jpg"
	Extant_Leaves_family_3_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/2048/3/jpg"
	Extant_Leaves_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/512/3/jpg"

	Extant_Leaves_family_10_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1024/10/jpg"
	Extant_Leaves_family_10_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1536/10/jpg"
	Extant_Leaves_family_10_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/2048/10/jpg"
	Extant_Leaves_family_20_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/512/20/jpg"
	Extant_Leaves_family_20_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1024/20/jpg"
	Extant_Leaves_family_20_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1536/20/jpg"
	Extant_Leaves_family_20_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/2048/20/jpg"
	Extant_Leaves_family_10_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/512/50/jpg"
	Extant_Leaves_family_50_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1024/50/jpg"
	Extant_Leaves_family_50_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1536/50/jpg"
	Extant_Leaves_family_50_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/2048/50/jpg"
	Extant_Leaves_family_100_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/512/100/jpg"
	Extant_Leaves_family_100_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1024/100/jpg"
	Extant_Leaves_family_100_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/1536/100/jpg"
	Extant_Leaves_family_100_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/2048/100/jpg"

	# ################################
	# ################################

	General_Fossil_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/512/full/jpg"
	General_Fossil_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1024/full/jpg"
	General_Fossil_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1536/full/jpg"
	General_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/2048/full/jpg"
	Florissant_Fossil_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/512/full/jpg"
	Florissant_Fossil_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1024/full/jpg"
	Florissant_Fossil_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1536/full/jpg"
	Florissant_Fossil_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/2048/full/jpg"

	General_Fossil_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/512/3/jpg"
	General_Fossil_family_3_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1024/3/jpg"
	General_Fossil_family_3_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1536/3/jpg"
	General_Fossil_family_3_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/2048/3/jpg"
	Florissant_Fossil_family_3_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/512/3/jpg"
	Florissant_Fossil_family_3_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1024/3/jpg"
	Florissant_Fossil_family_3_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1536/3/jpg"
	Florissant_Fossil_family_3_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/2048/3/jpg"

	
	General_Fossil_family_10_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/512/10/jpg"
	General_Fossil_family_10_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1024/10/jpg"
	General_Fossil_family_10_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1536/10/jpg"
	General_Fossil_family_10_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/2048/10/jpg"
	Florissant_Fossil_family_10_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/512/10/jpg"
	Florissant_Fossil_family_10_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1024/10/jpg"
	Florissant_Fossil_family_10_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1536/10/jpg"
	Florissant_Fossil_family_10_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/2048/10/jpg"


	General_Fossil_family_20_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/512/20/jpg"
	General_Fossil_family_20_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1024/20/jpg"
	General_Fossil_family_20_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1536/20/jpg"
	General_Fossil_family_20_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/2048/20/jpg"
	Florissant_Fossil_family_20_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/512/20/jpg"
	Florissant_Fossil_family_20_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1024/20/jpg"
	Florissant_Fossil_family_20_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1536/20/jpg"
	Florissant_Fossil_family_20_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/2048/20/jpg"

	
	General_Fossil_family_50_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/512/50/jpg"
	General_Fossil_family_50_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1024/50/jpg"
	General_Fossil_family_50_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/1536/50/jpg"
	General_Fossil_family_50_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/2048/50/jpg"
	Florissant_Fossil_family_50_512: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/512/50/jpg"
	Florissant_Fossil_family_50_1024: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1024/50/jpg"
	Florissant_Fossil_family_50_1536: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/1536/50/jpg"
	Florissant_Fossil_family_50_2048: str = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/2048/50/jpg"
	
	
	
	
	# ################################
	# ################################
	original: List[str] = None
	Fossil_512: List[str] = None
	Fossil_1024: List[str] = None
	Fossil_15106: List[str] = None
	Fossil_2048: List[str] = None

	Fossil_family_3_512: List[str] = None
	Fossil_family_3_1024: List[str] = None
	Fossil_family_3_1536: List[str] = None
	Fossil_family_3_2048: List[str] = None
	
	Fossil_family_10_512: List[str] = None
	Fossil_family_10_1024: List[str] = None
	Fossil_family_10_1536: List[str] = None
	Fossil_family_10_2048: List[str] = None

	Fossil_family_20_512: List[str] = None
	Fossil_family_20_1024: List[str] = None
	Fossil_family_20_1536: List[str] = None
	Fossil_family_20_2048: List[str] = None

	Fossil_family_50_512: List[str] = None
	Fossil_family_50_1024: List[str] = None
	Fossil_family_50_1536: List[str] = None
	Fossil_family_50_2048: List[str] = None


	def __post_init__(self):

		self.original: List[str] = [
			self.Extant_Leaves_original,
			self.General_Fossil_original,
			self.Florissant_Fossil_original,
		]
		# Exclude PNAS_family_100_original b/c it's not really useful for concatenation.

		self.Fossil_original: List[str] = [
			self.General_Fossil_original,
			self.Florissant_Fossil_original,
		]

		self.Fossil_512: List[str] = [self.General_Fossil_512, self.Florissant_Fossil_512]
		self.Fossil_1024: List[str] = [self.General_Fossil_1024, self.Florissant_Fossil_1024]
		self.Fossil_1536: List[str] = [self.General_Fossil_1536, self.Florissant_Fossil_1536]
		self.Fossil_2048: List[str] = [self.General_Fossil_2048, self.Florissant_Fossil_2048]
		self.Fossil_family_3_512: List[str] = [
			self.General_Fossil_family_3_512,
			self.Florissant_Fossil_family_3_512,
		]
		self.Fossil_family_3_1024: List[str] = [
			self.General_Fossil_family_3_1024,
			self.Florissant_Fossil_family_3_1024,
		]
		self.Fossil_family_3_1536: List[str] = [
			self.General_Fossil_family_3_1536,
			self.Florissant_Fossil_family_3_1536,
		]
		self.Fossil_family_3_2048: List[str] = [
			self.General_Fossil_family_3_2048,
			self.Florissant_Fossil_family_3_2048,
		]
###################
		
		self.Fossil_family_10_512: List[str] = [
			self.General_Fossil_family_10_512,
			self.Florissant_Fossil_family_10_512,
		]
		self.Fossil_family_10_1024: List[str] = [
			self.General_Fossil_family_10_1024,
			self.Florissant_Fossil_family_10_1024,
		]
		self.Fossil_family_10_1536: List[str] = [
			self.General_Fossil_family_10_1536,
			self.Florissant_Fossil_family_10_1536,
		]
		self.Fossil_family_10_2048: List[str] = [
			self.General_Fossil_family_10_2048,
			self.Florissant_Fossil_family_10_2048,
		]
		
		######################
		
		self.Fossil_family_20_512: List[str] = [
			self.General_Fossil_family_20_512,
			self.Florissant_Fossil_family_20_512,
		]
		self.Fossil_family_20_1024: List[str] = [
			self.General_Fossil_family_20_1024,
			self.Florissant_Fossil_family_20_1024,
		]
		self.Fossil_family_20_1536: List[str] = [
			self.General_Fossil_family_20_1536,
			self.Florissant_Fossil_family_20_1536,
		]
		self.Fossil_family_20_2048: List[str] = [
			self.General_Fossil_family_20_2048,
			self.Florissant_Fossil_family_20_2048,
		]
		
		######################
		
		self.Fossil_family_50_512: List[str] = [
			self.General_Fossil_family_50_512,
			self.Florissant_Fossil_family_50_512,
		]
		self.Fossil_family_50_1024: List[str] = [
			self.General_Fossil_family_50_1024,
			self.Florissant_Fossil_family_50_1024,
		]
		self.Fossil_family_50_1536: List[str] = [
			self.General_Fossil_family_50_1536,
			self.Florissant_Fossil_family_50_1536,
		]
		self.Fossil_family_50_2048: List[str] = [
			self.General_Fossil_family_50_2048,
			self.Florissant_Fossil_family_50_2048,
		]
		

	@property
	def datasets(self):
		return {
			"PNAS": self.PNAS,
			"Extant": self.Extant,
			"Fossil": self.Fossil,
			"all": self.original,
		}


	@property
	def tags(self):
		out = {}
		for k, v in self.datasets.items():
			if isinstance(v, list):
				out[k] = v
			else:
				out[k] = []
				for tag in v.keys():
					out[k].append(tag)
		return out
	
	
	def __repr__(self):
		return super().__repr__()


@dataclass
class ThirdPartyDatasets(BaseCatalog):
	"""
	ThirdPartyDatasets
	
	Released April 2022
	
	Extension of catalog_registry-style LeavesDb class-based registry of on-disk image datasets from our 3 main leaf datasets.
	
	Purpose is to integrate Herbarium 2022 data into in-house workflow.

	Released Dec 2021/Jan 2022 as a direct descendent of leavesdb-v1.0-patch.2
	"""
	
	Herbarium2022_960: str = "/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize"
	Herbarium2022_512: str = "/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize-512"


	@property
	def datasets(self):
		return {
			"Herbarium2022": self.Herbarium2022
		}


	def __repr__(self):
		return super().__repr__()












##############################


def query_dict(target: Dict[str, str], query: str) -> Dict[str, str]:
	"""
	Helper function
	Searches a target dictionary for k,v pairs for which the key contains a substring equal to {query}.
	Returned dictionary contains a minimum of 0 and maximum of len(target) items.
	"""
	return {k: v for k, v in vars(target).items() if query in k}


#############################

leavesdb_catalog_v0_3 = LeavesdbCatalogv0_3()
leavesdb_catalog_v1_0 = LeavesdbCatalogv1_0()
leavesdb_catalog_v1_1 = LeavesdbCatalogv1_1()


third_party = ThirdPartyDatasets()


# import rich.repr
# @rich.repr.auto
class AvailableDatasets:
	# class available_datasets:
	# class AvailableDatasets:
	"""
	Central location for querying all individual *datasets* distributed among multiple *versioned catalogs*.
	
	
	To do: Consider whether to move "third_party" versions to "extras" instead.
	
	
	"""

	versions = {"v0_3": leavesdb_catalog_v0_3, "v1_0": leavesdb_catalog_v1_0, "v1_1": leavesdb_catalog_v1_1, "third_party": third_party}
	extras = {}
	
	def display_all(self):
		# return yaml.dump({**self.versions, "extras":self.extras})
		# return {**self.versions, "extras":yaml.dump(self.extras)}
		return {**self.versions, "extras":self.extras}
	
	@property
	def db_versions(self):
		return self.versions
	
	
	@classmethod
	def add_dataset(cls, tag: str, root_dir: str) -> None:
		try:
			if tag in cls.extras:
				root_dir = cls.extras[tag]
			else:
				root_dir = cls.get_latest(tag)
			print(f"Attempted to add existing dataset with tag: {tag} at root_dir: {root_dir}. Continuing with existing entry.")
		except KeyError:
			cls.extras[tag] = root_dir
			print(f"Added new dataset with tag: {tag} to root_dir: {root_dir}")
	

	def __repr__(self):
		buffer = r"<" * 3 + r"-" * 10 + r">" * 3 + "\n"
		return buffer.join(
			[f"{k}:" + "\n" + v.__repr__() + "\n" for k, v in self.db_versions.items()]
		)

	def __rich__(self):
		buffer = r"<" * 3 + r"-" * 10 + r">" * 3 + "\n"
		return buffer.join(
			[f"{k}:" + "\n" + v.__rich__() + "\n" for k, v in self.db_versions.items()]
		)

	@classmethod
	def search(
		cls,
		query: str,
		version: Optional[str] = "v1_1"
	) -> Dict[str, Union[str, List[str]]]:
		"""
		Helper function
		Searches a target dictionary for k,v pairs for which the key contains a substring equal to {query}.
		Returned dictionary contains a minimum of 0 and maximum of len(target) items.

		Example:
		--------
		> from lightning_hydra_classifiers.data.utils import catalog_registry
		> available_datasets = catalog_registry.AvailableDatasets()
		> results = available_datasets.search("Exta")
		Query can be any partial string:
		> query = "Ex" or "Extant" or "512" or "original"

		"""
		target = cls.versions[version]
		return query_dict(target=target, query=query)

	@classmethod
	def query_tags(
		cls,
		dataset_name: str,
		threshold: Optional[int] = 0,
		y_col: str = "family",
		resolution: Optional[Tuple[str, int]] = "original",
	) -> str:
		"""
		Helper function
		Converts multiple kwargs into a single formatted str tag. Use tags to query specific datasets and their data/metadata.

		"""
		tag = dataset_name
		if int(threshold) > 0:
			tag += f"_{y_col}_{threshold}"
		tag += f"_{resolution}"

		try:
			cls.get_latest(tag)
		except KeyError as e:
			print(f"KeyError: Invalid dataset query. {tag} doesn't exist")
			print("tag: ", tag)
		return tag

	#		 if isinstance(resolution, int):

	@classmethod
	def get_latest(cls, tag: str) -> Union[str, List[str]]:
		"""
		Wrapper around available_datasets.get() that defaults to the latest dataset version containing the requested tag.

		Useful for datasets like PNAS, which havent changed since version 0_3.

		"""
		if "PNAS" in tag:
			try:
				out = cls.get(tag, version="v1_1")
				version = "v1_1"
			except KeyError:
				out = cls.get(tag, version="v0_3")
				version="v0_3"
		elif "Herbarium" in tag:
			out = cls.get(tag, version="third_party")
			version = "third_party"
		else:
			out = cls.get(tag, version="v1_1")
			version = "v1_1"
		
		log.info(f"Found latest version of {tag=} in {version=}")
		return out

	@classmethod
	def get(cls, tag: str, version: Optional[str] = "v1_1") -> Union[str, List[str]]:
		"""
		Get a dataset root path by providing the dataset tag/name + an optional version tag.

		Examples:
		---------
		available_datasets.get(tag='Fossil_2048', version='v1_0')
		>>['/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/General_Fossil/2048/full/jpg',
		   '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Fossil/Florissant_Fossil/2048/full/jpg']

		available_datasets.get(tag='Extant_Leaves_family_20_1024', version='v1_0')
		>>'/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images/Extant_Leaves/1024/20/jpg'

		"""
		return cls.versions[version][tag]
	
	
	@classmethod
	def get_image_label_list(cls, tag: str, version: Optional[str] = "v1_1") -> Union[str, List[str]]:
		pass


	@property
	def tags(self):
		out = {}
		for version, dataset in self.versions.items():
			out[version] = dataset.tags
		return out


def cmdline_args():
	p = argparse.ArgumentParser(
		description=(
			"catalog_registry.py -- Module containing key: value mappings between accepted dataset"
			" names with versions, and their corresponding locations on data_cifs."
		)
	)
	p.add_argument(
		"-t",
		"--tags",
		action="store_true",
		help=(
			"User provides this flag to display a concise summary of available data, containing all"
			" tag info while omitting paths."
		),
	)
	p.add_argument(
		"-d",
		"--display",
		action="store_true",
		help=(
			"User provides this flag to display a full listing of all datasets with versions,"
			" mapped to their expected data_cifs locations."
		),
	)
	return p.parse_args()


available_datasets = AvailableDatasets()

if __name__ == "__main__":

	args = cmdline_args()

	if args.tags:
		pp(available_datasets.tags)
	elif args.display:
		pp(available_datasets)
	else:
		print(
			f"Provide either --tags or --display if running catalog_registry.py from the command"
			f" line."
		)
