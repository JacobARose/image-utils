#!/usr/bin/env python
# coding: utf-8

"""
make_herbarium_2022_catalog_df.py

"""
# 
# Description: 
# 
# Created On: Sunday Feb 27th, 2022  
# Created By: Jacob A Rose

# ### Key constants

# DATASETS_ROOT = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images"
# EXTANT_ROOT = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Extant_Leaves/original/full/jpg"
# GENERAL_ROOT = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/General_Fossil/original/full/jpg"
# FLORISSANT_ROOT = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images/Fossil/Florissant_Fossil/original/full/jpg"

# with open(os.path.join(HERBARIUM_ROOT, "train_metadata.json")) as fp:
#	 train_data = json.load(fp)

# with open(os.path.join(HERBARIUM_ROOT, "test_metadata.json")) as fp:
#	 test_data = json.load(fp)

# for k,v in train_data.items():
#	 print(k, f"| Total:{len(v)}")
#	 print("First:", v[0])
#	 print("Last:", v[-1])
#	 print("="*15+"\n")

# assert len(train_data["annotations"]) == len(train_data["images"])

import argparse
import os
import sys
from typing import *
import json
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print as pp


# HERBARIUM_ROOT_DEFAULT = "/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize"

# from dotenv import load_dotenv
# load_dotenv()
import imutils
from imutils.big.split_catalog_utils import TRAIN_KEY, VAL_KEY, TEST_KEY

HERBARIUM_ROOT_DEFAULT = os.environ["HERBARIUM_ROOT_DEFAULT"]
CATALOG_DIR = os.environ["CATALOG_DIR"]
SPLITS_DIR = os.environ["SPLITS_DIR"]


def optimize_dtypes_train(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Convert column dtypes to optimal type for herbarium train metadata df.
	"""

	# Reduce total df size by optimizing dtypes per column
	cat_cols = ['genus_id', 'institution_id', 'category_id',
				'scientificName', 'family', 'genus', 'species','Species',
				'collectionCode', 'license', 'authors']
	if "y" in df.columns:
		cat_cols.append("y")
	
	str_cols = ['image_id', 'file_name', 'path']
	col_dtypes = {c:"category" for c in cat_cols if c in df.columns}
	col_dtypes.update({c:"string" for c in str_cols})

	# df = df.convert_dtypes()
	df = df.astype(col_dtypes)
	return df


def optimize_dtypes_test(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Convert column dtypes to optimal type for herbarium test metadata df.
	"""
	dtypes_test = {'image_id':"string",
					'file_name':"string",
					'license':"category",
					'path':"string"}
	dtypes_test= {col:dtype for col, dtype in dtypes_test.items() if col in df.columns}
	# Reduce total df size by optimizing dtypes per column
	df = df.astype(dtypes_test)
	return df


def read_train_df_from_csv(train_path,
						   nrows: Optional[int]=None,
						   index_col: int=0
						   ) -> pd.DataFrame:
	
	df = pd.read_csv(train_path, index_col=index_col, nrows=nrows)
	df = optimize_dtypes_train(df)
	return df


def read_test_df_from_csv(test_path,
						  nrows: Optional[int]=None,
						  index_col: int=0
						  ) -> pd.DataFrame:
	
	df = pd.read_csv(test_path, index_col=index_col, nrows=nrows)
	df = optimize_dtypes_test(df)
	return df


def read_all_from_csv(root_dir: str=None,
					  source_csv_paths: Optional[List[str]]=None,
					  subset_read_funcs: Union[Callable, Dict[str, Callable]]={
						  TRAIN_KEY: read_train_df_from_csv,
						  TEST_KEY: read_test_df_from_csv
					  },
					  return_dict: bool=False,
					 **kwargs) -> Tuple[pd.DataFrame]:
	"""
	Read the train_metadata.csv and test_metadata.csv files from `root_dir`
	
	Note: This is prior to any train-val splits.
	"""
	if source_csv_paths is not None:
		train_path, test_path = sorted(source_csv_paths)[::-1]
	else:
		train_path = Path(root_dir, "train_metadata.csv")
		test_path = Path(root_dir, "test_metadata.csv")
	
	if isinstance(subset_read_funcs, Callable):
		train_df = subset_read_funcs(train_path)
		test_df = subset_read_funcs(test_path)
	else:
		train_df = subset_read_funcs[TRAIN_KEY](train_path)
		test_df = subset_read_funcs[TEST_KEY](test_path)
		
	# train_df = read_train_df_from_csv(train_path)
	# test_df = read_test_df_from_csv(test_path)
	
	if return_dict:
		return {
			TRAIN_KEY: train_df,
			TEST_KEY: test_df
		}

	return train_df, test_df
	

	# read_train_df_from_csv,
	# read_test_df_from_csv
	
	
###################################
###################################

class HerbariumMetadata:
	
	TRAIN_KEYS = ['annotations', 'images', 'categories', 'genera', 'institutions', 'distances', 'license']
	TEST_KEYS = ['image_id', 'file_name', 'license']
	
	def __init__(self,
				 herbarium_root: str=HERBARIUM_ROOT_DEFAULT):
		self.herbarium_root = herbarium_root
		
	

	def get_train_df(self) -> pd.DataFrame:

		metadata_path = Path(self.herbarium_root, "train_metadata.json")

		with open(os.path.join(metadata_path)) as fp:
			train_data = json.load(fp)

		assert all([k in train_data.keys() for k in self.TRAIN_KEYS])

		train_annotations = pd.DataFrame(train_data['annotations'])

		train_categories = pd.DataFrame(train_data['categories']).set_index("category_id")
		train_genera = pd.DataFrame(train_data['genera']).set_index("genus_id")
		train_institutions = pd.DataFrame(train_data['institutions']).set_index("institution_id")
		train_images = pd.DataFrame(train_data['images']).set_index("image_id")

		df_train = pd.merge(train_annotations, train_images, how="left", right_index=True, left_on="image_id")
		df_train = pd.merge(df_train, train_categories, how="left", right_index=True, left_on="category_id")
		df_train = pd.merge(df_train, train_institutions, how="left", right_index=True, left_on="institution_id")

		df_train = df_train.assign(
			Species = df_train.apply(lambda x: " ".join([x.genus, x.species]), axis=1),
			path=df_train.file_name.apply(lambda x: str(Path(self.herbarium_root, "train_images", x)))
		)

		df_train = optimize_dtypes_train(df_train)

		print(f"training images: {len(df_train)}")

		return df_train

	
	def get_test_df(self) -> pd.DataFrame:

		metadata_path = Path(self.herbarium_root, "test_metadata.json")

		with open(os.path.join(metadata_path)) as fp:
			test_data = json.load(fp)

		assert all([k in test_data[0].keys() for k in self.TEST_KEYS])

		df_test = pd.DataFrame(test_data)
		df_test = df_test.assign(path=df_test.file_name.apply(lambda x: str(Path(self.herbarium_root, "test_images", x))))

		df_test = optimize_dtypes_test(df_test)
		print(f"test images: {len(df_test)}")

		return df_test


	def extract_metadata(self) -> Tuple[pd.DataFrame]:

		df_train = self.get_train_df()
		df_test = self.get_test_df()

		return df_train, df_test
	
	
	def write_herbarium_metadata2disk(
		self,
		output_dir: str=None,
		force_overwrite: bool=False
	) -> Tuple[Path]:
		"""
		Reads json metadata files from `root_dir`, parses into train & test dataframes, then writes to disk as csv files.
		"""
		assert os.path.isdir(output_dir)

		# df_train, df_test = extract_metadata(root_dir = root_dir)
		train_path = Path(output_dir, "train_metadata.csv")
		test_path = Path(output_dir, "test_metadata.csv")

		if os.path.exists(train_path) and not force_overwrite:
			print(train_path, "already exists, and force_overwrite==False, skipping write process.",
				  "Delete the existing file if intention is to refresh dataset.")
		elif os.path.exists(train_path):
			pp(f"force_overwrite is set to true, removing previously existing file: {os.path.basename(train_path)}")
			os.remove(train_path)

		if os.path.exists(test_path) and not force_overwrite:
			print(test_path, "already exists, and force_overwrite==False, skipping write process.",
				  "Delete the existing file or pass --force_overwrite if intention is to refresh dataset.")
		elif os.path.exists(test_path):
			pp(f"force_overwrite is set to true, removing previously existing file: {os.path.basename(test_path)}")
			os.remove(test_path)


			
		if not os.path.exists(train_path):
			print(f"Writing train data to: {train_path}")
			df_train = self.get_train_df()
			df_train.to_csv(train_path)

		if not os.path.exists(test_path):
			print(f"Writing test data to: {test_path}")
			df_test = self.get_test_df()
			df_test.to_csv(test_path)

		return train_path, test_path




def parse_args() -> argparse.Namespace:
	
	parser = argparse.ArgumentParser(
		"""Generate train_metadata.csv and test_metadata.csv from Herbarium 2022 original json metadata splits in train_metadata.json and test_metadata.json. Defaults to generating outputs within a `catalogs/` subdir of the location of the input json files.
		
		Inputs:
		* train_metadata.json
		* test_metadata.json		
		
		Outputs:
		* train_metadata.csv 
		* test_metadata.csv
		"""
	)
	parser.add_argument(
		"--target_dir", default=CATALOG_DIR, help="directory where catalog csv files are written"
	)
	parser.add_argument(
		"--herbarium_source_dir",
		default=HERBARIUM_ROOT_DEFAULT,
		help="Source directory containing original herbarium 2022 dataset, as accessed by kaggle.",
	)
	parser.add_argument(
		"--info", action="store_true", help="Flag to print execution variables then quit without execution."
	)
	parser.add_argument(
		"--force_overwrite", action="store_true", help="Flag to allow removal of pre-existing output files if they already exist, instead of skipping creation during execution."
	)
	
	args = parser.parse_args()
	# WORKING_DIR = "/media/data/jacob/GitHub/image-utils/notebooks/herbarium_2022/"
	# OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
	# DATA_DIR = os.path.join(WORKING_DIR, "data")
	
	if args.info:
		print("User passed --info, displaying execution args then exiting")
		pp(args)
		sys.exit(0)
		
	
	os.makedirs(args.target_dir, exist_ok=True)	
	
	return args








if __name__=="__main__":
	
	args = parse_args()
	
	metadata = HerbariumMetadata(herbarium_root=args.herbarium_source_dir)
	
	train_path, test_path = metadata.write_herbarium_metadata2disk(output_dir=args.target_dir,
																   force_overwrite=args.force_overwrite)

	# train_df, test_df = read_all_from_csv(root_dir=HERBARIUM_ROOT)

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
################################################################

################

import torch
import torchvision
from PIL import Image


class SupervisedImageDataset(torch.utils.data.Dataset):
	def __init__(self, 
				 df: pd.DataFrame,
				 path_col: str="path",
				 label_col: str="Family"):
		self.df = df
		self.path_col = path_col
		self.label_col = label_col
		self._create_labels()

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		row = self.df.iloc[index]
		return (
			# torchvision.transforms.functional.to_tensor(Image.open(row[self.path_col])),
			Image.open(row.loc[self.path_col]).convert('RGB'),
			row["int_labels"],
		)
	
	def _create_labels(self):
		
		self.classnames = list(set(self.df[self.label_col]))
		print(f"Found {len(self.classnames)} unique labels")

		self.label2int = {l:i for i,l in enumerate(self.classnames)}
		self.int2label = {i:l for l,i in self.label2int.items()}
		
		self.df.loc["int_labels"] = self.df[self.label_col].apply(lambda x: self.label2int[x])


# ## Can we stratify by genus or Family while classifying species?

# from sklearn.model_selection import train_test_split
# num_samples = df_train.shape[0]
# label_col = "category_id"
# train_size=0.7
# seed = 14

# x = np.arange(num_samples)
# y = df_train[label_col].values

# x_train, x_val, y_train, y_val = train_test_split(x, y,
#												   stratify=y,
#												   train_size=train_size,
#												   random_state=seed)
# train_data = df_train.iloc[x_train,:]
# val_data = df_train.iloc[x_val,:]

# train_data.shape[0] + val_data.shape[0]

# # plt.figure(figsize=(20,10))
# train_data.hist("category_id", bins=100, figsize=(20,10), alpha=0.5)
# plt.suptitle(f"Train Species counts ({train_data.shape[0]:,} imgs)")
# val_data.hist("category_id", bins=100, figsize=(20,10), alpha=0.5)
# plt.suptitle(f"Val Species counts ({val_data.shape[0]:,} imgs)")




def categorical_order(values, order=None):
	"""Return a list of unique data values.

	Determine an ordered list of levels in ``values``.

	Parameters
	----------
	values : list, array, Categorical, or Series
		Vector of "categorical" values
	order : list-like, optional
		Desired order of category levels to override the order determined
		from the ``values`` object.

	Returns
	-------
	order : list
		Ordered list of category levels

	"""
	if order is None:
		if hasattr(values, "categories"):
			order = values.categories
		else:
			try:
				order = values.cat.categories
			except (TypeError, AttributeError):
				try:
					order = values.unique()
				except AttributeError:
					order = pd.unique(values)

	return list(order)


