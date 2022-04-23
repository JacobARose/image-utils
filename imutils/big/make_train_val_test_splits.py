"""
make_train_val_test_splits.py

Load a csv file into a dataframe containing Leavesdb metadata, split it into train, val, and test subsets, then write to new location as csv files.


Created on: Wednesday April 6th, 2022
Created by: Jacob A Rose



python "/media/data_cifs/projects/prj_fossils/users/jacob/github/image-utils/imutils/big/make_train_val_test_splits.py" \
--root_dir "/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1" \
--label_col "family" \
--splits "0.5,0.2,0.3" \
--seed 14 \
--run_all


"""

import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Can we stratify by genus or Family while classifying species?
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
from typing import *
import json
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint as pp

# from imutils.big.make_herbarium_2022_catalog_df import (read_all_from_csv,
# 														read_train_df_from_csv,
# 														read_test_df_from_csv)
from imutils import catalog_registry
from imutils.ml.utils import template_utils

log = template_utils.get_logger(__file__)


# def train_val_split(df: pd.DataFrame,
# 					label_col = "scientificName",
# 					train_size=0.7,
# 					seed = 14
# 					) -> Tuple[pd.DataFrame]:

# 	num_samples = df.shape[0]
# 	x = np.arange(num_samples)
# 	y = df[label_col].values

# 	x_train, x_val, _, _ = train_test_split(x, y,
# 											stratify=y,
# 											train_size=train_size,
# 											random_state=seed)
# 	train_data = df.iloc[x_train,:]
# 	val_data = df.iloc[x_val,:]

# 	return train_data, val_data

####################################################

def trainvaltest_split(df: pd.DataFrame,
					   label_col = "family",
					   splits: List[float]=(0.5, 0.2, 0.3),
					   seed = 14,
					   stratify: bool=True
					   ) -> Dict[str,pd.DataFrame]:
	"""
	Wrapper function to split data into 3 stratified subsets specified by `splits`.
	
	User specifies absolute fraction of total requested for each subset (e.g. splits=[0.5, 0.2, 0.3])
	
	Function calculates adjusted fractions necessary in order to use sklearn's builtin train_test_split function over a sequence of 2 steps.
	
	Step 1: Separate test set from the rest of the data (constituting the union of train + val)
	
	Step 2: Separate the train and val sets from the remainder produced by step 1.

	Output:
		Dict: {'train':(x_train, y_train),
				'val':(x_val_y_val),
				'test':(x_test, y_test)}
				
	Example:
		>> data = torch.data.Dataset(...)
		>> y = data.targets
		>> data_splits = trainvaltest_split(x=None,
											y=y,
											splits=(0.5, 0.2, 0.3),
											random_state=0,
											stratify=True)
	
	"""
	
	
	assert len(splits) == 3, "Must provide eactly 3 float values for `splits`"
	assert np.isclose(np.sum(splits), 1.0), f"Sum of all splits values {splits} = {np.sum(splits)} must be 1.0"
	
	train_split, val_split, test_split = splits
	val_relative_split = val_split/(train_split + val_split)
	train_relative_split = train_split/(train_split + val_split)
	
	if stratify and (label_col is None):
		raise ValueError("If label_col is not provided, stratify must be set to False.")

	num_samples = df.shape[0]
	x = np.arange(num_samples)
	y = df[label_col].values
	
	# y = np.array(y)
	# if x is None:
	#	 x = np.arange(len(y))
	# else:
	#	 x = np.array(x)

	stratify_y = y if stratify else None	
	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y,
														test_size=test_split, 
														random_state=seed,
														stratify=y)
	
	stratify_y_train = y_train_val if stratify else None
	x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
													  test_size=val_relative_split,
													  random_state=seed, 
													  stratify=y_train_val)
	
	x = np.concatenate((x_train, x_val, x_test)).tolist()
	assert len(set(x)) == len(x), f"[Warning] Check for possible data leakage. len(set(x))={len(set(x))} != len(x)={len(x)}"

	train_data = df.iloc[x_train,:]
	val_data = df.iloc[x_val,:]
	test_data = df.iloc[x_test,:]


	log.debug(f"x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
	log.debug(f"x_val.shape={x_val.shape}, y_val.shape={y_val.shape}")
	log.debug(f"x_test.shape={x_test.shape}, y_test.shape={y_test.shape}")
	log.debug(f'Absolute splits: {[train_split, val_split, test_split]}')
	log.debug(f'Relative splits: [{train_relative_split:.2f}, {val_relative_split:.2f}, {test_split}]')
	

	return {"train":train_data,
			"val":val_data,
			"test":test_data}








def fit_and_encode_labels(train_data,
						  val_data,
						  test_data=None,
						  label_col: str="family"
						 ) -> Tuple[LabelEncoder, pd.DataFrame]:

	encoder = LabelEncoder()
	encoder.fit(train_data[label_col])
	
	split_data = {}
	
	train_data = train_data.assign(
		y = encoder.transform(train_data[label_col])
			).astype({"y":"category"})
	split_data["train"] = train_data
	
	val_data = val_data.assign(
		y = encoder.transform(val_data[label_col])
			).astype({"y":"category"})
	split_data["val"] = val_data

	if test_data is not None:
		if label_col in test_data.columns:
			test_data = test_data.assign(
				y = encoder.transform(test_data[label_col])
					).astype({"y":"category"})
			split_data["test"] = test_data

	return encoder, split_data




def save_label_encoder(encoder,
					   root_dir: str,
					   label_col: str) -> str:
	label_name = f"{label_col}-encoder.pkl"
	label_encoder_path = Path(root_dir, label_name)

	with open(label_encoder_path, mode="wb") as fp:
		pickle.dump(encoder, fp)
		
	return label_encoder_path


def read_label_encoder(label_encoder_path) -> LabelEncoder:

	with open(label_encoder_path, mode="rb") as fp:
		loaded_encoder = pickle.load(fp)
		
	return loaded_encoder


def format_output_cols(df: pd.DataFrame):
	col_order = ['path', 'y', "family", "genus", "species", "collection", "catalog_number"]
	col_order = [col for col in col_order if col in df.columns]
	return df[col_order]


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Convert column dtypes to optimal type for Leavesdb metadata df.
	"""

	# Reduce total df size by optimizing dtypes per column
	cat_cols = ['y', "family", "genus", "species", "collection"]
	if "y" in df.columns:
		cat_cols.append("y")
	
	str_cols = ['path', "catalog_number"]
	col_dtypes = {c:"category" for c in cat_cols if c in df.columns}
	col_dtypes.update({c:"string" for c in str_cols})
	
	# import pdb;pdb.set_trace()

	df = df.astype(col_dtypes)
	return df


def read_df_from_csv(path,
					 nrows: Optional[int]=None,
                     index_col=None
					) -> pd.DataFrame:
	
	df = pd.read_csv(path, index_col=index_col, nrows=nrows)
	df = optimize_dtypes(df)
	return df




	

def make_splits(df: pd.DataFrame,
				label_col = "scientificName",
				splits: List[float]=(0.5, 0.2, 0.3),
				seed = 14
				) -> Tuple[LabelEncoder, pd.DataFrame]:
	
	
	split_dfs = trainvaltest_split(df=df,
								   label_col=label_col,
								   splits=splits,
								   seed=seed,
								   stratify=True)
	


	encoder, split_dfs = fit_and_encode_labels(train_data=split_dfs["train"],
											   val_data=split_dfs["val"],
											   test_data=split_dfs["test"],
											   label_col=label_col)

	split_dfs["train"] = format_output_cols(split_dfs["train"])
	split_dfs["val"] = format_output_cols(split_dfs["val"])
	split_dfs["test"] = format_output_cols(split_dfs["test"])
	
	return encoder, split_dfs

	
def make_encode_save_splits(source_csv_path: str, #DATA_DIR,
							save_dir: str, #=None,
							label_col: str="family",
							splits: List[float]=(0.5, 0.2, 0.3),
							seed = 14):
	save_dir = Path(save_dir)
	os.makedirs(save_dir, exist_ok=True)
	
	df = read_df_from_csv(path=source_csv_path)
	
	encoder, split_dfs = make_splits(df=df,
									 label_col = label_col,
									 splits=splits,
									 seed=seed)
	
	label_encoder_path = save_label_encoder(encoder=encoder,
											root_dir=save_dir,
											label_col=label_col)

	split_dfs["train"].to_csv(save_dir / "train_metadata.csv")
	split_dfs["val"].to_csv(save_dir / "val_metadata.csv")
	split_dfs["test"].to_csv(save_dir / "test_metadata.csv")

	return {"label_encoder":encoder,
			"subsets":{**split_dfs}
		   }


# def read_train_df_from_csv(train_path,
#							nrows: Optional[int]=None
#							) -> pd.DataFrame:

#	 df = pd.read_csv(train_path, index_col=0, nrows=nrows)
#	 df = optimize_dtypes_train(df)
#	 return df


# def read_test_df_from_csv(test_path,
#						   nrows: Optional[int]=None
#						   ) -> pd.DataFrame:

#	 df = pd.read_csv(test_path, index_col=0, nrows=nrows)
#	 df = optimize_dtypes_test(df)
#	 return df


def find_label_encoder_path(source_dir: str, label_col: Optional[str]=None) -> Path:
	"""
	Parse the contents of source_dir and return the full path of the encoder file, by filtering for files with "encoder" in the name.
	If label_col is not specified and function locates 2 or more files with "encoder" in the name, then raises an error.
	"""
	file_name = [f for f in os.listdir(source_dir) if "encoder" in f]
	if isinstance(label_col, str):
		file_name = [f for f in file_name if label_col in f]

	if len(file_name)==0:
		return None
	assert len(file_name) == 1, "Warning: found ambiguous label encoder files in data splits directory or failed to specify label_col for dataset w/ more than 1 encoder. Please inspect contents of directory and try again.\n" + f"{source_dir}"
	
	file_name = file_name[0]
	return Path(source_dir, file_name)


def find_data_splits_dir(source_dir: str,
						 splits: List[float]=(0.5, 0.2, 0.3)
						) -> Path:
	"""
	Given a base path of `source_dir`, construct the correct data split dir path using chosen train_size.
	"""
	splits_subdir = f"splits=({splits[0]:.1f},{splits[1]:.1f},{splits[2]:.1f})"

	if splits_subdir in str(source_dir):
		return source_dir

	out_dir = Path(source_dir)
	if ("splits" in os.listdir(str(out_dir))) or ("splits" not in str(out_dir)):
		out_dir = out_dir / "splits"
		
	out_dir = out_dir / splits_subdir
	
	return out_dir




def check_already_built(splits_dir: str, label_col: str="family") -> bool:
	"""
	Checks splits_dir for completed set of files as evidence that previous run was successful.
	"""
	splits_dir = Path(splits_dir)
	label_encoder_path = find_label_encoder_path(splits_dir, label_col=label_col)
	try:
		assert os.path.isfile(str(label_encoder_path))
		assert os.path.isfile(splits_dir / "train_metadata.csv")
		assert os.path.isfile(splits_dir / "val_metadata.csv")
		assert os.path.isfile(splits_dir / "test_metadata.csv")
		return True
	except AssertionError:
		return False
	

def read_encoded_splits(source_dir: str,
						label_encoder_path: str=None,
						label_col: str="family",
						include=["train","val","test"],
                        index_col: int=None):
	
	source_dir = Path(source_dir)
	
	if label_encoder_path is None:
		label_encoder_path = find_label_encoder_path(source_dir, label_col=label_col)
	
	encoder = read_label_encoder(label_encoder_path=label_encoder_path)
	data = {"label_encoder":encoder,
			"subsets":{}}
	
	if "train" in include:
		data["subsets"]["train"] = read_df_from_csv(source_dir / "train_metadata.csv", index_col=index_col)
	if "val" in include:
		data["subsets"]["val"] = read_df_from_csv(source_dir / "val_metadata.csv", index_col=index_col)
	if "test" in include:
		data["subsets"]["test"] = read_df_from_csv(source_dir / "test_metadata.csv", index_col=index_col)

	return data



def main(source_dir: str, #=DATA_DIR,
		 splits_dir: Optional[str]=None,
		 label_col: str="family",
		 splits: List[float]=(0.5, 0.2, 0.3),
		 seed: int=14):
	"""
	If splits already exist on disk, read and return their contents.
	If they dont, read original herbarium train metadata, apply train val split, and write contents to new directory
	
	Returns a fitted LabelEncoder object + 3 DataFrames,
	- The train & val DataFrames contain labeled/supervised datasets
	- The test DataFrames contain unlabeled/unsupervised datasets
	
	"""
	
	# if splits_dir is None:
	#	 splits_dir = Path(source_dir, "splits", f"train_size-{train_size}")
	#	 os.makedirs(splits_dir, exist_ok=True)
	# import pdb; pdb.set_trace()

	if True: #not check_already_built(splits_dir):
		print(f"Making & saving train-val-test splits in the following directory:",
			  '\n' + str(splits_dir))
		if not os.path.exists(splits_dir):
			print(f"Creating directory structure")
			os.makedirs(splits_dir, exist_ok=True)
	
		csv_filename = [f for f in os.listdir(source_dir) if f.endswith(".csv")][0]
	
		source_csv_path = os.path.join(source_dir, csv_filename)
		
		print(f"Reading from: {csv_filename}")
	
		data = make_encode_save_splits(source_csv_path=source_csv_path,
									   save_dir=splits_dir,
									   label_col=label_col,
									   splits=splits,
									   seed=seed)
		return data
	else:
		
		print(f"Already completed previous run. train-val-test splits are in the following directory:",
			  '\n' + str(splits_dir))
		return None


# HERBARIUM_ROOT = "/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize"
# WORKING_DIR = "/media/data/jacob/GitHub/image-utils/notebooks/herbarium_2022/"
# OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
# DATA_DIR = os.path.join(WORKING_DIR, "data")


# from dotenv import load_dotenv
# load_dotenv()

# HERBARIUM_ROOT_DEFAULT = os.environ.get("HERBARIUM_ROOT_DEFAULT")
# CATALOG_DIR = os.environ.get("CATALOG_DIR")
# SPLITS_DIR = os.environ.get("SPLITS_DIR")


CATALOG_ROOT_DIR = "/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1"
available_catalogs = [
	# 'Extant_Leaves_1024',
	# 'Extant_Leaves_512',
	'Extant_Leaves_family_100_1024',
	'Extant_Leaves_family_100_512',
	'Extant_Leaves_family_10_1024',
	'Extant_Leaves_family_10_512',
	'Extant_Leaves_family_3_1024',
	'Extant_Leaves_family_3_512',
	# 'Fossil_1024',
	# 'Fossil_512',
	'Fossil_family_3_1024',
	'Fossil_family_3_512',
	'PNAS_family_100_1024',
	'PNAS_family_100_512'
]





	
def parse_args() -> argparse.Namespace:
	
	parser = argparse.ArgumentParser("""Generate train-val-test splits dataset from Leavesdb v1.1 catalogs.""")
	parser.add_argument(
		"--root_dir", default=CATALOG_ROOT_DIR, help="Root directory where inidividual datasets each have their own subdir, within which lie the catalog files."
	)
	parser.add_argument(
		"--sub_dir", default="Extant_Leaves_family_10_512", help="Root directory where inidividual datasets each have their own subdir, within which lie the catalog files."
	)

	parser.add_argument(
		"--splits_dir",
		default=None, #SPLITS_DIR,
		help="Target directory in which to save the csv files for each split. ",
	)
	parser.add_argument(
		"--label_col",
		default="family", #SPLITS_DIR,
		help="The column to encode as labels & use for stratification.",
	)
	parser.add_argument(
		"--splits",
		default="0.5,0.2,0.3",
		type=str,
		help="3 floats representing the train, val, and test fractions of the data catalogs to be made.",
	)
	parser.add_argument(
		"--seed", default=14, type=int, help="Random seed."
	)
	parser.add_argument(
		"--run_all", action="store_true", help="Flag to create splits for all subdirs in args.root_dir."
	)

	parser.add_argument(
		"--info", action="store_true", help="Flag to print execution variables then quit without execution."
	)
	parser.add_argument(
		"--force_overwrite", action="store_true", help="Flag to allow removal of pre-existing output files if they already exist, instead of skipping creation during execution."
	)
	
	args = parser.parse_args()
	args.splits = tuple(float(frac) for frac in args.splits.split(","))
	
	if args.info:
		print("User passed --info, displaying execution args then exiting")
		pp(args)
		sys.exit(0)
	
	assert os.path.isdir(args.root_dir)
	
	return args




if __name__ == "__main__":
	
	args = parse_args()

			
	if args.run_all:
		print(f"--run_all was passed, generating train-val-test splits for {len(available_catalogs)} datasets.")
		for sub_dir in tqdm(available_catalogs):
			source_dir = os.path.join(args.root_dir, sub_dir)

			args.splits_dir = find_data_splits_dir(source_dir=source_dir,
												   splits=args.splits)
			print(f"Using args.splits_dir: {args.splits_dir}")
			os.makedirs(args.splits_dir, exist_ok=True)

			main(source_dir=source_dir,
				 splits_dir=args.splits_dir,
				 label_col=args.label_col,
				 splits=args.splits,
				 seed=args.seed)


	else:
		
		source_dir = os.path.join(args.root_dir, args.sub_dir)
		
		if args.splits_dir is None:
			args.splits_dir = find_data_splits_dir(source_dir=source_dir,
												   splits=args.splits)		
		print(f"Using args.splits_dir: {args.splits_dir}")
		os.makedirs(args.splits_dir, exist_ok=True)

		main(source_dir=source_dir,
			 splits_dir=args.splits_dir,
			 label_col=args.label_col,
			 splits=args.splits,
			 seed=args.seed)