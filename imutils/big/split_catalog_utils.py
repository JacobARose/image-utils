"""

imutils/big/split_catalog_utils.py

Created On: Thursday April 21st, 2022  
Created By: Jacob A Rose


"""

from imutils.ml.utils import template_utils
# from imutils.ml.utils.label_utils import LabelEncoder
# from imutils import catalog_registry
# import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Can we stratify by genus or Family while classifying species?
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
from typing import *
# import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
# import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint as pp



log = template_utils.get_logger(__name__)

__all__ = ['train_val_split', 'trainvaltest_split', 
		   'read_label_encoder', 'save_label_encoder',
		   'fit_and_encode_labels', "format_output_cols",
		   'find_data_splits_dir', 'make_encode_save_splits',
		   "TRAIN_KEY", "VAL_KEY", "TEST_KEY"]

# __all__ += ["DataSplitter"]


TRAIN_KEY: str = "train"
VAL_KEY: str = "val"
TEST_KEY: str = "test"



def train_val_split(df: pd.DataFrame,
					stratify_col: str = "scientificName",
					train_size: float=0.7,
					seed: int= 14,
					return_dict: bool=False
					) -> Union[Dict[str,pd.DataFrame],
							   Tuple[pd.DataFrame]]:

	num_samples = df.shape[0]
	x = np.arange(num_samples)
	y = df[stratify_col].values

	x_train, x_val, _, _ = train_test_split(x, y,
											stratify=y,
											train_size=train_size,
											random_state=seed)
	train_data = df.iloc[x_train,:]
	val_data = df.iloc[x_val,:]

	if return_dict:
		return {
			TRAIN_KEY: train_data,
			VAL_KEY: val_data
		}

	return train_data, val_data

####################################################

def trainvaltest_split(df: pd.DataFrame,
					   stratify_col: str = "family",
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
	
	if stratify and (y is None):
		raise ValueError("If y is not provided, stratify must be set to False.")
	
	num_samples = df.shape[0]
	x = np.arange(num_samples)
	y = df[stratify_col].values

	stratify_y = y if stratify else None	
	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y,
														test_size=test_split, 
														random_state=random_state,
														stratify=stratify_y)
														# stratify=y)
	
	stratify_y_train = y_train_val if stratify else None
	x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
													  test_size=val_relative_split,
													  random_state=random_state, 
													  stratify=stratify_y_train)
													  # stratify=y_train_val)
	
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
						  val_data=None,
						  test_data=None,
						  label_col: str="scientificName",
						  transformed_label_col: str="y",
						 ) -> Tuple[LabelEncoder, pd.DataFrame]:
	"""
	
	Return:
		(label encoder, output_data)
			- label encoder: a scikit-learn LabelEncoder fit on the train_data
			output_data: 1-3 dataframes, depending on whether val_data & test_data were passed in function's arguments.
	
	"""

	encoder = LabelEncoder()
	encoder.fit(train_data[label_col])

	train_data = train_data.assign(**{
		transformed_label_col: encoder.transform(train_data[label_col])
	}).astype({
		transformed_label_col: "category"
	})
	
	output_data = [train_data]
	
	if isinstance(val_data, pd.DataFrame):
		val_data = val_data.assign(**{
			transformed_label_col: encoder.transform(val_data[label_col])
		}).astype({
			transformed_label_col: "category"
		})
		output_data.append(val_data)

	if isinstance(test_data, pd.DataFrame):
		if label_col in test_data.columns:
			test_data = test_data.assign(**{
				transformed_label_col: encoder.transform(test_data[label_col])
			}).astype({
				transformed_label_col: "category"
			})
			output_data.append(test_data)

	
	
	return encoder, output_data



# def fit_and_encode_labels(train_data,
# 						  val_data,
# 						  label_col: str="scientificName",
# 						  transformed_label_col: str="y",
# 						 ) -> Tuple[LabelEncoder, pd.DataFrame]:

# 	encoder = LabelEncoder()
# 	encoder.fit(train_data[label_col])
# 	train_data = train_data.assign(
# 		y = encoder.transform(train_data[label_col])
# 			).astype({"y":"category"})
# 	val_data = val_data.assign(
# 		y = encoder.transform(val_data[label_col])
# 			).astype({"y":"category"})

# 	return encoder, train_data, val_data


import pickle

def save_label_encoder(encoder,
					   root_dir: str,
					   label_col: str) -> str:
	label_name = f"{label_col}-encoder.pkl"
	label_encoder_path = Path(root_dir, label_name)

	with open(label_encoder_path, mode="wb") as fp:
		pickle.dump(encoder, fp)
		
	return label_encoder_path

def read_label_encoder(label_encoder_path: str) -> LabelEncoder:

	with open(label_encoder_path, mode="rb") as fp:
		loaded_encoder = pickle.load(fp)
		
	return loaded_encoder







def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Override/Rewrite this function for performing any preprocessing on the columns prior to any other steps.
	
	"""
	# col_order = ['Species', 'path', 'y', 'category_id', 
	# 			 'genus_id', 'institution_id', 'image_id', 'file_name',
	# 			 'license', 'scientificName', 'family', 'genus', 'species', 'authors',
	# 			 'collectionCode']
	# col_order = [col for col in col_order if col in df.columns]
	return df.convert_dtypes()

def read_df_from_csv(path,
					 nrows: Optional[int]=None,
					 index_col=None
					) -> pd.DataFrame:
	"""
	default csv reader, user should override for custom preprocessing.
	"""
	df = pd.read_csv(path, index_col=index_col, nrows=nrows)
	df = optimize_dtypes(df)
	return df


def read_all_from_csv(source_csv_paths: List[str],
					  select_subsets: List[str]=None,
					  subset_read_funcs: Union[Callable, Dict[str, Callable]]=read_df_from_csv
					 ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
	"""
	Read the train_metadata.csv and test_metadata.csv files from `root_dir`
	
	Note: This is prior to any train-val splits.
	"""
	# select_subsets = select_subsets or []
	# files = [f for f in os.listdir(root_dir) if f.endswith(".csv")]
	
	# source_csv_path=source_csv_path
	paths = {}
	output_data = {}
	if len(source_csv_paths)==1:
		paths = source_csv_paths[0]
		read_func = subset_read_funcs
		output_data = read_func(paths)
	else:
		for item in source_csv_paths:
			subset_key = item.split("_")[0]
			if isinstance(select_subsets, List):
				if subset_key in select_subsets:
					paths[subset_key] = Path(root_dir, item)
			else:
				paths[subset_key] = Path(root_dir, item)
		for subset_key in paths.keys():
			if isinstance(subset_read_funcs, Dict):
				read_func = subset_read_funcs.get(subset_key, list(subset_read_funcs.values())[0])
			else:
				read_func = subset_read_funcs
			output_data[subset_key] = read_func(paths[subset_key])
		
	return output_data



def make_splits(df: pd.DataFrame,
				label_col: str="scientificName",
				stratify_col: str="scientificName",
				train_size: Optional[float]=0.7,
				splits: Optional[Tuple[float]]=None,
				seed: int=14,
				format_output_cols_func: Callable=lambda x: x,
				format_output_test_cols_func: Optional[Callable]=None,
				) -> Tuple[LabelEncoder, Dict[str, pd.DataFrame]]:
	
	assert splits is None or train_size is None, ValueError("Either `splits` or `train_size should be specified, the other must be set to None.")
	
	if splits is None:
		train_data, val_data = train_val_split(df=df, #_train,
											   stratify_col=stratify_col,
											   train_size=train_size,
											   seed=seed,
											   return_dict=False)

		encoder, output_data = fit_and_encode_labels(train_data=train_data,
													 val_data=val_data,
													 label_col=label_col)
		
		subset_keys = [TRAIN_KEY, VAL_KEY]
		output_data = {subset_key: subset for subset_key, subset in list(zip(subset_keys, output_data))}
		for i, subset in tqdm(list(output_data.items()), desc=f"{train_size=}"):
			output_data[i] = format_output_cols_func(subset)

	else:
		output_data = trainvaltest_split(df=df,
									   stratify_col=stratify_col,
									   splits=splits,
									   seed=seed)

		encoder, output_data = fit_and_encode_labels(train_data=output_data["train"],
													 val_data=output_data["val"],
													 test_data=output_data["test"],
													 label_col=label_col)

		for subset_key, subset in tqdm(list(output_data.items()), desc=f"{splits=}"):
			if subset_key == "test" and isinstance(format_output_test_cols_func, Callable):
				output_data[subset_key] = format_output_test_cols_func(subset)
			else:
				output_data[subset_key] = format_output_cols_func(subset)		
		
	
	return encoder, output_data

	
def make_encode_save_splits(#source_dir: str, #DATA_DIR,
							source_csv_paths: List[str],
							save_dir: str,
							label_col: str="scientificName",
							stratify_col: str="scientificName",
							train_size: Optional[float]=None,
							splits: Optional[Tuple[float]]=None,
							seed: int=14,
							select_subsets: List[str]=None,
							subset_read_funcs: Union[Callable, Dict[str, Callable]]=None,
							read_csv_func: Callable=None,
							format_output_cols_func: Callable=lambda x: x,
					 ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
	
	from imutils.big.make_herbarium_2022_catalog_df import read_all_from_csv
	

	subset_read_funcs = subset_read_funcs or read_df_from_csv
	read_csv_func = read_csv_func or read_all_from_csv
	
	
	save_dir = Path(save_dir)
	os.makedirs(save_dir, exist_ok=True)
	
	source_data = read_csv_func(source_csv_paths=source_csv_paths,
									select_subsets=select_subsets,
									subset_read_funcs=subset_read_funcs)
	output_data = {}

	# import pdb;pdb.set_trace()
	
	if isinstance(source_data, Dict):
		train_df = source_data[TRAIN_KEY]
		if TEST_KEY in source_data:
			output_data[TEST_KEY] = source_data[TEST_KEY]
	else:
		train_df = source_data
	assert VAL_KEY not in source_data
	
	
	encoder, data_splits = make_splits(df=train_df,
									   label_col=label_col,
									   stratify_col=stratify_col,
									   train_size=train_size,
									   splits=splits,
									   seed=seed,
									   format_output_cols_func=format_output_cols_func)
	
	output_data.update(data_splits)

	label_encoder_path = save_label_encoder(encoder=encoder,
											root_dir=save_dir,
											label_col=label_col)

	output_data[TRAIN_KEY].to_csv(save_dir / "train_metadata.csv")
	output_data[VAL_KEY].to_csv(save_dir / "val_metadata.csv")
	output_data[TEST_KEY].to_csv(save_dir / "test_metadata.csv")

	return {"label_encoder":encoder,
			"subsets":{
				TRAIN_KEY: output_data[TRAIN_KEY],
				VAL_KEY: output_data[VAL_KEY],
				TEST_KEY: output_data[TEST_KEY]
	}
		   }


	



# def read_train_df_from_csv(train_path,
#							nrows: Optional[int]=None
#							) -> pd.DataFrame:
	
#	 df = pd.read_csv(train_path, index_col=0, nrows=nrows)
#	 df = optimize_dtypes_train(df)
#	 return df


def find_label_encoder_path(source_dir: str, 
							label_col: Optional[str]=None) -> Path:
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
						 stratify_col: str="scientificName",
						 train_size: float=None,
						 splits: Tuple[float]=None): #(0.5, 0.2, 0.3)) -> Path:
	"""
	Given a base path of `source_dir`, construct the correct data split dir path using chosen train_size.
	
	[TODO] Formalize the test cases for this function
	"""

	source_dir = Path(source_dir)
	stratify_str = f"stratify-{stratify_col}"
	if "splits" not in str(source_dir):
		source_dir = source_dir / "splits"
	# os.makedirs(source_dir, exist_ok=True)

	if isinstance(train_size, float) and splits is None:	
		splits_subdir_str = f"train_size-{train_size:.1f}"
	else:
		splits_subdir_str = f"splits-({splits[0]:.1f},{splits[1]:.1f},{splits[2]:.1f})"
		
	splits_subdir_str_base = splits_subdir_str.split("-")[0] + "-"
		

	if splits_subdir_str in str(source_dir):
		# print(0)
		return source_dir
	elif splits_subdir_str_base in source_dir.name:
		# print(1)
		return source_dir.parent / splits_subdir_str
	if stratify_str in str(source_dir.name):
		# print(2)
		return source_dir / splits_subdir_str
	elif "stratify" in str(source_dir.name):
		# print(3)
		return source_dir.parent / stratify_dir / splits_subdir_str
	else:
		# print(4)
		return source_dir / stratify_str / splits_subdir_str











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
						label_col: str="scientificName",
						include=["train","val","test"],
						index_col: int=0):
	
	from imutils.big.make_herbarium_2022_catalog_df import (read_all_from_csv,
															read_train_df_from_csv,
															read_test_df_from_csv)

	
	source_dir = Path(source_dir)
	
	if label_encoder_path is None:
		label_encoder_path = find_label_encoder_path(source_dir, label_col=label_col)
	
	encoder = read_label_encoder(label_encoder_path=label_encoder_path)
	data = {"label_encoder":encoder,
			"subsets":{}}
	
	if "train" in include:
		data["subsets"]["train"] = read_train_df_from_csv(source_dir / "train_metadata.csv", index_col=index_col)
	if "val" in include:
		data["subsets"]["val"] = read_train_df_from_csv(source_dir / "val_metadata.csv", index_col=index_col)
	if "test" in include:
		data["subsets"]["test"] = read_test_df_from_csv(source_dir / "test_metadata.csv", index_col=index_col)

	return data