"""
make_train_val_splits.py

Load a csv file into a dataframe containing herbarium metadata, split it into train and val subsets, then write to new location as csv files.


Created on: Sunday March 6th, 2022
Created by: Jacob A Rose

"""

import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Can we stratify by genus or Family while classifying species?
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
from typing import *
import json
from functools import partial
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint as pp

from imutils.big.make_herbarium_2022_catalog_df import (read_all_from_csv,
														read_train_df_from_csv,
														read_test_df_from_csv)

from imutils.big.split_catalog_utils import (make_encode_save_splits,
											 find_data_splits_dir,
											 check_already_built,
											 TRAIN_KEY,
											 VAL_KEY,
											 TEST_KEY)



def format_output_cols(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Override/Rewrite this function for performing any preprocessing on the columns prior to any other steps.
	
	"""
	col_order = ['Species', 'path', 'y', 'category_id', 
				 'genus_id', 'institution_id', 'image_id', 'file_name',
				 'license', 'scientificName', 'family', 'genus', 'species', 'authors',
				 'collectionCode']
	col_order = [col for col in col_order if col in df.columns]
	return df[col_order].convert_dtypes()



def main(source_dir: str, #=DATA_DIR,
		 save_dir: Optional[str]=None,
		 label_col: str="scientificName",
		 stratify_col: str="scientificName",
		 train_size: float=0.8,
		 force_overwrite: bool=False,
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

	if check_already_built(save_dir):
		if force_overwrite:
			pp(f"force_overwrite is set to true, replacing previously existing files within: {save_dir}")
		else:
			print(f"Already completed previous run. train-val-test splits are in the following directory:",
				  '\n' + str(save_dir))
			return None
		
	source_csv_paths = [Path(source_dir, f) for f in os.listdir(source_dir) if f.endswith(".csv")]

	
	data = make_encode_save_splits(source_csv_paths=source_csv_paths,
								   save_dir=save_dir,
								   label_col=label_col,
								   stratify_col=stratify_col,
								   train_size=train_size,
								   seed=seed,
								   splits=None,
								   select_subsets=None,
								   subset_read_funcs={
									   TRAIN_KEY: read_train_df_from_csv,
									   TEST_KEY: read_test_df_from_csv},
								   read_csv_func=partial(read_all_from_csv, return_dict=True),
								   format_output_cols_func=format_output_cols)
	
	return data



# HERBARIUM_ROOT = "/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize"
# WORKING_DIR = "/media/data/jacob/GitHub/image-utils/notebooks/herbarium_2022/"
# OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
# DATA_DIR = os.path.join(WORKING_DIR, "data")


# from dotenv import load_dotenv
# load_dotenv()
import imutils

HERBARIUM_ROOT_DEFAULT = os.environ.get("HERBARIUM_ROOT_DEFAULT")
CATALOG_DIR = os.environ.get("CATALOG_DIR")
SPLITS_DIR = os.environ.get("SPLITS_DIR")

"""
python "/media/data_cifs/projects/prj_fossils/users/jacob/github/image-utils/imutils/big/make_train_val_splits.py" \
--source_dir "/media/data/home/jrose3/data_cache/herbarium-2022-fgvc9_resize-512" \
--train_size=0.8 \
--label_col="scientificName" \
--stratify_col="scientificName"
"""

"""
python "/media/data_cifs/projects/prj_fossils/users/jacob/github/image-utils/imutils/big/make_train_val_splits.py" \
--source_dir "/media/data/home/jrose3/data_cache/herbarium-2022-fgvc9_resize-512" \
--train_size=0.8 \
--label_col="family" \
--stratify_col="scientificName"
"""






def parse_args() -> argparse.Namespace:
	
	parser = argparse.ArgumentParser("""Generate train-val splits dataset from Herbarium 2022 train dataset.""")
	parser.add_argument(
		"--source_dir", default=CATALOG_DIR, help="directory where catalog csv files are read, followed by splitting"
	)
	parser.add_argument(
		"--save_dir",
		default=None, #SPLITS_DIR,
		help="Target directory in which to save the csv files for each split. ",
	)
	parser.add_argument(
		"--train_size",
		default=0.7,
		type=float,
		help="ratio of the train input csv to keep for train, with 1 - train_size kept for val.",
	)
	parser.add_argument(
		"--label_col",
		default="scientificName",
		help="The column to encode as labels. Set --stratify_by_col in order to specify how to determine balanced splits correctly.",
	)
	parser.add_argument(
		"--stratify_col",
		default="scientificName",
		help="The column to use for stratification. Set --label_col in order to specify which column to use as the basis for integer-encoded labels for single-label, multiclass classification.",
	)
	parser.add_argument(
		"--seed", default=14, type=int, help="Random seed."
	)
	parser.add_argument(
		"--info", action="store_true", help="Flag to print execution variables then quit without execution."
	)
	parser.add_argument(
		"--force_overwrite", action="store_true", help="Flag to allow removal of pre-existing output files if they already exist, instead of skipping creation during execution."
	)
	
	args = parser.parse_args()	
	args.train_size = float(args.train_size)
	
	
	if args.save_dir is None:
		args.save_dir = find_data_splits_dir(source_dir=args.source_dir,
											   stratify_col=args.stratify_col,
											   train_size=args.train_size,
											   splits=None)
		
	# train_size_str = f"train_size-{args.train_size:.1f}"
	# stratify_by_col_str = f"stratify_by_col-{args.stratify_by_col}"
	# if args.splits_dir is None:
	# 	if SPLITS_DIR is not None:
	# 		args.splits_dir = SPLITS_DIR
	# 	else:
	# 		args.splits_dir = Path(args.source_dir, "splits")
	# 		args.splits_dir = args.splits_dir / stratify_by_col_str / train_size_str

	
	if args.info:
		print("User passed --info, displaying execution args then exiting")
		pp(args)
		sys.exit(0)
	
	assert os.path.isdir(args.source_dir)
	
	print(f"Using args.splits_dir: {args.save_dir}")
	os.makedirs(args.save_dir, exist_ok=True)
	
	return args










if __name__ == "__main__":
	
	args = parse_args()
	
	main(source_dir=args.source_dir,
		 save_dir=args.save_dir,
		 label_col=args.label_col,
		 stratify_col=args.stratify_col,
		 train_size=args.train_size,
		 force_overwrite=args.force_overwrite,
		 seed=args.seed)
								   
								   