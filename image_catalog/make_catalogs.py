"""

image-utils/image_catalog/make_catalogs.py

Author: Jacob A Rose
Created: Wednesday July 28th, 2021
Updated: Saturday October 15th, 2022

command-line script written in python to scan structured image directories to generate experiment directories containing csv datasets and yaml configs

Currently covers:
	- leavesdb v0_3
	- leavesdb v1_0

Work In Progress:
	- leavesdb v1_1

TODO:
	- Add to a make file in base directory
	- Add more flexible configuration



- Sunday June 12th, 2022: 
	Added
		* make_florissant_fossil
		* make_general_fossil


python "/media/data_cifs/projects/prj_fossils/users/jacob/github/image-utils/image_catalog/make_catalogs.py" --info

python "/media/data_cifs/projects/prj_fossils/users/jacob/github/image-utils/image_catalog/make_catalogs.py" --all

python "/media/data_cifs/projects/prj_fossils/users/jacob/github/image-utils/image_catalog/make_catalogs.py" --make_original




python "/media/data_cifs/projects/prj_fossils/users/jacob/github/image-utils/image_catalog/make_catalogs.py" --all -o "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/catalogs"

"""

import argparse
import collections
import os
import sys
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from rich import print as pp
from typing import *
import pandas as pd
from omegaconf import DictConfig
from PIL import Image

from image_catalog import catalog_registry
from image_catalog.utils import template_utils

from image_catalog.utils.etl_utils import ETL
from image_catalog.utils.common_catalog_utils import CSVDatasetConfig, ImageFileDatasetConfig, CSVDataset, ImageFileDataset

log = template_utils.get_logger(__file__)

#######################################################
#######################################################


def create_dataset_A_in_B(dataset_A, dataset_B) -> pd.DataFrame:

	A_w_B = dataset_A.intersection(dataset_B)

	columns = [*[col for col in A_w_B.columns if col.endswith("_x")], *["catalog_number"]]
	A_in_B = A_w_B.reset_index()[columns].sort_values("catalog_number")

	print(f"A_in_B.columns: {A_in_B.columns}")
	A_in_B = A_in_B.rename(columns={col: col.split("_x")[0] for col in A_in_B.columns})

	return A_in_B


#######################################################
#######################################################


def export_dataset_catalog_configuration(
	output_dir: str = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_1",
	base_dataset_name="Extant_Leaves",
	threshold=100,
	resolution=512,
	version: str = "v1_1",
	path_schema: str = "{family}_{genus}_{species}_{collection}_{catalog_number}",
):

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


##############################################


def export_composite_dataset_catalog_configuration(
	output_dir: str = ".",
	csv_cfg_path_A: str = None,
	csv_cfg_path_B: str = None,
	composition: str = "-",
) -> Tuple["CSVDataset", "CSVDatasetConfig"]:

	csv_config_A = CSVDatasetConfig.load(path=csv_cfg_path_A)
	csv_config_B = CSVDatasetConfig.load(path=csv_cfg_path_B)

	dataset_A = CSVDataset.from_config(csv_config_A)
	dataset_B = CSVDataset.from_config(csv_config_B)
	print(f"num_samples A: {len(dataset_A)}")
	print(f"num_samples B: {len(dataset_B)}")

	print(f"producing composition: {composition}")
	if composition == "-":
		dataset_A_composed_B = dataset_A - dataset_B
		full_name = f"{csv_config_A.full_name}_minus_{csv_config_B.full_name}"
		out_dir = os.path.join(output_dir, full_name)
		os.makedirs(out_dir, exist_ok=True)
		print(f"num_samples A-B: {len(dataset_A_composed_B)}")

	if composition == "intersection":
		dataset_A_composed_B = dataset_A.intersection(dataset_B)
		full_name = f"{csv_config_A.full_name}_w_{csv_config_B.full_name}"
		print(f"num_samples A_w_B: {len(dataset_A_composed_B)}")

		out_dir = os.path.join(output_dir, full_name)
		os.makedirs(out_dir, exist_ok=True)

	########################################
	A_in_B = create_dataset_A_in_B(dataset_A, dataset_B)
	csv_dataset_pathname = f"{csv_config_A.full_name}_in_{csv_config_B.full_name}"
	csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
	ETL.df2csv(A_in_B, path=csv_dataset_out_path)
	A_in_B_config = CSVDatasetConfig(
		full_name=csv_dataset_pathname, data_path=csv_dataset_out_path, subset_key="all"
	)
	csv_dataset_config_out_path = os.path.join(out_dir, f"A_in_B-CSVDataset-config.yaml")
	A_in_B_config.save(csv_dataset_config_out_path)

	#########################################
	B_in_A = create_dataset_A_in_B(dataset_B, dataset_A)

	csv_dataset_pathname = f"{csv_config_B.full_name}_in_{csv_config_A.full_name}"
	csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
	ETL.df2csv(B_in_A, path=csv_dataset_out_path)
	B_in_A_config = CSVDatasetConfig(
		full_name=csv_dataset_pathname, data_path=csv_dataset_out_path, subset_key="all"
	)

	csv_dataset_config_out_path = os.path.join(out_dir, f"B_in_A-CSVDataset-config.yaml")
	B_in_A_config.save(csv_dataset_config_out_path)

	#########################################

	inputs_dir = os.path.join(out_dir, "inputs")
	os.makedirs(os.path.join(inputs_dir, "A"), exist_ok=True)
	os.makedirs(os.path.join(inputs_dir, "B"), exist_ok=True)
	shutil.copyfile(csv_cfg_path_A, os.path.join(inputs_dir, "A", Path(csv_cfg_path_A).name))
	shutil.copyfile(csv_cfg_path_B, os.path.join(inputs_dir, "B", Path(csv_cfg_path_B).name))

	csv_dataset_pathname = f"{full_name}-full_dataset"
	csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
	ETL.df2csv(dataset_A_composed_B, path=csv_dataset_out_path)
	####################
	####################
	csv_dataset_config_out_path = os.path.join(out_dir, f"CSVDataset-config.yaml")
	A_composed_B_config = CSVDatasetConfig(
		full_name=full_name, data_path=csv_dataset_out_path, subset_key="all"
	)
	A_composed_B_config.save(csv_dataset_config_out_path)

	print(f"[FINISHED] DATASET: {full_name}")
	print(f"Newly created dataset assets located at:  {out_dir}")

	if composition == "-":
		dataset_A_composed_B = CSVDataset.from_config(A_composed_B_config)
		return dataset_A_composed_B, A_composed_B_config

	if composition == "intersection":
		return (A_in_B, B_in_A), A_composed_B_config


############################################
############################################
############################################
############################################


def make_all_original(args):
	#	 if "output_dir" not in args:
	#		 args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"

	output_dir = args.output_dir
	version = args.version

	base_dataset_name = ""  # original"
	#						   "Extant_Leaves",
	#						   "Florissant_Fossil",
	#						   "General_Fossil"]

	resolution = "original"
	path_schema = "{family}_{genus}_{species}_{collection}_{catalog_number}"

	print(f"Beginning make_all_original()")
	#	 for base_dataset_name in base_dataset_names:
	export_dataset_catalog_configuration(
		output_dir=output_dir,
		base_dataset_name=base_dataset_name,
		threshold=0,
		resolution=resolution,
		version=version,
		path_schema=path_schema,
	)

	print(f"FINISHED ALL IN original datasets")
	print("==" * 15)


def make_fossil(args):

	output_dir = args.output_dir
	version = args.version

	base_dataset_name = "Fossil"
	thresholds = [None, 3]
	resolutions = [512, 1024, 1536, 2048]
	path_schema = "{family}_{genus}_{species}_{collection}_{catalog_number}"

	print(
		f"Beginning make_fossil() for {len(resolutions)}x resolutions and {len(thresholds)}x"
		" thresholds"
	)

	for threshold in thresholds:
		for resolution in resolutions:
			export_dataset_catalog_configuration(
				output_dir=output_dir,
				base_dataset_name=base_dataset_name,
				threshold=threshold,
				resolution=resolution,
				version=version,
				path_schema=path_schema,
			)

	print(f"FINISHED ALL IN Fossil")
	print("==" * 15)
	

def make_general_fossil(args):

	output_dir = args.output_dir
	version = args.version

	base_dataset_name = "General_Fossil"
	thresholds = [None, 3, 10, 20, 50]
	resolutions = [512, 1024, 1536, 2048]
	path_schema = "{family}_{genus}_{species}_{collection}_{catalog_number}"

	print(
		f"Beginning make_general_fossil() for {len(resolutions)}x resolutions and {len(thresholds)}x"
		" thresholds"
	)

	for threshold in thresholds:
		for resolution in resolutions:
			export_dataset_catalog_configuration(
				output_dir=output_dir,
				base_dataset_name=base_dataset_name,
				threshold=threshold,
				resolution=resolution,
				version=version,
				path_schema=path_schema,
			)

	print(f"FINISHED ALL IN General Fossil")
	print("==" * 15)
	

######################


def make_florissant_fossil(args):

	output_dir = args.output_dir
	version = args.version

	base_dataset_name = "Florissant_Fossil"
	thresholds = [None, 3, 10, 20, 50]
	resolutions = [512, 1024, 1536, 2048]
	path_schema = "{family}_{genus}_{species}_{collection}_{catalog_number}"

	print(
		f"Beginning make_florissant_fossil() for {len(resolutions)}x resolutions and {len(thresholds)}x"
		" thresholds"
	)

	for threshold in thresholds:
		for resolution in resolutions:
			export_dataset_catalog_configuration(
				output_dir=output_dir,
				base_dataset_name=base_dataset_name,
				threshold=threshold,
				resolution=resolution,
				version=version,
				path_schema=path_schema,
			)

	print(f"FINISHED ALL IN Florissant Fossil")
	print("==" * 15)
	

######################

	

def make_extant(args):
	#	 if "output_dir" not in args:
	#		 args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"

	output_dir = args.output_dir
	version = args.version

	base_dataset_name = "Extant_Leaves"
	thresholds = [None, 3, 10, 100]
	resolutions = [512, 1024, 1536, 2048]
	path_schema = "{family}_{genus}_{species}_{collection}_{catalog_number}"

	print(
		f"Beginning make_extant() for {len(resolutions)}x resolutions and {len(thresholds)}x"
		" thresholds"
	)
	for threshold in thresholds:
		for resolution in resolutions:
			export_dataset_catalog_configuration(
				output_dir=output_dir,
				base_dataset_name=base_dataset_name,
				threshold=threshold,
				resolution=resolution,
				version=version,
				path_schema=path_schema,
			)

	print(f"FINISHED ALL IN Extant_Leaves")
	print("==" * 15)


def make_pnas(args):
	#	 if "output_dir" not in args:
	#		 args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"

	output_dir = args.output_dir
	version = args.version

	base_dataset_name = "PNAS"
	thresholds = [100]
	resolutions = [512, 1024, 1536, 2048]
	path_schema = "{family}_{genus}_{species}_{catalog_number}"

	print(
		f"Beginning make_pnas() for {len(resolutions)}x resolutions and {len(thresholds)}x"
		" thresholds"
	)
	for threshold in thresholds:
		for resolution in resolutions:
			export_dataset_catalog_configuration(
				output_dir=output_dir,
				base_dataset_name=base_dataset_name,
				threshold=threshold,
				resolution=resolution,
				version=version,
				path_schema=path_schema,
			)

	print(f"FINISHED ALL IN PNAS")
	print("==" * 15)


def make_extant_minus_pnas(args):
	#	 if "output_dir" not in args:
	#		 args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"

	output_dir = args.output_dir
	version = args.version

	base_names = {"A": "Extant_Leaves", "B": "PNAS"}
	thresholds = [{"A": 100, "B": 100}, {"A": 10, "B": 100}]
	resolutions = [512, 1024, 1536, 2048]
	class_type = "family"

	for threshold in thresholds:
		for resolution in resolutions:
			dataset_full_names = {
				"A": "_".join([base_names["A"], class_type, str(threshold["A"]), str(resolution)]),
				"B": "_".join([base_names["B"], class_type, str(threshold["B"]), str(resolution)]),
			}

			csv_cfg_path_A = os.path.join(
				output_dir, dataset_full_names["A"], "CSVDataset-config.yaml"
			)
			csv_cfg_path_B = os.path.join(
				output_dir, dataset_full_names["B"], "CSVDataset-config.yaml"
			)
			dataset, cfg = export_composite_dataset_catalog_configuration(
				output_dir=output_dir,
				csv_cfg_path_A=csv_cfg_path_A,
				csv_cfg_path_B=csv_cfg_path_B,
				composition="-",
			)

	print(f"FINISHED ALL IN Extant-PNAS")
	print("==" * 15)


def make_pnas_minus_extant(args):

	#	 if "output_dir" not in args:
	#		 args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"
	output_dir = args.output_dir

	base_names = {"A": "PNAS", "B": "Extant_Leaves"}
	thresholds = [{"A": 100, "B": 100}, {"A": 100, "B": 10}]
	resolutions = [512, 1024, 1536, 2048]
	class_type = "family"

	for threshold in thresholds:
		for resolution in resolutions:
			dataset_full_names = {
				"A": "_".join([base_names["A"], class_type, str(threshold["A"]), str(resolution)]),
				"B": "_".join([base_names["B"], class_type, str(threshold["B"]), str(resolution)]),
			}

			csv_cfg_path_A = os.path.join(
				output_dir, dataset_full_names["A"], "CSVDataset-config.yaml"
			)
			csv_cfg_path_B = os.path.join(
				output_dir, dataset_full_names["B"], "CSVDataset-config.yaml"
			)
			dataset, cfg = export_composite_dataset_catalog_configuration(
				output_dir=output_dir,
				csv_cfg_path_A=csv_cfg_path_A,
				csv_cfg_path_B=csv_cfg_path_B,
				composition="-",
			)

	print(f"FINISHED ALL IN Extant-PNAS")
	print("==" * 15)


def make_extant_w_pnas(args):

	#	 if "output_dir" not in args:
	#		 args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"
	output_dir = args.output_dir

	base_names = {"A": "Extant_Leaves", "B": "PNAS"}
	thresholds = [{"A": 100, "B": 100}, {"A": 10, "B": 100}]
	resolutions = [512, 1024, 1536, 2048]
	class_type = "family"

	for threshold in thresholds:
		for resolution in resolutions:
			dataset_full_names = {
				"A": "_".join([base_names["A"], class_type, str(threshold["A"]), str(resolution)]),
				"B": "_".join([base_names["B"], class_type, str(threshold["B"]), str(resolution)]),
			}

			csv_cfg_path_A = os.path.join(
				output_dir, dataset_full_names["A"], "CSVDataset-config.yaml"
			)
			csv_cfg_path_B = os.path.join(
				output_dir, dataset_full_names["B"], "CSVDataset-config.yaml"
			)
			dataset, cfg = export_composite_dataset_catalog_configuration(
				output_dir=output_dir,
				csv_cfg_path_A=csv_cfg_path_A,
				csv_cfg_path_B=csv_cfg_path_B,
				composition="intersection",
			)

	print(f"FINISHED ALL IN Extant_w_PNAS")
	print("==" * 15)


CSV_CATALOG_DIR_V0_3 = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"
CSV_CATALOG_DIR_V1_0 = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0"
CSV_CATALOG_DIR_V1_1 = "/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1"



EXPERIMENTAL_DATASETS_DIR = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/experimental_datasets"


def cmdline_args():
	p = argparse.ArgumentParser(
		description=(
			"Export a series of dataset artifacts (containing csv catalog, yml config, json labels)"
			" for each dataset, provided that the corresponding images are pointed to by one of the"
			" file paths hard-coded in catalog_registry.py."
		)
	)
	p.add_argument(
		"-o",
		"--output_dir",
		dest="output_dir",
		type=str,
		default="/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1",
		# default="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0",
		help=(
			"Output root directory. Each unique dataset will be allotted its own subdirectory"
			" within this root dir."
		),
	)
	p.add_argument(
		"-a",
		"--all",
		dest="make_all",
		action="store_true",
		help=(
			"If user provides this flag, produce all currently in-production datasets in the most"
			" recent version (currently == 'v1_1')."
		),
	)
	p.add_argument(
		"--seed", default=14, type=int, help="Random seed."
	)
	p.add_argument(
		"--info", action="store_true", help="Flag to print execution variables then quit without execution."
	)
	p.add_argument(
		"-v",
		"--version",
		dest="version",
		type=str,
		default="v1_1",
		help="Available dataset versions: [v0_3, v1_0, v1_1].",
	)
	p.add_argument(
		"--fossil",
		dest="make_fossil",
		action="store_true",
		help=(
			"If user provides this flag, produce all configurations of the combined Florissant +"
			" General Fossil collections."
		),
	)
	p.add_argument(
		"--general-fossil",
		dest="make_general_fossil",
		action="store_true",
		help=(
			"If user provides this flag, produce configurations of the General Fossil collection."
		),
	)
	p.add_argument(
		"--florissant-fossil",
		dest="make_florissant_fossil",
		action="store_true",
		help=(
			"If user provides this flag, produce configurations of the Florissant Fossil collection."
		),
	)
	p.add_argument(
		"--extant",
		dest="make_extant",
		action="store_true",
		help=(
			"If user provides this flag, produce all currently in-production resolutions+thresholds"
			" in the dataset: Extant_Leaves's most recent version (currently == 'v1_1')."
		),
	)
	p.add_argument(
		"--pnas",
		dest="make_pnas",
		action="store_true",
		help=(
			f"If user provides this flag, produce all currently in-production"
			f" resolutions+thresholds in the dataset: PNAS's most recent version (currently =="
			f" 'v1_1')."
		),
	)
	#					help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
	p.add_argument(
		"--extant-pnas",
		dest="make_extant_minus_pnas",
		action="store_true",
		help=(
			"If user provides this flag, produce all currently in-production resolutions+thresholds"
			" in the dataset: Extant-PNAS's most recent version (currently == 'v1_1')."
		),
	)
	#					help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
	p.add_argument(
		"--pnas-extant",
		dest="make_pnas_minus_extant",
		action="store_true",
		help=(
			"If user provides this flag, produce all currently in-production resolutions+thresholds"
			" in the dataset: PNAS-Extant's most recent version (currently == 'v1_1')."
		),
	)
	#					help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
	p.add_argument(
		"--extant-w-pnas",
		dest="make_extant_w_pnas",
		action="store_true",
		help=(
			"If user provides this flag, produce all currently in-production resolutions+thresholds"
			" in the dataset: Extant_w_PNAS's most recent version (currently == 'v1_1')."
		),
	)
	#					help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
	p.add_argument(
		"--original",
		dest="make_all_original",
		action="store_true",
		help=(
			f"If user provides this flag, produce all currently in-production"
			f" resolutions+thresholds in the original datasets' () most recent version (currently"
			f" == 'v1_1'). Note that this excludes PNAS."
		),
	)
	#					help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
	
	args = p.parse_args()
	
	if args.info:
		print("<---------[INFO]--------->")
		print("User passed --info, displaying execution args then exiting")
		pp(vars(args))
		sys.exit(0)
	
	assert os.path.isdir(args.output_dir)
	
	return args


# make_all_original

if __name__ == "__main__":
	#	 import sys
	#	 args = sys.argv

	args = cmdline_args()

	os.makedirs(args.output_dir, exist_ok=True)

	if args.make_fossil or args.make_all:
		make_fossil(args)
	if args.make_florissant_fossil or args.make_all:		
		make_florissant_fossil(args)
	if args.make_general_fossil or args.make_all:		
		make_general_fossil(args)
	if args.make_extant or args.make_all:
		make_extant(args)
	if args.make_pnas or args.make_all:
		make_pnas(args)
	if args.make_extant_minus_pnas or args.make_all:
		make_extant_minus_pnas(args)
	if args.make_pnas_minus_extant or args.make_all:
		make_pnas_minus_extant(args)
	if args.make_extant_w_pnas or args.make_all:
		make_extant_w_pnas(args)
	if args.make_all_original or args.make_all:
		make_all_original(args)


#######################################################
#######################################################
#######################################################
#######################################################