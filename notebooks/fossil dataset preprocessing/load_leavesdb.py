"""

load_leavesdb.py


Helper script


Contains functions based on prototypes originally developed in a jupyter notebook for loading leavesdb datasets.



Created by: Jacob A Rose
Created on: August 7th, 2022


"""


import os
from rich import print as pp

import numpy as np
from typing import *
import inspect
from tqdm.auto import tqdm
from pathlib import Path
import logging
from imutils.catalog_registry import available_datasets
from imutils.big.common_catalog_utils import DataETL


if LOAD_LEAVESDB:
    dataset_catalog_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1"
    dataset_names = sorted(os.listdir(dataset_catalog_dir))
    main_datasets = [d for d in dataset_names if (not "_minus_" in d) and (not "_w_" in d) and (not "original" in d) and ("512" in d) and ("family" in d)]

    %%time
    # data_dirs = [Path(dataset_catalog_dir, d) for d in main_datasets]
    data_assets = [
        {"config_path": Path(dataset_catalog_dir, d, "CSVDataset-config.yaml"),
         "dataset_name": d}
        for d in main_datasets
    ]

    datasets = {}
    for asset in tqdm(data_assets):
        datasets[asset["dataset_name"]] = DataETL.import_dataset_state(**asset)
        pp(asset["dataset_name"])

    print(len(datasets))
    # datasets = {k:v for k, v in datasets.items() if "512" in k}
    pp(list(datasets.keys()))