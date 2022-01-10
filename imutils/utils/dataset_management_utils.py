"""
image-utils/imutils/dataset_management_utils.py


Created on: Tuesday, July 27th, 2021
Author: Jacob A Rose


"""

import collections
import dataclasses
import json
import numbers
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torchdatasets as torchdata
from more_itertools import collapse, flatten
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

from imutils.utils.etl_utils import Extract
from imutils.utils.SmartCrop import CleverCrop

__all__ = [
    "CleverCrop",
    "Extract",
    "DatasetFilePathParser",
    "parse_df_catalog_from_image_directory",
    "dataframe_difference",
    "diff_dataset_catalogs",
]


##################
##################


import argparse
import sys
from typing import *


class DatasetFilePathParser:
    @classmethod
    def get_parser(cls, dataset_name: str) -> Dict[str, Callable]:
        if "Extant_Leaves" in dataset_name:
            return cls().ExtantLeavesParser
        if "Fossil" in dataset_name:
            return cls().FossilParser
        if "PNAS" in dataset_name:
            return cls().PNASParser

    #     @property
    @classmethod
    def parse_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.astype(
            {
                "path": pd.StringDtype(),
                "family": pd.CategoricalDtype(),
                "genus": pd.CategoricalDtype(),
                "species": pd.CategoricalDtype(),
                "catalog_number": pd.StringDtype(),
                "relative_path": pd.StringDtype(),
                "root_dir": pd.CategoricalDtype(),
            }
        )

    @property
    def ExtantLeavesParser(self):
        return {
            "family": lambda x, col: Path(x[col]).stem.split("_")[0],
            "genus": lambda x, col: Path(x[col]).stem.split("_")[1],
            "species": lambda x, col: Path(x[col]).stem.split("_")[2],
            "catalog_number": lambda x, col: Path(x[col]).stem.split("_", maxsplit=4)[-1],
            "relative_path": lambda x, col: str(
                Path(x[col]).relative_to(Path(x[col]).parent.parent)
            ),
            "root_dir": lambda x, col: str(Path(x[col]).parent.parent),
        }

    @property
    def FossilParser(self):
        return {
            "family": lambda x, col: Path(x[col]).stem.split("_")[0],
            "genus": lambda x, col: Path(x[col]).stem.split("_")[1],
            "species": lambda x, col: Path(x[col]).stem.split("_")[2],
            "catalog_number": lambda x, col: Path(x[col]).stem.split("_", maxsplit=4)[-1],
            "relative_path": lambda x, col: str(
                Path(x[col]).relative_to(Path(x[col]).parent.parent)
            ),
            "root_dir": lambda x, col: str(Path(x[col]).parent.parent),
        }

    @property
    def PNASParser(self):
        return {
            "family": lambda x, col: Path(x[col]).stem.split("_")[0],
            "genus": lambda x, col: Path(x[col]).stem.split("_")[1],
            "species": lambda x, col: Path(x[col]).stem.split("_")[2],
            "catalog_number": lambda x, col: Path(x[col]).stem.split("_", maxsplit=3)[-1],
            "relative_path": lambda x, col: str(
                Path(x[col]).relative_to(Path(x[col]).parent.parent)
            ),
            "root_dir": lambda x, col: str(Path(x[col]).parent.parent),
        }


def parse_df_catalog_from_image_directory(
    root_dir: str, dataset_name: str = "Extant_Leaves"
) -> pd.DataFrame:
    """
    Crawls root_dir and collects absolute paths of any images into a dataframe. Then, extracts
    maximum available metadata from file paths (e.g. family, species labels in file name).

    Metadata fields in each file path are specified by using a DatasetFilePathParser object.

    Arguments:

        root_dir (str):
            Location of the Imagenet-format organized image data on disk
    Returns:
        data_df (pd.DataFrame):


    """

    parser = DatasetFilePathParser().get_parser(dataset_name)
    data_df = Extract.df_from_dir(root_dir)["all"]
    if data_df.shape[0] == 0:
        print("Empty data catalog, skipping parsing step.")
        return data_df
    for col, func in parser.items():
        print(col)
        data_df = data_df.assign(**{col: data_df.apply(lambda x: func(x, "path"), axis=1)})

    data_df = DatasetFilePathParser.parse_dtypes(data_df)
    return data_df


def dataframe_difference(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    id_col: str = "relative_path",
    keep_cols: Optional[List[str]] = None,
):
    """
    Find rows which are different between two DataFrames.

    Example:

        shared, diff, source_only, target_only = dataframe_difference(source_df=data_df,
                                                                      target_df=target_data_df,
                                                                      id_col="relative_path",
                                                                      keep_cols=["path"])
    """
    keep_cols = keep_cols or []
    #     import pdb;pdb.set_trace()

    comparison_df = source_df.merge(
        target_df.loc[:, [id_col, *keep_cols]], how="outer", on=id_col, indicator=True
    )

    comparison_df = comparison_df.replace({"left_only": "source_only", "right_only": "target_only"})

    shared = comparison_df[comparison_df["_merge"] == "both"]
    diff = comparison_df[comparison_df["_merge"] != "both"]
    source_only = comparison_df[comparison_df["_merge"] == "source_only"].rename(
        columns={"path_x": "path"}
    )
    target_only = comparison_df[comparison_df["_merge"] == "target_only"].rename(
        columns={"path_y": "path"}
    )

    return shared, diff, source_only, target_only


def diff_dataset_catalogs(
    source_catalog: pd.DataFrame, target_catalog: pd.DataFrame
) -> Tuple[pd.DataFrame]:
    """
    Find the shared and unique rows between 2 dataframes based on the "relative_path" column.
    """

    shared, diff, source_only, target_only = dataframe_difference(
        source_df=source_catalog,
        target_df=target_catalog,
        id_col="relative_path",
        keep_cols=["path", "catalog_number"],
    )

    num_preexisting = sum([shared.shape[0] + target_only.shape[0]])
    if num_preexisting > 0:
        print(f"Found {num_preexisting} previously generated files in target location.")
        print(
            f"""
        shared: {shared.shape[0]}
        diff: {diff.shape[0]}
        source_only: {source_only.shape[0]}
        target_only: {target_only.shape[0]}
        """
        )
    else:
        print(f"No previously generated files found in target location.")

    return shared, diff, source_only, target_only
