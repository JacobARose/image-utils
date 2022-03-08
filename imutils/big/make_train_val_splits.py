"""
make_train_val_splits.py

Load a csv file into a dataframe containing herbarium metadata, split it into train and val subsets, then write to new location as csv files.


Created on: Sunday March 6th, 2022
Created by: Jacob A Rose

"""


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Can we stratify by genus or Family while classifying species?
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
from typing import *
import json
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint as pp

from imutils.big.make_herbarium_2022_catalog_df import (read_all_from_csv,
                                                        read_train_df_from_csv,
                                                        read_test_df_from_csv)


HERBARIUM_ROOT = "/media/data_cifs/projects/prj_fossils/data/raw_data/herbarium-2022-fgvc9_resize"
WORKING_DIR = "/media/data/jacob/GitHub/image-utils/notebooks/herbarium_2022/"
OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
DATA_DIR = os.path.join(WORKING_DIR, "data")





def train_val_split(df: pd.DataFrame,
                    label_col = "scientificName",
                    train_size=0.7,
                    seed = 14
                    ) -> Tuple[pd.DataFrame]:

    num_samples = df.shape[0]
    # label_col = "category_id"
    x = np.arange(num_samples)
    y = df[label_col].values

    x_train, x_val, _, _ = train_test_split(x, y,
                                            stratify=y,
                                            train_size=train_size,
                                            random_state=seed)
    train_data = df.iloc[x_train,:]
    val_data = df.iloc[x_val,:]

    return train_data, val_data


def fit_and_encode_labels(train_data,
                          val_data,
                          label_col: str="scientificName"
                         ) -> Tuple[LabelEncoder, pd.DataFrame]:

    encoder = LabelEncoder()
    encoder.fit(train_data[label_col])
    train_data = train_data.assign(
        y = encoder.transform(train_data[label_col])
            ).astype({"y":"category"})
    val_data = val_data.assign(
        y = encoder.transform(val_data[label_col])
            ).astype({"y":"category"})

    return encoder, train_data, val_data


import pickle

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
    col_order = ['Species', 'path', 'y', 'category_id', 
                 'genus_id', 'institution_id', 'image_id', 'file_name',
                 'license', 'scientificName', 'family', 'genus', 'species', 'authors',
                 'collectionCode']
    return df[col_order]


# def make_splits_and_write2disk(root_dir: str, output_dir: str=None) -> Tuple[Path]:
#     """
#     Reads json metadata files from `root_dir`, parses into train & test dataframes, then writes to disk as csv files.
#     """
    
#     assert os.path.isdir(output_dir)
    
#     # df_train, df_test = extract_train_test_metadata(root_dir = root_dir)
#     train_path = Path(output_dir, "train_metadata.csv")
#     test_path = Path(output_dir, "test_metadata.csv")
    
#     # try:
#     if os.path.exists(train_path):
#         print(train_path, "already exists, skipping write process.",
#               "Delete the existing file if intention is to refresh dataset.")
#         print(f"Reading train data from: {train_path}")
#         df_train = read_train_df_from_csv(train_path)
#     else:
#         df_train = herbarium_train_metadata2df(root_dir)
#         df_train.to_csv(train_path)

#     if os.path.exists(test_path):
#         print(test_path, "already exists, skipping write process.",
#               "Delete the existing file if intention is to refresh dataset.")
#         print(f"Reading test data from: {test_path}")
#         df_test = read_test_df_from_csv(test_path)
#     else:
#         df_test = herbarium_test_metadata2df(root_dir)
#         df_test.to_csv(test_path)

#     return train_path, test_path






    

def make_splits(df: pd.DataFrame,
                label_col = "scientificName",
                train_size=0.7,
                seed = 14
                ) -> Tuple[LabelEncoder, pd.DataFrame]:
    
    train_data, val_data = train_val_split(df=df, #_train,
                                           label_col=label_col,
                                           train_size=train_size,
                                           seed=seed)

    encoder, train_data, val_data = fit_and_encode_labels(train_data,
                                                          val_data,
                                                          label_col=label_col)

    train_data = format_output_cols(train_data)
    val_data = format_output_cols(val_data)
    
    return encoder, train_data, val_data

    
def make_encode_save_splits(source_dir: str=DATA_DIR,
                            save_dir: str=None,
                            label_col: str="scientificName",
                            train_size=0.7,
                            seed = 14):
    
    df_train, df_test = read_all_from_csv(root_dir=source_dir)
    
    encoder, train_data, val_data = make_splits(df=df_train,
                                                label_col = label_col,
                                                train_size=train_size,
                                                seed=seed)

    
    label_encoder_path = save_label_encoder(encoder=encoder,
                                            root_dir=save_dir,
                                            label_col=label_col)

    train_data.to_csv(save_dir / "train_metadata.csv")
    val_data.to_csv(save_dir / "val_metadata.csv")
    df_test.to_csv(save_dir / "test_metadata.csv")

    return encoder, train_data, val_data, df_test


# def read_train_df_from_csv(train_path,
#                            nrows: Optional[int]=None
#                            ) -> pd.DataFrame:
    
#     df = pd.read_csv(train_path, index_col=0, nrows=nrows)
#     df = optimize_dtypes_train(df)
#     return df


# def read_test_df_from_csv(test_path,
#                           nrows: Optional[int]=None
#                           ) -> pd.DataFrame:
    
#     df = pd.read_csv(test_path, index_col=0, nrows=nrows)
#     df = optimize_dtypes_test(df)
#     return df



def read_encoded_splits(source_dir: str,
                        label_encoder_path: str):
    
    
    # out_dir = Path(source_dir, "splits", f"train_size-{train_size}")
    # os.makedirs(out_dir, exist_ok=True)
    
    encoder = read_label_encoder(label_encoder_path=label_encoder_path)
    
    train_data = read_train_df_from_csv(source_dir / "train_metadata.csv")
    val_data = read_train_df_from_csv(source_dir / "val_metadata.csv")
    df_test = read_test_df_from_csv(source_dir / "test_metadata.csv")

    # train_data = pd.read_csv(source_dir / "train_metadata.csv")
    # val_data = pd.read_csv(source_dir / "val_metadata.csv")
    # df_test = pd.read_csv(source_dir / "test_metadata.csv")

    return encoder, train_data, val_data, df_test



def main(source_dir: str=DATA_DIR,
         split_dir: Optional[str]=None,
         label_col: str="scientificName",
         train_size: float=0.7,
         seed: int=14):
    """
    If splits already exist on disk, read and return their contents.
    If they dont, read original herbarium train metadata, apply train val split, and write contents to new directory
    
    Returns a fitted LabelEncoder object + 3 DataFrames,
    - The train & val DataFrames contain labeled/supervised datasets
    - The test DataFrames contain unlabeled/unsupervised datasets
    
    """
    
    if split_dir is None:
        split_dir = Path(source_dir, "splits", f"train_size-{train_size}")
        os.makedirs(split_dir, exist_ok=True)

    if len(os.listdir(split_dir)) < 3:
        print(f"Making & saving train-val-test splits in the following directory:",
              '\n' + str(split_dir))
    
        encoder, train_data, val_data, df_test = make_encode_save_splits(source_dir=source_dir,
                                                                         save_dir=split_dir,
                                                                         label_col=label_col,
                                                                         train_size=train_size,
                                                                         seed=seed)

    else:
        
        print(f"Reading previously made train-val-test splits from the following directory:",
              '\n' + str(split_dir))
        
        label_name = f"{label_col}-encoder.pkl"
        label_encoder_path = Path(split_dir, label_name)
        
        encoder, train_data, val_data, df_test = read_encoded_splits(source_dir=split_dir,
                                                                     label_encoder_path=label_encoder_path)
        
    return encoder, train_data, val_data, df_test


if __name__ == "__main__":
    
    main(source_dir=DATA_DIR,
         label_col="scientificName",
         train_size=0.7,
         seed=14)