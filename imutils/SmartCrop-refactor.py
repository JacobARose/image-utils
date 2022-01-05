#!/usr/bin/env python
# coding: utf-8

# # SmartCrop refactor - Actor model
#
#
# Author: Jacob A Rose
# Created on: Thursday August 19th, 2021


import argparse
import logging
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL
import torch
from PIL import Image, ImageStat
from rich import print as pp
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm, trange

seed = 334455
random.seed(seed)
np.random.seed(seed)
# from ResizeRight.resize_right import resize_right, interp_methods
from functools import partial

import modin.pandas as pd
import ray
import torchvision
from imutils.ResizeRight.resize_right import (
    interp_methods,
    resize_right,
)
from modin.config import ProgressBar
from torch import nn
from torchvision import transforms, utils
from tqdm import tqdm

ProgressBar.enable()
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", 200)

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.cache_size = 0
# InteractiveShell.ast_node_interactivity = "all"

from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")
import json



from typing import *

import dask
import dask.array as da
import dask.dataframe as dd
from dask import delayed
from lightning_hydra_classifiers.utils.dataset_management_utils import (
    Extract as ExtractBase,
)
from lightning_hydra_classifiers.utils.dataset_management_utils import (
    DatasetFilePathParser,
    parse_df_catalog_from_image_directory,
)
####################################
####################################
from skimage import io
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from munch import Munch


totensor: Callable = torchvision.transforms.ToTensor()


def toPIL(img: torch.Tensor, mode="RGB") -> Callable:
    return torchvision.transforms.ToPILImage(mode)(img)


class NormalizeImage(nn.Module):
    @staticmethod
    def channelwise_min(img: torch.Tensor) -> torch.Tensor:
        return img.min(dim=1).values.min(dim=1).values

    @staticmethod
    def channelwise_max(img: torch.Tensor) -> torch.Tensor:
        return img.max(dim=1).values.max(dim=1).values

    @staticmethod
    def normalize_image(img: torch.Tensor) -> torch.Tensor:
        """Enforce pixel bounds to range [0.0, 1.0]"""
        img_min = NormalizeImage.channelwise_min(img).view(-1, 1, 1)
        img_max = NormalizeImage.channelwise_max(img).view(-1, 1, 1)
        return (img - img_min) / (img_max - img_min)


########################
########################


@dataclass
class CleverCropConfig:

    interp_method: Callable = interp_methods.cubic
    antialiasing: bool = True
    target_shape: Tuple[int] = (3, 128, 128)
    max_aspect_ratio: float = 1.2
    grayscale: bool = False
    normalize: bool = True


class CleverCrop:
    def __init__(
        self,
        interp_method=interp_methods.cubic,
        antialiasing=True,
        target_shape: Tuple[int] = (3, 128, 128),
        max_aspect_ratio: float = 1.2,
        grayscale: bool = False,
        normalize: bool = True,
    ):
        """[summary]

        Args:
            interp_method: Defaults to interp_methods.cubic.
            antialiasing (bool): Defaults to True.
            target_shape (Tuple[int], optional): [description]. Defaults to (3,128,128).
            grayscale (bool, optional): [description]. Defaults to False.
            max_aspect_ratio (float): Defaults to 1.2
            normalize (bool): Defaults to True
                If True, normalize final image tensor to the range [0.0, 1.0]

        """

        self.interp_method = interp_method
        self.antialiasing = antialiasing
        self.target_shape = target_shape
        self.num_output_channels = self.target_shape[0]
        self.max_aspect_ratio = max_aspect_ratio
        self.grayscale = grayscale
        self.normalize = normalize

        self.resize = partial(
            resize_right.resize,
            #                               out_shape=self.target_shape,
            interp_method=self.interp_method,
            antialiasing=self.antialiasing,
        )
        self.normalize_image = NormalizeImage.normalize_image

        print("CleverCrop.__init__() ->", repr(self))

    @staticmethod
    def aspect_ratio(img: torch.Tensor):

        minside = np.min(img.shape[1:])
        maxside = np.max(img.shape[1:])

        aspect_ratio = maxside / minside
        return aspect_ratio

    def __call__(
        self, img: torch.Tensor, target_shape: Optional[Tuple[int]] = None
    ) -> torch.Tensor:
        """[summary]

        Args:
            img (torch.Tensor): [description]
            target_shape: (Optional[Tuple[int]]): Optionally override this CleverCrop class instance's init value.

        Returns:
            torch.Tensor: [description]
        """
        target_shape = target_shape or self.target_shape

        minside = np.min(img.shape[1:])  # + 1
        maxside = np.max(img.shape[1:])  # + 1
        new_img = img

        aspect_ratio = maxside / minside
        if aspect_ratio > self.max_aspect_ratio:
            #             print(f'Aspect Ratio = {aspect_ratio:.2f} > max_aspect_ratio = {self.max_aspect_ratio}, stacking.')
            num_repeats = np.floor((maxside / minside))
            min_dim = np.argmin(img.shape[1:]) + 1
            for _ in range(int(num_repeats)):
                new_img = torch.cat([new_img, img], dim=min_dim)
        if maxside == img.shape[2]:
            #             print(f'maxside = {maxside} is width, rotating.')
            new_img = torch.rot90(new_img, k=1, dims=[1, 2])

        new_img = self.resize(new_img, out_shape=target_shape)

        if self.grayscale:
            num_output_channels = self.target_shape[0]
            new_img = torchvision.transforms.functional.rgb_to_grayscale(
                img=new_img, num_output_channels=self.num_output_channels
            )

        if self.normalize:
            new_img = self.normalize_image(new_img)

        return new_img

    def __repr__(self):
        return json.dumps(
            {
                "interp_method": str(self.interp_method),
                "antialiasing": self.antialiasing,
                "target_shape": self.target_shape,
                "num_output_channels": self.num_output_channels,
                "max_aspect_ratio": self.max_aspect_ratio,
                "grayscale": self.grayscale,
                "normalize": self.normalize,
            }
        )


os.environ[
    "TEST_IMAGES"
] = "/media/data/jacob/GitHub/lightning-hydra-classifiers/tests/test_images/"


class test_CleverCrop:

    test_data_dir: str = os.getenv("TEST_IMAGES")
    tall_img_path: str = os.path.join(test_data_dir, "tall_aspect_ratio_leaf_image.jpg")
    wide_img_path: str = os.path.join(test_data_dir, "wide_aspect_ratio_leaf_image.jpg")

    def __init__(self, res=256):
        self.seed = 333333
        self.target_shape = (3, res, res)
        self.tall_img = totensor(PIL.Image.open(self.tall_img_path))
        self.wide_img = totensor(PIL.Image.open(self.wide_img_path))

        self.config = CleverCropConfig(
            interp_method=interp_methods.cubic,
            antialiasing=True,
            target_shape=self.target_shape,
            max_aspect_ratio=1.2,
            grayscale=False,
            normalize=True,
        )

    def run(self):
        transform = CleverCrop(**asdict(self.config))

        tall_cropped = transform(self.tall_img)
        wide_cropped = transform(self.wide_img)

        print("Tall:")
        print(f"Input shape: {self.tall_img.shape}")
        print(f"Output shape: {tall_cropped.shape}")

        print("Wide:")
        print(f"Input shape: {self.wide_img.shape}")
        print(f"Output shape: {wide_cropped.shape}")

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.tall_img.permute(1, 2, 0))
        ax[1].imshow(tall_cropped.permute(1, 2, 0))

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.wide_img.permute(1, 2, 0))
        ax[1].imshow(wide_cropped.permute(1, 2, 0))


def split_df_into_chunks(data_df: pd.DataFrame, num_chunks: int) -> List[pd.DataFrame]:
    idx_chunks = [list(idx) for idx in more_itertools.divide(num_chunks, range(len(data_df)))]
    df_chunks = [data_df.iloc[idx, :] for idx in idx_chunks]
    return df_chunks



# @dask.delayed
def open_image(row: List[Dict[str, Any]]) -> np.ndarray:
    return io.imread(row["path"])


# @dask.delayed
def transform(
    img: np.ndarray, row: Dict[str, Any] = None, target_shape: Optional[Tuple[int]] = None
) -> np.ndarray:

    img = transforms.ToTensor()(img)
    img = clever_crop(img, target_shape=target_shape)
    img = (img * 255.0).to(torch.uint8).numpy()
    img = np.moveaxis(img, 0, -1)
    return img


# @dask.delayed
def write_jpeg(img: np.ndarray, row: Dict[str, Any]) -> bool:

    if os.path.isfile(row["target_path"]):
        return True
    try:
        io.imsave(row["target_path"], img, quality=100)
        out = os.path.isfile(row["target_path"])
    except Exception as e:
        print("Write image error:", row["target_path"], e)
        out = False
    return out


# @dask.delayed
def batch_ETL(batch_records, target_shape: Tuple[int]):

    resize_transform = partial(transform, target_shape=target_shape)
    imgs = []
    for i, rec in enumerate(batch_records):
        if os.path.isfile(rec["target_path"]):
            continue
        img = open_image(row=rec)
        img = resize_transform(img=img, row=rec)
        img = write_jpeg(img=img, row=rec)

        #         img = dask.delayed(open_image)(row=rec)
        #         img = dask.delayed(resize_transform)(img=img, row=rec)
        #         img = dask.delayed(write_jpeg)(img=img, row=rec)
        imgs.append(img)

    print(f"Computing {len(imgs)} images, Skipping {len(batch_records) - len(imgs)}")
    return imgs

# root_dir = "/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images"
# print(f'Initiating conversion of images to new image shape = {(3, config.resolution, config.resolution)}')
clever_crop = CleverCrop()  # target_shape=target_config.target_shape)


# ### Query catalog

# from lightning_hydra_classifiers.data.utils.catalog_registry import *
# from lightning_hydra_classifiers.utils.dataset_management_utils import (
#     Extract,  # as ExtractBase
# )

#     tag = available_datasets.query_tags(dataset_name=config.dataset_name,
#                                         y_col=config.y_col,
#                                         threshold=config.threshold,
#                                         resolution=config.resolution)
#     root_dir = available_datasets.get_latest(tag)
#     print(f"Tag: {tag}")
#     print(f"Root Dir: {root_dir}")
#     data_df = parse_df_catalog_from_image_directory(root_dir=root_dir, dataset_name=config.dataset_name)

#     target_dir = root_dir.replace("original", f"{target_config.resolution}")
#     data_df = data_df.assign(target_path = data_df.apply(lambda x: str(Path(target_dir, x.family, Path(x.path).name)), axis=1))
#     family_dirs = list(set(data_df.target_path.apply(lambda x: str(Path(x).parent))))
#     [os.makedirs(subdir, exist_ok=True) for subdir in family_dirs];


def query_and_preprocess_catalog(config, target_config):
    tag = available_datasets.query_tags(
        dataset_name=config.dataset_name,
        y_col=config.y_col,
        threshold=config.threshold,
        resolution=config.resolution,
    )
    root_dir = available_datasets.get_latest(tag)
    print(f"Tag: {tag}")
    print(f"Root Dir: {root_dir}")
    data_df = parse_df_catalog_from_image_directory(
        root_dir=root_dir, dataset_name=config.dataset_name
    )

    target_dir = root_dir.replace("original", f"{target_config.resolution}")
    data_df = data_df.assign(
        target_path=data_df.apply(
            lambda x: str(Path(target_dir, x.family, Path(x.path).name)), axis=1
        )
    )
    family_dirs = list(set(data_df.target_path.apply(lambda x: str(Path(x).parent))))
    [os.makedirs(subdir, exist_ok=True) for subdir in family_dirs]

    return data_df


# ### Dask Cluster
from dask.distributed import Client, LocalCluster, progress


def launch_dask_client(config):
    cluster = LocalCluster(
        dashboard_address=8989,
        # scheduler_port=8989,
        threads_per_worker=config.threads_per_worker,
        n_workers=config.num_cpus,
    )
    client = Client(cluster)

    return client


# ### Scratch
import dask.bag as db


def process_data_records(data_df: pd.DataFrame, config, target_config):
    clever_crop = CleverCrop(target_shape=target_config.target_shape)

    records = data_df.to_dict("records")  # [:500]
    record_bag = db.from_sequence(records, partition_size=config.chunksize)
    delayed_records = record_bag.to_delayed()
    pp(config)
    pp(target_config)
    delayed_results = []
    for i, rec in enumerate(delayed_records):
        delayed_results.append(
            dask.delayed(batch_ETL)(batch_records=rec, target_shape=target_config.target_shape)
        )

    results = dask.persist(*delayed_results)
    progress(delayed_results)
    print("success?")
    return dask.compute(*delayed_results)


def setup_configs(args):
    print("args:", vars(args))
    print(args.resolution)
    config = Munch(
        chunksize=args.chunksize,
        num_cpus=args.num_workers,
        threads_per_worker=args.threads_per_worker,
    )
    config.update(
        dict(dataset_name=args.dataset_name, y_col="family", threshold=0, resolution="original")
    )
    target_config = Munch(resolution=args.resolution, y_col="family", threshold=0)
    config.target_shape = (3, config.resolution, config.resolution)
    target_config.target_shape = (3, target_config.resolution, target_config.resolution)

    return config, target_config


def cmdline_args():
    p = argparse.ArgumentParser(
        description=(
            "Resize datasets to a new resolution, using the clever crop resize function. Requires"
            " source images to exist in {root_dir}/original/jpg and be organized into 1 subdir"
            " per-class."
        )
    )
    p.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        type=int,
        help="target resolution, images resized to (3, res, res).",
    )
    p.add_argument(
        "-n",
        "--dataset_name",
        dest="dataset_name",
        type=str,
        default="Extant_Leaves",
        help="""Base dataset_name to be used to query the source root_dir.""",
    )
    p.add_argument(
        "-a",
        "--run-all",
        dest="run_all",
        action="store_true",
        help=(
            "Flag for when user would like to run through all default threshold arguments on a"
            " given dataset. Currently includes resolutions = [512, 1024, 1536, 2048]."
        ),
    )
    p.add_argument("--num_workers", dest="num_workers", type=int, default=8)
    #                    help="Number of parallel processes to be used by pandas to efficiently construct symlinks.")
    p.add_argument("--threads_per_worker", dest="threads_per_worker", type=int, default=8)
    p.add_argument("--chunksize", dest="chunksize", type=int, default=50)
    return p.parse_args()


def main(args):

    config, target_config = setup_configs(args)
    data_df = query_and_preprocess_catalog(config, target_config)
    client = launch_dask_client(config)
    results = process_data_records(data_df, config, target_config)


if __name__ == "__main__":

    args = cmdline_args()
    print("args:", args)

    if args.run_all:
        logging.info(
            "[INITIATING] Creation of multiple dataset versions using smart-crop at the following"
            f" resolutions: {[512, 1024, 1536, 2048]}."
        )
        for resolution in [512, 1024, 1536, 2048]:
            args.resolution = resolution
            main(args)

    elif isinstance(args.resolution, int):
        main(args)

    print("Finished?")
