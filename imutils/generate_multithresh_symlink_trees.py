"""

generate_multithresh_symlink_trees.py

Created On: Wednesday Aug 11th, 2021
Created By: Jacob A Rose

Summary: This script takes a source dataset of images organized into class-wise subdirs, and produces a set of symlink trees linking to it, each one containing only the classes that have at least as many images as that version's threshold.

# print all directories and quit before launch
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multithresh_symlink_trees.py" --dry-run -a


# Clean, & create, all symlink dirs.
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multithresh_symlink_trees.py" --task "clean+create" -a

# Clean, then create, all symlink dirs.
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multithresh_symlink_trees.py" --task clean -a
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multithresh_symlink_trees.py" --task create -a

"""
import argparse
import contextlib
import io
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from torch.utils.data import Dataset, Subset, random_split, DataLoader
import torch
import torchvision
from pandarallel import pandarallel
from plumbum import local
from rich import print as pp
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

tqdm.pandas()


############################


class CatalogDataset(ImageFolder):

    columns = (
        "absolute_path",
        "family",
        "genus",
        "species",
        "collection",
        "catalog_number",
        "relative_path",
        "root_dir",
    )
    root = None

    def __init__(
        self,
        data_catalog: pd.DataFrame,
        path_col: str = "absolute_path",
        label_col: str = "family",
        class2idx: Optional[Dict[str, int]] = None,
    ):
        #         super().__init__('.')

        self.data = data_catalog
        self.records = self.data.to_records()

        if not isinstance(class2idx, Dict):
            class2idx = {
                label: idx for idx, label in enumerate(sorted(set(data_catalog[label_col])))
            }
        self.class2idx = class2idx

        self.paths = self.records[path_col].tolist()
        self.targets = [self.class2idx[label] for label in self.records[label_col]]
        self.samples = [(path, label_idx) for path, label_idx in zip(self.paths, self.targets)]

        if data.root_dir.nunique() == 1:
            self.root = data.root_dir[0]

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        x, y = self.samples[index]

        x = Image.open(x)

        x = self.transforms(x)
        return (x, y)

    def __len__(self):
        return len(self.records)


def dataset2catalog(
    root_dir: Optional[Union[str, Path]] = None, dataset: Optional[torch.utils.data.Dataset] = None
) -> pd.DataFrame:

    """
    Simple wrapper around torchvision.datasets.ImageFolder
    """
    #     if root_dir is None and dataset is None:
    #         print("Error: Either root_dir or dataset cannot be None")
    #         return
    if root_dir is None:
        if dataset is None:
            print("Error: Either root_dir or dataset cannot be None")
            return
        root_dir = dataset.root
    elif isinstance(root_dir, str):
        if isinstance(dataset, torch.utils.data.Dataset):
            print(
                "Warning: Both root_dir and dataset provided, ignoring dataset and building from"
                " root_dir"
            )
        dataset = get_image_dataset(root_dir)

    #     dataset = get_image_dataset(root_dir)
    classes = dataset.classes
    samples = pd.DataFrame(dataset.samples, columns=["absolute_path", "family_idx"])

    samples = samples.assign(
        family=samples.family_idx.apply(lambda x: classes[x]).astype(pd.CategoricalDtype())
    ).drop(columns=["family_idx"])

    samples = samples.assign(
        genus=samples.absolute_path.apply(lambda x: Path(x).stem.split("_")[1]).astype(
            pd.CategoricalDtype()
        ),
        species=samples.absolute_path.apply(lambda x: Path(x).stem.split("_")[2]).astype(
            pd.CategoricalDtype()
        ),
        collection=samples.absolute_path.apply(lambda x: Path(x).stem.split("_")[3]).astype(
            pd.CategoricalDtype()
        ),
        catalog_number=samples.absolute_path.apply(
            lambda x: Path(x).stem.split("_", maxsplit=4)[-1]
        ).astype(pd.StringDtype()),
        relative_path=samples.absolute_path.apply(
            lambda x: str(Path(x).relative_to(root_dir))
        ).astype(pd.StringDtype()),
        root_dir=root_dir,
    )
    return samples


def plot_kfold_class_distributions(
    y: List[int], kfolds: int = 10, seed: int = None, name: str = None, bins: int = None
) -> np.array:
    """
    Create k-stratified folds of a list of int labels `y`, plot their distributions per-foldm, and return a dataframe containing indices as values, and each fold as a separate column.

    Arguments:
        y: List[int]
            The integer-encoded class labels for N samples contained in their true order, in a list.
            shape = (N,)
        kfolds: int=10
            Integer number of folds to be split
        seed: int=None
            random_state for kfold shuffling
        name: str=None
            Optional str to be added to suptitle to distinguish this dataset
    Returns:
        splits_idx: np.array[int]
            An array containing integer index values for selecting the true label from y.
            shape = (N//kfolds, kfolds)

    """

    skf = StratifiedKFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.seed)

    splits_idx = pd.DataFrame(
        [sorted(test) for train, test in skf.split(range(len(y)), y)]
    ).T.convert_dtypes()

    splits_y = pd.DataFrame(np.array(y)[splits_idx.dropna().values.astype(int)])
    splits_y.T.index.name = "kfold"

    num_classes = len(set(y))
    bins = bins or num_classes // 6

    splits_y.stack().hist(
        by="kfold", alpha=0.4, bins=bins, figsize=(14, 14), sharex=True, sharey=True, color="b"
    )

    title = f"class distributions across k={cfg.kfolds} StratifiedFolds"
    if isinstance(name, str):
        title = f"{name} | {title}"

    plt.suptitle(title)
    #     plt.suptitle(f'class distributions across k={cfg.kfolds} StratifiedFolds')
    plt.tight_layout()

    return splits_idx


def filter_rare_classes_from_dataframe(
    data: pd.DataFrame, y_col: str = "family", threshold: int = 1, verbose: bool = True
) -> pd.DataFrame:
    """
    Low class-count dataframe filter function

    Filter dataframe `data` to only include classes with a minimum of `threshold` counts.
    Classes are defined by the column of `y_col`

    """
    class_counts = data.value_counts(y_col)
    class_counts = class_counts[class_counts >= threshold]
    include_classes = class_counts.index

    filtered_data = data[data[y_col].isin(include_classes)]

    if verbose:
        print(f"Num_classes: Previous={len(set(data[y_col]))}, Now={len(include_classes)}")
        print(f"Num_samples: Previous={data.shape[0]}, Now={filtered_data.shape[0]}")

    return filtered_data


##################################################


def create_symlink(src_path: str, target_path: str, keep_existing: bool = False):
    if os.path.exists(target_path):
        if keep_existing:
            return
        os.path.unlink(target_path)
    os.symlink(src_path, target_path)


#     os.symlink(x.absolute_path, x.target_path)


## creates symlinks for 1 dataset
def create_symlinks(
    data: pd.DataFrame,
    target_dir: str,
    y_col: str,
    keep_existing: bool = True,
    parallel: bool = True,
    skip_symlinks: bool = False,
):

    if os.path.isdir(target_dir) and not keep_existing:
        shutil.rmtree(target_dir)

    subdirs = sorted(set(data[y_col]))
    for subdir in subdirs:
        os.makedirs(Path(target_dir, subdir), exist_ok=True)

    data = data.assign(target_path=data.relative_path.apply(lambda x: str(Path(target_dir, x))))
    if skip_symlinks:
        return data

    print(f"Generating {len(subdirs)} subdirs in directory {target_dir}")
    print(f"Generating {data.shape[0]} symlinks in generated subdirs")
    if parallel:
        data.parallel_apply(
            lambda x: create_symlink(
                src_path=x.absolute_path, target_path=x.target_path, keep_existing=keep_existing
            ),
            axis=1,
        )
    else:
        data.apply(
            lambda x: create_symlink(
                src_path=x.absolute_path, target_path=x.target_path, keep_existing=keep_existing
            ),
            axis=1,
        )

    return data


######################################


## Filters classes and creates symlinks for 1 dataset
def filter_rare_classes_and_create_symlinks(
    data: pd.DataFrame,
    target_dir: str,
    y_col: str = "family",
    threshold: int = 10,
    skip_symlinks: bool = False,
) -> pd.DataFrame:

    filtered_catalog = filter_rare_classes_from_dataframe(
        data=data, y_col=y_col, threshold=threshold, verbose=True
    )
    print(f"Dataset: {cfg.dataset_name}, Target dir: {target_dir}")

    symlink_data_catalog = create_symlinks(
        data=filtered_catalog, target_dir=target_dir, y_col=y_col, skip_symlinks=skip_symlinks
    )

    is_link = symlink_data_catalog.target_path.parallel_apply(os.path.islink)
    print(
        "Expected Num_samples: ",
        symlink_data_catalog.shape[0],
        "Verified existing Num_samples: ",
        is_link.sum(),
    )
    print(f"Finished threshold={threshold}")
    print("=" * 25)
    return symlink_data_catalog


# python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multithresh_symlink_trees.py" --dry-run --resolution 512 1024 1536 2048 --dataset_name General_Fossil Florissant_Fossil --num_workers 8

# python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multithresh_symlink_trees.py" --task clean -r 512 --dataset_name Florissant_Fossil --dry-run


def cmdline_args(args=""):
    p = argparse.ArgumentParser(
        description="Produce symlink trees from source dataset, or clean them up."
    )
    p.add_argument(
        "-t",
        "--task",
        dest="task",
        type=str,
        choices=["create", "clean", "clean+create"],
        default="create",
        help=(
            "Specify whether to create or clean (unlink) symlink trees according to the query"
            " produced by the other cmdline args."
        ),
    )
    p.add_argument(
        "-data",
        "--dataset_name",
        dest="dataset_name",
        type=str,
        nargs="+",
        choices=["Extant_Leaves", "Florissant_Fossil", "General_Fossil", "all"],
        help=(
            "Which dataset names to produce multiple threshold versions of. Currently available:"
            " ['Extant_Leaves', 'Florissant_Fossil', 'General_Fossil']"
        ),
    )
    p.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        type=int,
        nargs="*",
        default=512,
        help="Resolution(s) to build symlinks from, images should be resized to (3, res, res).",
    )
    p.add_argument(
        "-d",
        "--root_dir",
        dest="root_dir",
        type=str,
        default="/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_1/images",
        help="""Destination image root dir. Script will expect source images to exist in class-wise subdirs in ".{dataset_name}/original/full/jpg". Then, for creating the target images it will create subdirs ".{dataset_name}/{resolution}/{threshold}/jpg" for user-input threshold value.""",
    )
    p.add_argument(
        "-a",
        "--run-all",
        dest="run_all",
        action="store_true",
        help=(
            "Overrides any values provided to --dataset_name. Flag for when user would like to run"
            " through all default threshold arguments on all datasets. Currently available:"
            " ['Extant_Leaves', 'Florissant_Fossil', 'General_Fossil']."
        ),
    )
    p.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        default=16,
        help="Number of parallel processes to be used by pandas to efficiently construct symlinks.",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "Flag for displaying the configurations indicated by args, then exiting prior to"
            " actually constructing anything on disk."
        ),
    )
    args = p.parse_args(args)

    #     if args.task == ""
    if args.run_all:
        args.resolution = [512, 1024, 1536, 2048]
        print("[RUNNING ALL THRESHOLDS]")
    if args.dataset_name == "all":
        args.dataset_name = ["Extant_Leaves", "Florissant_Fossil", "General_Fossil"]
        print("[RUNNING ALL DATASETS]")
    return args


######################################################

if __name__ == "__main__":

    args = cmdline_args(sys.argv[1:])
    image_root_dir = Path(args.root_dir)
    dataset_names = args.dataset_name
    resolutions = args.resolution
    #     if args.run_all:
    #         args.resolution = ["original", 512, 1024, 1536, 2048]
    #         args.dataset_name = ['Extant_Leaves', 'Florissant_Fossil', 'General_Fossil']
    subdirs = {
        "Extant_Leaves": "Extant_Leaves",
        "Florissant_Fossil": str(Path("Fossil", "Florissant_Fossil")),
        "General_Fossil": str(Path("Fossil", "General_Fossil")),
    }
    dataset_thresholds = {
        "Extant_Leaves": [3, 10, 20, 50, 100],
        "Florissant_Fossil": [3, 10, 20, 50],
        "General_Fossil": [3, 10, 20, 50],
    }
    y_col = "family"
    seed = 3546

    ## Create subdirs for each combination of (resolution, threshold, dataset)
    dataset_root_dirs = {}
    dataset_root_dirs_flat = {}

    for name in dataset_names:
        dataset_root_dirs[name] = {}  # str(image_root_dir / subdirs[name])
        for resolution in resolutions:
            dataset_root_dirs[name][resolution] = {}
            for threshold in dataset_thresholds[name]:
                dataset_root_dirs[name][resolution][threshold] = str(
                    image_root_dir / subdirs[name] / str(resolution) / str(threshold) / "jpg"
                )
                dataset_root_dirs_flat[
                    f"{name}_{resolution}_{y_col}_{threshold}"
                ] = dataset_root_dirs[name][resolution][threshold]

    if args.dry_run:
        print(
            "Dry Run exiting before any changes on disk. User cmd line args would otherwise"
            f" produce {len(dataset_root_dirs_flat)} different configurations."
        )
        pp(dataset_root_dirs_flat)
        exit(0)

    num_workers = args.num_workers
    pandarallel.initialize(nb_workers=num_workers, progress_bar=True)

    ## Define subdirs for each source dataset in its original resolution and "full" class listing (i.e. threshold=0)
    ## Load each of these source datasets using torchvision.datasets.ImageFolder
    full_root_dirs = {}
    source_datasets = {}
    for name in dataset_names:
        source_datasets[name] = {}
        full_root_dirs[name] = {}
        for resolution in resolutions:
            full_root_dirs[name][resolution] = str(
                image_root_dir / subdirs[name] / str(resolution) / "full" / "jpg"
            )
            #         full_root_dirs[name] = str(image_root_dir / subdirs[name] / "original" / "full" / "jpg")
            source_datasets[name][resolution] = torchvision.datasets.ImageFolder(
                full_root_dirs[name][resolution]
            )

    print(
        f"Producing {len(dataset_root_dirs_flat)} unique configurations for symlink trees, across"
        f" datasets: {dataset_names}"
    )

    class Config:
        pass

    #     resolutions = args.resolution #[512, 1024, 1536, 2048]
    i = 0
    skip_symlinks = False  # True
    symlink_data_catalogs = {}
    for dataset_name in dataset_names:
        symlink_data_catalogs[dataset_name] = {}
        for resolution in resolutions:
            symlink_data_catalogs[dataset_name][resolution] = {}
            for threshold in dataset_thresholds[dataset_name]:

                cfg = Config()
                cfg.dataset_name = dataset_name
                cfg.resolution = resolution
                cfg.threshold = threshold
                cfg.seed = seed
                cfg.y_col = y_col

                data = source_datasets[cfg.dataset_name][cfg.resolution]
                data_catalog = dataset2catalog(root_dir=None, dataset=data)
                target_dir = dataset_root_dirs[cfg.dataset_name][cfg.resolution][cfg.threshold]

                if "clean" in args.task:
                    if os.path.isdir(target_dir):
                        print(
                            f"[CLEANING] - [{time.ctime()}] - {i} - dataset: {dataset_name} -"
                            f" resolution: {resolution} - threshold: {threshold}"
                        )
                        print("\t\t - " + f"target_dir: {target_dir.rstrip('jpg')}")
                        shutil.rmtree(target_dir.rstrip("jpg"))
                        print(
                            f"[FINISHED] - [{time.ctime()}] - {i} - dataset: {dataset_name} -"
                            f" resolution: {resolution} - thresholds:"
                            f" {dataset_thresholds[dataset_name]}"
                        )

                if "create" in args.task:
                    print(
                        f"[CREATING] - [{time.ctime()}] - {i} - dataset: {dataset_name} -"
                        f" resolution: {resolution} - threshold: {threshold}"
                    )
                    symlink_data_catalogs[cfg.dataset_name][cfg.resolution][
                        cfg.threshold
                    ] = filter_rare_classes_and_create_symlinks(
                        data=data_catalog,
                        target_dir=target_dir,
                        y_col=cfg.y_col,
                        threshold=cfg.threshold,
                        skip_symlinks=skip_symlinks,
                    )
                i += 1
            print(
                f"[FINISHED] - [{time.ctime()}] - {i} - dataset: {dataset_name} - resolution:"
                f" {resolution} - thresholds: {dataset_thresholds[dataset_name]}"
            )

    mode = "w"
    if os.path.isfile(Path(image_root_dir, "Dataset summary.txt")):
        mode = "a"
    f = open(Path(image_root_dir, "Dataset summary.txt"), mode)
    with contextlib.redirect_stdout(f):
        print("=" * 60)
        print()
        print(f"Last task: {args.task}")
        print(f"Time: {time.ctime()}")
        print(f"root_dir: {image_root_dir}")
        for dataset_name, v in symlink_data_catalogs.items():
            print("=" * 30)
            print(f"Dataset: {dataset_name}")
            for resolution, v_i in v.items():
                print(f"Resolution: {resolution}")
                for threshold_i, v_ii in v_i.items():
                    print(f"Threshold: {threshold_i} -- {v_ii.shape[0]} Samples")
                    print(f"Path: {dataset_root_dirs[dataset_name][resolution][threshold_i]}")
                    if os.path.exists(
                        dataset_root_dirs[dataset_name][resolution][threshold_i]
                    ) and len(os.listdir(dataset_root_dirs[dataset_name][resolution][threshold_i])):
                        print("Status: Exists")
                    else:
                        print("Status: Cleaned")

    f.close()
