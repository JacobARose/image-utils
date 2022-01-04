"""

lightning_hydra_classifiers/data/utils/make_catalogs.py

Author: Jacob A Rose
Created: Wednesday July 28th, 2021

generate experiment directories containing csv datasets and yaml configs

Currently covers:
    - leavesdb v0_3
    - leavesdb v1_0

Work In Progress:
    - leavesdb v1_1

TODO:
    - Add to a make file in base directory
    - Add more flexible configuration


python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/make_catalogs.py" --all


python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/make_catalogs.py" --make_original


python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/make_catalogs.py" --make_extant


"""

import argparse
import collections
import os
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import *

import pandas as pd
import torch
import torchdata
import torchvision

# from lightning_hydra_classifiers.utils.common_utils import LabelEncoder
from IPython.display import display

# from lightning_hydra_classifiers.data.utils import catalog_registry
from lightning_hydra_classifiers.data.datasets.common import (
    ETL,
    CSVDataset,
    CSVDatasetConfig,
    ImageFileDataset,
    ImageFileDatasetConfig,
    PathSchema,
    SampleSchema,
)
from lightning_hydra_classifiers.utils.common_utils import (
    DataSplitter,
    LabelEncoder,
    trainval_split,
    trainvaltest_split,
)
from omegaconf import DictConfig
from PIL import Image

# from lightning_hydra_classifiers.data.common import PathSchema
# from lightning_hydra_classifiers.utils.dataset_management_utils import Extract as ExtractBase
# from lightning_hydra_classifiers.utils.etl_utils import ETL as ETLBase






#######################################################
#######################################################

def create_dataset_A_in_B(dataset_A,
                          dataset_B) -> pd.DataFrame:

    A_w_B = dataset_A.intersection(dataset_B)

    columns = [*[col for col in A_w_B.columns if col.endswith("_x")], *["catalog_number"]]
    A_in_B = A_w_B.reset_index()[columns].sort_values("catalog_number")

    print(f"A_in_B.columns: {A_in_B.columns}")
    A_in_B = A_in_B.rename(columns = {col: col.split("_x")[0] for col in A_in_B.columns})

    return A_in_B










#######################################################
#######################################################



def export_dataset_catalog_configuration(output_dir: str = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0",
                                         base_dataset_name = "Extant_Leaves",
                                         threshold = 100,
                                         resolution = 512,
                                         version: str = "v1_0",
                                         path_schema: str = "{family}_{genus}_{species}_{collection}_{catalog_number}"):

    image_file_config = ImageFileDatasetConfig(base_dataset_name = base_dataset_name,
                                               class_type = "family",
                                               threshold = threshold,
                                               resolution = resolution,
                                               version=version,
                                               path_schema = path_schema)

    out_dir = os.path.join(output_dir, image_file_config.full_name)
    os.makedirs(out_dir, exist_ok=True)

    csv_out_path = os.path.join(out_dir, f"{image_file_config.full_name}-full_dataset.csv")
    image_file_config_out_path = os.path.join(out_dir,"ImageFileDataset-config.yaml")
    csv_config_out_path = os.path.join(out_dir,"CSVDataset-config.yaml")

    dataset = ImageFileDataset.from_config(image_file_config, subset_keys=['all'])
    ETL.df2csv(dataset.samples_df,
                   path = csv_out_path)
    image_file_config.save(image_file_config_out_path)

    csv_config = CSVDatasetConfig(full_name = image_file_config.full_name,
                                  data_path = csv_out_path,
                                  subset_key = "all")

    csv_config.save(csv_config_out_path)

    print(f"[FINISHED] DATASET FULL NAME: {csv_config.full_name}")
    print(f"Newly created dataset assets located at:  {out_dir}")

    return dataset, image_file_config, csv_config

##############################################


def export_composite_dataset_catalog_configuration(output_dir: str = ".",
                                                   csv_cfg_path_A: str=None,
                                                   csv_cfg_path_B: str=None,
                                                   composition: str="-") -> Tuple[CSVDataset, CSVDatasetConfig]:


    csv_config_A = CSVDatasetConfig.load(path = csv_cfg_path_A)
    csv_config_B = CSVDatasetConfig.load(path = csv_cfg_path_B)

    dataset_A = CSVDataset.from_config(csv_config_A)
    dataset_B = CSVDataset.from_config(csv_config_B)
    print(f"num_samples A: {len(dataset_A)}")
    print(f"num_samples B: {len(dataset_B)}")

    print(f"producing composition: {composition}")
    if composition == '-':
        dataset_A_composed_B = dataset_A - dataset_B
        full_name = f"{csv_config_A.full_name}_minus_{csv_config_B.full_name}"
        out_dir = os.path.join(output_dir, full_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"num_samples A-B: {len(dataset_A_composed_B)}")


    if composition == 'intersection':
        dataset_A_composed_B = dataset_A.intersection(dataset_B)
        full_name = f"{csv_config_A.full_name}_w_{csv_config_B.full_name}"
        print(f"num_samples A_w_B: {len(dataset_A_composed_B)}")

        out_dir = os.path.join(output_dir, full_name)
        os.makedirs(out_dir, exist_ok=True)


    ########################################
    A_in_B = create_dataset_A_in_B(dataset_A,
                                   dataset_B)
    csv_dataset_pathname = f"{csv_config_A.full_name}_in_{csv_config_B.full_name}"
    csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
    ETL.df2csv(A_in_B,
                   path = csv_dataset_out_path)
    A_in_B_config = CSVDatasetConfig(full_name = csv_dataset_pathname,
                                     data_path = csv_dataset_out_path,
                                     subset_key = "all")
    csv_dataset_config_out_path = os.path.join(out_dir,f"A_in_B-CSVDataset-config.yaml")
    A_in_B_config.save(csv_dataset_config_out_path)

    #########################################
    B_in_A = create_dataset_A_in_B(dataset_B,
                                   dataset_A)

    csv_dataset_pathname = f"{csv_config_B.full_name}_in_{csv_config_A.full_name}"
    csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
    ETL.df2csv(B_in_A,
                   path = csv_dataset_out_path)
    B_in_A_config = CSVDatasetConfig(full_name = csv_dataset_pathname,
                                     data_path = csv_dataset_out_path,
                                     subset_key = "all")

    csv_dataset_config_out_path = os.path.join(out_dir,f"B_in_A-CSVDataset-config.yaml")
    B_in_A_config.save(csv_dataset_config_out_path)


    #########################################

    inputs_dir = os.path.join(out_dir, "inputs")
    os.makedirs(os.path.join(inputs_dir, "A"), exist_ok=True)
    os.makedirs(os.path.join(inputs_dir, "B"), exist_ok=True)
    shutil.copyfile(csv_cfg_path_A, os.path.join(inputs_dir, "A", Path(csv_cfg_path_A).name))
    shutil.copyfile(csv_cfg_path_B, os.path.join(inputs_dir, "B", Path(csv_cfg_path_B).name))

    csv_dataset_pathname = f"{full_name}-full_dataset"
    csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
    ETL.df2csv(dataset_A_composed_B,
                   path = csv_dataset_out_path)
    ####################
    ####################
    csv_dataset_config_out_path = os.path.join(out_dir,f"CSVDataset-config.yaml")
    A_composed_B_config = CSVDatasetConfig(full_name = full_name,
                                        data_path = csv_dataset_out_path,
                                        subset_key = "all")
    A_composed_B_config.save(csv_dataset_config_out_path)

    print(f"[FINISHED] DATASET: {full_name}")
    print(f"Newly created dataset assets located at:  {out_dir}")

    if composition == '-':
        dataset_A_composed_B = CSVDataset.from_config(A_composed_B_config)
        return dataset_A_composed_B, A_composed_B_config

    if composition == 'intersection':
        return (A_in_B, B_in_A), A_composed_B_config









############################################
############################################
############################################
############################################


def make_all_original(args):
#     if "output_dir" not in args:
#         args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"

    output_dir = args.output_dir
    version = args.version

    base_dataset_name = "" #original"
#                           "Extant_Leaves",
#                           "Florissant_Fossil",
#                           "General_Fossil"]

    resolution = "original"
    path_schema = "{family}_{genus}_{species}_{collection}_{catalog_number}"

    print(f'Beginning make_all_original()')
#     for base_dataset_name in base_dataset_names:
    export_dataset_catalog_configuration(output_dir=output_dir,
                                         base_dataset_name = base_dataset_name,
                                         threshold = 0,
                                         resolution = resolution,
                                         version = version,
                                         path_schema = path_schema)

    print(f'FINISHED ALL IN original datasets')
    print('=='*15)






def make_fossil(args):

    output_dir = args.output_dir
    version = args.version

    base_dataset_name = "Fossil"
    thresholds = [None,3]
    resolutions = [512,1024]
    path_schema = "{family}_{genus}_{species}_{collection}_{catalog_number}"

    print(f'Beginning make_fossil() for {len(resolutions)}x resolutions and {len(thresholds)}x thresholds')

    for threshold in thresholds:
        for resolution in resolutions:
            export_dataset_catalog_configuration(output_dir=output_dir,
                                                 base_dataset_name = base_dataset_name,
                                                 threshold = threshold,
                                                 resolution = resolution,
                                                 version = version,
                                                 path_schema = path_schema)

    print(f'FINISHED ALL IN Fossil')
    print('=='*15)


def make_extant(args):
#     if "output_dir" not in args:
#         args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"

    output_dir = args.output_dir
    version = args.version

    base_dataset_name = "Extant_Leaves"
    thresholds = [None, 3, 10,100]
    resolutions = [512,1024]
    path_schema = "{family}_{genus}_{species}_{collection}_{catalog_number}"

    print(f'Beginning make_extant() for {len(resolutions)}x resolutions and {len(thresholds)}x thresholds')
    for threshold in thresholds:
        for resolution in resolutions:
            export_dataset_catalog_configuration(output_dir=output_dir,
                                                 base_dataset_name = base_dataset_name,
                                                 threshold = threshold,
                                                 resolution = resolution,
                                                 version = version,
                                                 path_schema = path_schema)

    print(f'FINISHED ALL IN Extant_Leaves')
    print('=='*15)


def make_pnas(args):
#     if "output_dir" not in args:
#         args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"

    output_dir = args.output_dir
    version = args.version

    base_dataset_name = "PNAS"
    thresholds = [100]
    resolutions = [512,1024]
    path_schema = "{family}_{genus}_{species}_{catalog_number}"

    print(f'Beginning make_pnas() for {len(resolutions)}x resolutions and {len(thresholds)}x thresholds')
    for threshold in thresholds:
        for resolution in resolutions:
            export_dataset_catalog_configuration(output_dir=output_dir,
                                                 base_dataset_name = base_dataset_name,
                                                 threshold = threshold,
                                                 resolution = resolution,
                                                 version = version,
                                                 path_schema = path_schema)

    print(f'FINISHED ALL IN PNAS')
    print('=='*15)


def make_extant_minus_pnas(args):
#     if "output_dir" not in args:
#         args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"

    output_dir = args.output_dir
    version = args.version

    base_names = {"A": "Extant_Leaves",
                  "B": "PNAS"}
    thresholds = [{"A":100,
                   "B":100},
                 {"A":10,
                   "B":100}]
    resolutions = [512, 1024]
    class_type = "family"


    for threshold in thresholds:
        for resolution in resolutions:
            dataset_full_names = {"A":"_".join([base_names["A"], class_type, str(threshold["A"]), str(resolution)]),
                                  "B":"_".join([base_names["B"], class_type, str(threshold["B"]), str(resolution)])}

            csv_cfg_path_A = os.path.join(output_dir, dataset_full_names["A"], "CSVDataset-config.yaml")
            csv_cfg_path_B = os.path.join(output_dir, dataset_full_names["B"], "CSVDataset-config.yaml")
            dataset, cfg = export_composite_dataset_catalog_configuration(output_dir=output_dir,
                                                                          csv_cfg_path_A=csv_cfg_path_A,
                                                                          csv_cfg_path_B=csv_cfg_path_B,
                                                                          composition="-")


    print(f'FINISHED ALL IN Extant-PNAS')
    print('=='*15)


def make_pnas_minus_extant(args):

#     if "output_dir" not in args:
#         args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"
    output_dir = args.output_dir


    base_names = {"A": "PNAS",
                  "B": "Extant_Leaves"}
    thresholds = [{"A":100,
                   "B":100},
                 {"A":100,
                  "B":10}]
    resolutions = [512, 1024]
    class_type = "family"


    for threshold in thresholds:
        for resolution in resolutions:
            dataset_full_names = {"A":"_".join([base_names["A"], class_type, str(threshold["A"]), str(resolution)]),
                                  "B":"_".join([base_names["B"], class_type, str(threshold["B"]), str(resolution)])}

            csv_cfg_path_A = os.path.join(output_dir, dataset_full_names["A"], "CSVDataset-config.yaml")
            csv_cfg_path_B = os.path.join(output_dir, dataset_full_names["B"], "CSVDataset-config.yaml")
            dataset, cfg = export_composite_dataset_catalog_configuration(output_dir=output_dir,
                                                                          csv_cfg_path_A=csv_cfg_path_A,
                                                                          csv_cfg_path_B=csv_cfg_path_B,
                                                                          composition="-")


    print(f'FINISHED ALL IN Extant-PNAS')
    print('=='*15)


def make_extant_w_pnas(args):

#     if "output_dir" not in args:
#         args.output_dir = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"
    output_dir = args.output_dir


    base_names = {"A": "Extant_Leaves",
                  "B": "PNAS"}
    thresholds = [{"A":100,
                   "B":100},
                 {"A":10,
                   "B":100}]
    resolutions = [512, 1024]
    class_type = "family"


    for threshold in thresholds:
        for resolution in resolutions:
            dataset_full_names = {"A":"_".join([base_names["A"], class_type, str(threshold["A"]), str(resolution)]),
                                  "B":"_".join([base_names["B"], class_type, str(threshold["B"]), str(resolution)])}

            csv_cfg_path_A = os.path.join(output_dir, dataset_full_names["A"], "CSVDataset-config.yaml")
            csv_cfg_path_B = os.path.join(output_dir, dataset_full_names["B"], "CSVDataset-config.yaml")
            dataset, cfg = export_composite_dataset_catalog_configuration(output_dir=output_dir,
                                                                          csv_cfg_path_A=csv_cfg_path_A,
                                                                          csv_cfg_path_B=csv_cfg_path_B,
                                                                          composition="intersection")


    print(f'FINISHED ALL IN Extant_w_PNAS')
    print('=='*15)










CSV_CATALOG_DIR_V0_3 = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v0_3"
CSV_CATALOG_DIR_V1_0 = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0"
EXPERIMENTAL_DATASETS_DIR = "/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/experimental_datasets"


def cmdline_args():
    p = argparse.ArgumentParser(description="Export a series of dataset artifacts (containing csv catalog, yml config, json labels) for each dataset, provided that the corresponding images are pointed to by one of the file paths hard-coded in catalog_registry.py.")
    p.add_argument("-o", "--output_dir", dest="output_dir", type=str,
                   default="/media/data_cifs/projects/prj_fossils/users/jacob/experiments/July2021-Nov2021/csv_datasets/leavesdb-v1_0",
                   help="Output root directory. Each unique dataset will be allotted its own subdirectory within this root dir.")
    p.add_argument("-a", "--all", dest="make_all", action="store_true",
                   help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
    p.add_argument("-v", "--version", dest="version", type=str, default='v1_0',
                   help="Available dataset versions: [v0_3, v1_0].")
    p.add_argument("--fossil", dest="make_fossil", action="store_true",
                   help="If user provides this flag, produce all configurations of the combined Florissant + General Fossil collections.")
    p.add_argument("--extant", dest="make_extant", action="store_true",
                   help="If user provides this flag, produce all currently in-production resolutions+thresholds in the dataset: Extant_Leaves's most recent version (currently == 'v1_0').")
    p.add_argument("--pnas", dest="make_pnas", action="store_true",
                   help=f"If user provides this flag, produce all currently in-production resolutions+thresholds in the dataset: PNAS's most recent version (currently == 'v1_0').")
#                    help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
    p.add_argument("--extant-pnas", dest="make_extant_minus_pnas", action="store_true",
                   help="If user provides this flag, produce all currently in-production resolutions+thresholds in the dataset: Extant-PNAS's most recent version (currently == 'v1_0').")
#                    help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
    p.add_argument("--pnas-extant", dest="make_pnas_minus_extant", action="store_true",
                   help="If user provides this flag, produce all currently in-production resolutions+thresholds in the dataset: PNAS-Extant's most recent version (currently == 'v1_0').")
#                    help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
    p.add_argument("--extant-w-pnas", dest="make_extant_w_pnas", action="store_true",
                   help="If user provides this flag, produce all currently in-production resolutions+thresholds in the dataset: Extant_w_PNAS's most recent version (currently == 'v1_0').")
#                    help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
    p.add_argument("--original", dest="make_all_original", action="store_true",
                   help=f"If user provides this flag, produce all currently in-production resolutions+thresholds in the original datasets' () most recent version (currently == 'v1_0'). Note that this excludes PNAS.")
#                    help="If user provides this flag, produce all currently in-production datasets in the most recent version (currently == 'v1_0').")
    return p.parse_args()

make_all_original

if __name__ == "__main__":
#     import sys
#     args = sys.argv

    args = cmdline_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.make_fossil or args.make_all:
        make_fossil(args)
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



































#######################################################
#######################################################
#######################################################
#######################################################


# totensor: Callable = torchvision.transforms.ToTensor()
# def toPIL(img: torch.Tensor, mode="RGB") -> Callable:
#     return torchvision.transforms.ToPILImage(mode)(img)

# class Extract(ExtractBase):


#     @classmethod
#     def export_dataset_state(cls,
#                              output_dir: Union[str, Path],
#                              df: pd.DataFrame=None,
#                              config: DictConfig=None,
#                              encoder: LabelEncoder=None,
#                              dataset_name: Optional[str]="dataset"
#                              ) -> None:

#         paths = {ftype: str(output_dir / str(dataset_name + ext)) for ftype, ext in cls.data_file_ext_maps.items()}

#         output_dir = Path(output_dir)
#         if isinstance(df, pd.DataFrame):
#             cls.df2csv(df = df,
#                        path = paths["df"])
#             if config:
#                 config.data_path = paths["df"]
#         if isinstance(encoder, LabelEncoder):
#             cls.labels2json(encoder=encoder,
#                             path = paths["encoder"])
#             if config:
#                 config.label_encoder_path = paths["encoder"]
#         if isinstance(config, CSVDatasetConfig):
#             config.save(path = paths["config"])
# #             cls.config2yaml(config=config,
# #                             path = paths["config"])


#     @classmethod
#     def import_dataset_state(cls,
#                              data_dir: Optional[Union[str, Path]]=None,
#                              config_path: Optional[Union[Path, str]]=None,
#                             ) -> Tuple["CSVDataset", "CSVDatasetConfig"]:
#         if (not os.path.exists(str(data_dir))) and (not os.path.exists(config_path)):
#             raise ValueError("Either data_dir or config_path must be existing paths")

#         if os.path.isdir(str(data_dir)):
#             data_dir = Path(data_dir)
#         paths = {}

#         # import config yaml file
#         if os.path.isfile(str(config_path)):
#             paths['config'] = config_path
# #             config = cls.config_from_yaml(path = paths["config"])
#             config = CSVDatasetConfig.load(path = paths["config"])
#             if hasattr(config, "data_path"):
#                 paths["df"] = str(config.data_path)
#             if hasattr(config, "label_encoder_path"):
#                 paths["encoder"] = str(config.label_encoder_path)
#             data_dir = Path(os.path.dirname(config_path))

#         for ftype, ext in cls.data_file_ext_maps.items():
#             if ftype not in paths:
#                 paths[ftype] = str(list(data_dir.glob("*" + ext))[0])


#         config.data_path = str(paths["df"])
#         config.label_encoder_path = str(paths["encoder"])
#         if os.path.isfile(paths["encoder"]):
#             # import label encodings json file if it exists
#             label_encoder = cls.labels_from_json(path = paths["encoder"])

#         # import dataset samples from a csv file as a CustomDataset/CSVDataset object
#         dataset = CSVDataset.from_config(config,
#                                          eager_encode_targets=True)
#         dataset.setup(samples_df=dataset.samples_df,
#                       label_encoder=label_encoder,
#                       fit_targets=True)

#         return dataset, config


# @dataclass
# class BaseDatasetConfig:

#     def save(self,
#              path: Union[str, Path]) -> None:

#         cfg = asdict(self)
# #         cfg = DictConfig({k: getattr(self,k) for k in self.keys()})
#         Extract.config2yaml(cfg, path)

#     @classmethod
#     def load(cls,
#              path: Union[str, Path]) -> "DatasetConfig":

#         cfg = Extract.config_from_yaml(path)

# #         keys = cls.__dataclass_fields__.keys()
#         cfg = cls(**{k: cfg[k] for k in cls.keys()})
#         return cfg

#     @classmethod
#     def keys(cls):
#         return cls.__dataclass_fields__.keys()

#     def __repr__(self):
#         out = f"{type(self)}" + "\n"
#         out += "\n".join([f"{k}: {getattr(self, k)}" for k in self.keys()])
# #         out += f"\nroot_dir: {self.root_dir}"
# #         out += "\nsubset_dirs: \n\t" + '\n\t'.join(self.subset_dirs)
#         return out


# @dataclass
# class DatasetConfig(BaseDatasetConfig):
#     base_dataset_name: str = "Extant_Leaves"
#     class_type: str = "family"
#     threshold: Optional[int] = 10
#     resolution: int = 512
#     version: str = "v1_0"
#     path_schema: str = "{family}_{genus}_{species}_{collection}_{catalog_number}"

#     @property
#     def available_versions(self) -> List[str]:
#         return list(catalog_registry.available_datasets().versions.keys())

#     @property
#     def full_name(self) -> str:
#         name  = self.base_dataset_name
#         if self.threshold:
#             name += f"_{self.class_type}_{self.threshold}"
#         name += f"_{self.resolution}"
#         return name


# class ImageFileDatasetConfig(DatasetConfig):

#     @property
#     def root_dir(self):
#         return catalog_registry.available_datasets.get(self.full_name, version=self.version)

#     def is_valid_subset(self, subset: str):
#         for s in ("train", "val", "test"):
#             if s in subset:
#                 return True
#         return False

#     @property
#     def subsets(self):
#         if isinstance(self.root_dir, list):
#             return []
#         return [s for s in os.listdir(self.root_dir) if self.is_valid_subset(s)]

#     @property
#     def subset_dirs(self):
#         return [os.path.join(self.root_dir, subset) for subset in self.subsets]
# #         subsets = self.subsets
# #         return [os.path.join(self.root_dir, subset) for subset in subsets if self.is_valid_subset(subset)]

#     def locate_files(self) -> Dict[str, List[Path]]:
#         return Extract.locate_files(self.root_dir)


#     @cached_property
#     def num_samples(self):
# #         subset_dirs = {Path(subset_dir).stem: Path(subset_dir) for subset_dir in self.subset_dirs}
#         files = {subset: f for subset, f in self.locate_files().items() if self.is_valid_subset(subset)}
#         return {subset: len(list(f)) for subset, f in files.items()}
# #         return {subset: len(list(subset_dirs[subset].rglob("*/*.jpg"))) for subset in subset_dirs.keys()}


#     def __repr__(self):
#         out = super().__repr__()
#         out += f"\nroot_dir: {self.root_dir}"
#         out += "\nsubsets: "
#         for i, subset in enumerate(self.subsets):
#             out += '\n\t' + f"{subset}:"
#             out += '\n\t\t' + f"subdir: {self.subset_dirs[i]}"
#             out += '\n\t\t' + f"subset_num_samples: {self.num_samples[subset]}"
#         return out


# @dataclass
# class CSVDatasetConfig(BaseDatasetConfig):

#     full_name: str = None
#     data_path: str = None
#     label_encoder_path: Optional[str] = None
#     subset_key: str = "all"

#     def update(self, **kwargs):
#         if "subset_key" in kwargs:
#             self.subset_key = kwargs["subset_key"]
#         if "num_samples" in kwargs:
#             self.num_samples = {self.subset_key: kwargs["num_samples"]}

#     @cached_property
#     def num_samples(self):
#         return {self.subset_key: len(self.locate_files())}

#     def __repr__(self):
#         out = super().__repr__()
#         out += '\n' + f"num_samples: {self.num_samples[self.subset_key]}"
#         return out

#     def locate_files(self) -> pd.DataFrame: #Dict[str, List[Path]]:
#         return Extract.df_from_csv(self.data_path)

#     def load_label_encoder(self) -> Union[None, LabelEncoder]:
#         if os.path.exists(str(self.label_encoder_path)):
#             return Extract.labels_from_json(str(self.label_encoder_path))
#         return

#     @classmethod
#     def export_dataset_state(cls,
#                              output_dir: Union[str, Path],
#                              df: pd.DataFrame=None,
#                              config: DictConfig=None,
#                              encoder: LabelEncoder=None,
#                              dataset_name: Optional[str]="dataset"
#                              ) -> None:
#         Extract.export_dataset_state(output_dir=output_dir,
#                                      df=df,
#                                      config=config,
#                                      encoder=encoder,
#                                      dataset_name=dataset_name)


#     @classmethod
#     def import_dataset_state(cls,
#                              data_dir: Optional[Union[str, Path]]=None,
#                              config_path: Optional[Union[Path, str]]=None,
#                             ) -> Tuple["CSVDataset", "CSVDatasetConfig"]:

#         return Extract.import_dataset_state(data_dir=data_dir,
#                                             config_path=config_path)





# ####################################
# ####################################



# @dataclass
# class SampleSchema:
#     path : Union[str, Path] = None
#     family : str = None
#     genus : str = None
#     species : str = None
#     collection : str = None
#     catalog_number : str = None

#     @classmethod
#     def keys(cls):
#         return list(cls.__dataclass_fields__.keys())

#     def __getitem__(self, index: int):
#         return getattr(self, self.keys()[index])


# class CustomDataset(torchdata.datasets.Files): # (CommonDataset):

#     def __init__(self,
#                  files: List[Path]=None,
#                  samples_df: pd.DataFrame=None,
#                  path_schema: Path = "{family}_{genus}_{species}_{collection}_{catalog_number}",
#                  return_signature: List[str] = ["image","target"], #,"path"],
#                  eager_encode_targets: bool = False,
#                  config: Optional[BaseDatasetConfig]=None,
#                  transform=None):
#         files = files or []
#         super().__init__(files=files)
# #         self.samples_df = samples_df
#         self.path_schema = PathSchema(path_schema)
#         self._return_signature = collections.namedtuple("return_signature", return_signature)

#         self.x_col = "path"
#         self.y_col = "family"
#         self.id_col = "catalog_number"
#         self.config = config or {}
#         self.transform = transform
#         self.eager_encode_targets = eager_encode_targets
#         self.setup(samples_df=samples_df)


#     def fetch_item(self, index: int) -> Tuple[str]:
#         sample = self.parse_sample(index)
#         image = Image.open(sample.path)
#         return self.return_signature(image=image,
#                                      target=getattr(sample, self.y_col),
#                                      path=getattr(sample, self.x_col))

#     def return_signature(self, **kwargs):
#         return self._return_signature(**{key: kwargs[key] for key in self._return_signature._fields if key in kwargs})

#     def __getitem__(self, index: int):

#         item = self.fetch_item(index)
#         image, target, path = item.image, item.target, str(getattr(item, "path", None))

#         target = self.label_encoder.class2idx[target]

#         if self.transform is not None:
# #             image = totensor(image)
#             image = self.transform(image)
# #         if self.target_transform is not None:
# #             target = self.target_transform(target)

#         if path:
#             return tuple(self.return_signature(image=image, target=target, path=path))
#         return tuple(self.return_signature(image=image, target=target))


#     def setup(self,
#               samples_df: pd.DataFrame=None,
#               label_encoder: LabelEncoder=None,
#               fit_targets: bool=True):
#         """
#         Running setup() should result in the Dataset having assigned values for:
#             self.samples
#             self.targets
#             self.samples_df
#             self.label_encoder

#         """
#         if samples_df is not None:
#             self.samples_df = samples_df.convert_dtypes()
#         self.samples = [self.parse_sample(idx) for idx in range((len(self)))]
#         self.targets = [sample[1] for sample in self.samples]
#         self.samples_df = pd.DataFrame(self.samples).convert_dtypes()

#         self.label_encoder = label_encoder or LabelEncoder()
#         if fit_targets:
#             self.label_encoder.fit(self.targets)

#         if self.eager_encode_targets:
#             self.targets = self.label_encoder.encode(self.targets).tolist()

#     @classmethod
#     def from_config(cls, config: DatasetConfig, subset_keys: List[str]=None) -> "CustomDataset":
#         pass

#     def parse_sample(self, index: int):
#         pass

#     @property
#     def classes(self):
#         return self.label_encoder.classes

#     def __repr__(self):
#         disp = f"""<{str(type(self)).strip("'>").split('.')[1]}>:"""
#         disp += '\n\t' + self.config.__repr__().replace('\n','\n\t')
# #         if len(self):
# #             disp += "\n\t" + f"num_samples: {len(self)}"
#         return disp


#     @classmethod
#     def get_files_from_samples(cls,
#                                samples: Union[pd.DataFrame, List],
#                                x_col: Optional[str]="path"):
#         if isinstance(samples, pd.DataFrame):
#             if x_col in samples.columns:
#                 files = list(samples[x_col].values)
#             else:
#                 files = list(samples.iloc[:,0].values)
#         elif isinstance(samples, list):
#             files = [s[0] for s in self.samples]

#         return files

#     def intersection(self, other, suffixes=("_x","_y")):
#         samples_df = self.samples_df
#         other_df = other.samples_df

#         intersection = samples_df.merge(other_df, how='inner', on=self.id_col, suffixes=suffixes)
#         return intersection

#     def __add__(self, other):

#         intersection = self.intersection(other)[self.id_col].tolist()
#         samples_df = self.samples_df

#         left_union = samples_df[samples_df[self.id_col].apply(lambda x: x in intersection)]

#         return left_union

#     def __sub__(self, other):

#         intersection = self.intersection(other)[self.id_col].tolist()
#         samples_df = self.samples_df

#         remainder = samples_df[samples_df[self.id_col].apply(lambda x: x not in intersection)]

#         return remainder

#     def filter(self, indices, subset_key: Optional[str]="all"):
#         out = type(self)(samples_df = self.samples_df.iloc[indices,:],
#                          config = deepcopy(self.config))
#         out.config.update(subset_key=subset_key,
#                           num_samples=len(out))
#         return out

#     def get_unsupervised(self):
#         return UnsupervisedDatasetWrapper(self)



# class UnsupervisedDatasetWrapper(torchdata.datasets.Files):#torchvision.datasets.ImageFolder):

#     def __init__(self, dataset):
#         super().__init__(files=dataset.files)
#         self.dataset = dataset
# #         super().__init__(samples_df=dataset.samples_df,
# #                            path_schema=dataset.path_schema)



#     def __getitem__(self, index):
#         return self.dataset[index][0]

#     def __len__(self):
#         return len(self.dataset)

#     def __repr__(self):
#         out = "<UnsupervisedDatasetWrapper>\n"
#         out += self.dataset.__repr__()
#         return out

# #     def parse_sample(self, index: int):
# #         path = self.files[index]





# class ImageFileDataset(CustomDataset):

#     @classmethod
#     def from_config(cls, config: DatasetConfig, subset_keys: List[str]=None) -> "CustomDataset":
#         files = config.locate_files()
#         if isinstance(subset_keys, list):
#             files = {k: files[k] for k in subset_keys}
#         if len(files.keys())==1:
#             files = files[subset_keys[0]]
#         new = cls(files=files,
#                   path_schema=config.path_schema)
#         new.config = config
#         return new

#     def parse_sample(self, index: int):
#         path = self.files[index]
#         family, genus, species, collection, catalog_number = self.path_schema.parse(path)

#         return SampleSchema(path=path,
#                              family=family,
#                              genus=genus,
#                              species=species,
#                              collection=collection,
#                              catalog_number=catalog_number)




# class CSVDataset(CustomDataset):

#     @classmethod
#     def from_config(cls,
#                     config: DatasetConfig,
#                     subset_keys: List[str]=None,
#                     eager_encode_targets: bool=False) -> Union[Dict[str, "CSVDataset"], "CSVDataset"]:

#         files_df = config.locate_files()
#         if subset_keys is None:
#             subset_keys = ['all']
#         if isinstance(subset_keys, list) and isinstance(files_df, dict):
#             files_df = {k: files_df[k] for k in subset_keys}
#             new = {k: cls(samples_df=files_df[k],  \
#                           eager_encode_targets=eager_encode_targets) for k in subset_keys}
#             for k in subset_keys:
#                 new[k].config = deepcopy(config)
#                 new[k].config.subset_key = k

#         if len(subset_keys)==1:
#             if isinstance(files_df, dict):
#                 files_df = files_df[subset_keys[0]]
#             new = cls(samples_df=files_df,
#                       eager_encode_targets=eager_encode_targets)
#             new.config = config
#             new.config.subset_key = subset_keys[0]
#         return new

#     def setup(self,
#               samples_df: pd.DataFrame=None,
#               label_encoder: LabelEncoder=None,
#               fit_targets: bool=True):

#         if samples_df is not None:
#             self.samples_df = samples_df.convert_dtypes()
#         self.files = self.samples_df[self.x_col].apply(lambda x: Path(x)).tolist()
#         super().setup(samples_df=self.samples_df,
#                       label_encoder=label_encoder,
#                       fit_targets=fit_targets)
# #         self.samples = [self.parse_sample(idx) for idx in range((len(self)))]
# #         self.targets = [sample[1] for sample in self.samples]
# #         self.samples_df = pd.DataFrame(self.samples).convert_dtypes()

# #         self.label_encoder = label_encoder or LabelEncoder()
# #         if fit_targets:
# #             self.label_encoder.fit(self.targets)

# #         if self.eager_encode_targets:
# #             self.targets = self.label_encoder.encode(self.targets).tolist()


#     def parse_sample(self, index: int):

#         row = self.samples_df.iloc[index,:].tolist()
#         path, family, genus, species, collection, catalog_number = row
#         return SampleSchema(path=path,
#                              family=family,
#                              genus=genus,
#                              species=species,
#                              collection=collection,
#                              catalog_number=catalog_number)













#######################################################
#######################################################
#######################################################
#######################################################









#         #########################################

#         A_in_B = create_dataset_A_in_B(dataset_A,
#                                        dataset_B)

#         csv_dataset_pathname = f"{csv_config_A.full_name}_in_{csv_config_B.full_name}"
#         csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
#         csv_dataset_config_out_path = os.path.join(out_dir,f"A_in_B-CSVDataset-config.yaml")
# #         print(f"A_in_B.columns: {A_in_B.columns}")
#         Extract.df2csv(A_in_B,
#                        path = csv_dataset_out_path)
#         A_in_B_config = CSVDatasetConfig(full_name = csv_dataset_pathname,
#                                          data_path = csv_dataset_out_path,
#                                          subset_key = "all")
#         A_in_B_config.save(csv_dataset_config_out_path)

#         #########################################
#         B_in_A = create_dataset_A_in_B(dataset_B,
#                                        dataset_A)

#         csv_dataset_pathname = f"{csv_config_B.full_name}_in_{csv_config_A.full_name}"
#         csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
#         Extract.df2csv(B_in_A,
#                        path = csv_dataset_out_path)
#         B_in_A_config = CSVDatasetConfig(full_name = csv_dataset_pathname,
#                                          data_path = csv_dataset_out_path,
#                                          subset_key = "all")

#         csv_dataset_config_out_path = os.path.join(out_dir,f"B_in_A-CSVDataset-config.yaml")
#         B_in_A_config.save(csv_dataset_config_out_path)




##############################################

# def export_composite_dataset_catalog_configuration(output_dir: str = ".",
#                                                    csv_cfg_path_A: str=None,
#                                                    csv_cfg_path_B: str=None,
#                                                    composition: str="-") -> Tuple[CSVDataset, CSVDatasetConfig]:


#     csv_config_A = CSVDatasetConfig.load(path = csv_cfg_path_A)
#     csv_config_B = CSVDatasetConfig.load(path = csv_cfg_path_B)

#     dataset_A = CSVDataset.from_config(csv_config_A)
#     dataset_B = CSVDataset.from_config(csv_config_B)
#     print(f"num_samples A: {len(dataset_A)}")
#     print(f"num_samples B: {len(dataset_B)}")

#     print(f"producing composition: {composition}")
#     if composition == '-':
#         dataset_A_composed_B = dataset_A - dataset_B
#         full_name = f"{csv_config_A.full_name}_minus_{csv_config_B.full_name}"
#         print(f"num_samples A-B: {len(dataset_A_composed_B)}")

#     if composition == 'intersection':
#         dataset_A_composed_B = dataset_A.intersection(dataset_B)
#         full_name = f"{csv_config_A.full_name}_w_{csv_config_B.full_name}"
#         print(f"num_samples A_w_B: {len(dataset_A_composed_B)}")

#         out_dir = os.path.join(output_dir, full_name)
#         os.makedirs(out_dir, exist_ok=True)

#         csv_dataset_pathname = f"{csv_config_A.full_name}_in_{csv_config_B.full_name}"
#         csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
#         csv_dataset_config_out_path = os.path.join(out_dir,f"A_in_B-CSVDataset-config.yaml")
#         columns = [*[col for col in dataset_A_composed_B.columns if col.endswith("_x")], *["catalog_number"]]
#         extant_in_pnas = dataset_A_composed_B.reset_index()[columns].sort_values("catalog_number")
# #         extant_in_pnas = extant_in_pnas.loc[:,[*list(SampleSchema.keys())]]

#         print(f"extant_in_pnas.columns: {extant_in_pnas.columns}")

#         extant_in_pnas = extant_in_pnas.rename(columns = {col: col.split("_x")[0] for col in extant_in_pnas.columns})
#         Extract.df2csv(extant_in_pnas,
#                        path = csv_dataset_out_path)

#         A_in_B_config = CSVDatasetConfig(full_name = csv_dataset_pathname,
#                                          data_path = csv_dataset_out_path,
#                                          subset_key = "all")
#         A_in_B_config.save(csv_dataset_config_out_path)

#         csv_dataset_pathname = f"{csv_config_B.full_name}_in_{csv_config_A.full_name}"
#         csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
#         csv_dataset_config_out_path = os.path.join(out_dir,f"B_in_A-CSVDataset-config.yaml")
#         columns = [*[col for col in dataset_A_composed_B.columns if col.endswith("_y")], *["catalog_number"]]
#         pnas_in_extant = dataset_A_composed_B.reset_index()[columns].sort_values("catalog_number")
# #         pnas_in_extant = pnas_in_extant.loc[:,[*list(SampleSchema.keys())]]
#         pnas_in_extant = pnas_in_extant.rename(columns = {col: col.split("_y")[0] for col in pnas_in_extant.columns})
#         Extract.df2csv(pnas_in_extant,
#                        path = csv_dataset_out_path)
#         B_in_A_config = CSVDatasetConfig(full_name = csv_dataset_pathname,
#                                          data_path = csv_dataset_out_path,
#                                          subset_key = "all")
#         B_in_A_config.save(csv_dataset_config_out_path)


#     out_dir = os.path.join(output_dir, full_name)
#     os.makedirs(out_dir, exist_ok=True)

#     inputs_dir = os.path.join(out_dir, "inputs")
# #     os.makedirs(inputs_dir, exist_ok=True)
#     os.makedirs(os.path.join(inputs_dir, "A"), exist_ok=True)
#     os.makedirs(os.path.join(inputs_dir, "B"), exist_ok=True)
#     shutil.copyfile(csv_cfg_path_A, os.path.join(inputs_dir, "A", Path(csv_cfg_path_A).name))
#     shutil.copyfile(csv_cfg_path_B, os.path.join(inputs_dir, "B", Path(csv_cfg_path_B).name))

#     csv_dataset_pathname = f"{full_name}-full_dataset"
#     csv_dataset_out_path = os.path.join(out_dir, csv_dataset_pathname + ".csv")
#     csv_dataset_config_out_path = os.path.join(out_dir,f"CSVDataset-config.yaml")

#     Extract.df2csv(dataset_A_composed_B,
#                    path = csv_dataset_out_path)

#     A_composed_B_config = CSVDatasetConfig(full_name = full_name,
#                                         data_path = csv_dataset_out_path,
#                                         subset_key = "all")
#     A_composed_B_config.save(csv_dataset_config_out_path)

#     print(f"[FINISHED] DATASET: {full_name}")
#     print(f"Newly created dataset assets located at:  {out_dir}")

#     if composition == '-':
#         dataset_A_composed_B = CSVDataset.from_config(A_composed_B_config)
#         return dataset_A_composed_B, A_composed_B_config

#     if composition == 'intersection':
#         return (extant_in_pnas, pnas_in_extant), A_composed_B_config
