#!/bin/bash

# Script: generate_all_data_multi-res-thresh.sh

# Location: "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_all_data_multi-res-thresh.sh"

# Created by: Jacob A Rose
# Created on: Wednesday Sept 22nd, 2021

"""
Description:

    - Run this script with no arguments in order to generate all single-resolution datasets for all base datasets. Then, for each resolution, produce all thresholded-versions as symlink trees.

    Datasets:
        - Extant_Leaves
        - General_Fossil
        - Florissant_Fossil

    Resolutions:
        - 512
        - 1024
        - 1536
        - 2048


resize

create_multithresh_symlink_tree

clean

make_catalogs

"""

# Run all 3 datasets for all 4 resolutions in serial
echo "Initiating generate_multires_images.py"
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multires_images.py" --dataset_name "all" --run-all

# Clean, & create, all symlink dirs for all thresholds and all datasets.
echo "Initiating generate_multithresh_symlink_trees.py"
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/generate_multithresh_symlink_trees.py" --task "clean+create" -data "all" -a


echo "Initiating clean_ipynb_ckpts.sh"
bash "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/clean_ipynb_ckpts.sh"



echo "Initiating make_catalogs.py"
python "/media/data/jacob/GitHub/lightning-hydra-classifiers/lightning_hydra_classifiers/data/utils/make_catalogs.py" --all

echo "FINISHED EVERYTHING SUCCESSFULLY. Go take a walk or something."
