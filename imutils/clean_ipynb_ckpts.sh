#!/bin/bash
# Cleanup all .ipynb_checkpoints littering the leavesdb-v1.0 dataset on disk.

export DEFAULT_ROOT="/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images"

export IMAGE_ROOT=${1:-$DEFAULT_ROOT}

echo "1st arg = $1"
echo "IMAGE_ROOT=$IMAGE_ROOT"

echo "Cleaning .ipynb_ckpt leftovers recursively starting at root: $IMAGE_ROOT"

rm -rvf `find $IMAGE_ROOT -type d -name .ipynb_checkpoints`

echo "Finished cleaning"
